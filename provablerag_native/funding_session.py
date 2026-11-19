from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time


LIVE_LEASES = {}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class HistoryAttestationSession:
    def __init__(self, registry, report_path=None):
        self.registry, self.report_path = registry, Path(report_path) if report_path else None
        self.phase, self.owner_pid = 'new', os.getpid()
        self.calls, self.retired_checks = 0, 0
        self.history = None
        self.full_checks = []
        self._mutex = threading.RLock()
        self._lease = None
        self.bound_pools = []

    def blocked(self, reason):
        raise self.registry.ModelBlocked(reason)

    @staticmethod
    def money(value):
        if isinstance(value, bool):
            raise ValueError('invalid_money')
        result = Decimal(str(value))
        if not result.is_finite() or result < 0:
            raise ValueError('invalid_money')
        return result

    @staticmethod
    def file_identity(path):
        value = os.stat(path)
        return value.st_dev, value.st_ino

    def head_signature(self, head):
        return {'generation': head['generation'], 'sha256': head['sha256'],
                'model_root': str(head['model_root']), 'search_root': str(head['search_root']),
                'record_path': str(head['record_path'])}

    def full_history(self, history):
        start = time.monotonic()
        count = 0
        for name, expected in history.items():
            path = self.registry.evidence_path(name)
            if expected is None:
                if path.exists() or path.is_symlink():
                    self.blocked('attestation_sealed_history_appeared:' + name)
            elif self.registry.file_hash(path) != expected:
                self.blocked('attestation_sealed_history_changed:' + name)
            count += 1
        self.full_checks.append({'phase': self.phase, 'history_entries': count,
                                 'actual_seconds': time.monotonic() - start, 'full_hashes': True})

    def initial_history(self, history):
        if self.phase != 'starting' or os.getpid() != self.owner_pid:
            self.blocked('attestation_initial_validation_outside_lease')
        self.full_history(history)
        if self.history is not None and digest(self.history) != digest(history):
            self.blocked('attestation_history_changed_during_startup')
        self.history = deepcopy(history)

    def freeze_finance(self, allocation, head):
        cap = self.money(allocation['user_total_cap_cny'])
        if cap != 500 or allocation.get('currency') != 'CNY':
            self.blocked('attestation_requires_existing_500_CNY_authorization')
        total = self.money(allocation['prior_budget_occupancy_upper_cny'])
        roles = {}
        for name, item in allocation['roles'].items():
            slots = item['slots']
            if type(slots) is not int or slots < 1:
                self.blocked('attestation_invalid_slot_count')
            per_slot = self.money(item['per_slot_cny'])
            if per_slot <= 0:
                self.blocked('attestation_invalid_slot_cap')
            roles[name] = {'slots': slots, 'per_slot_cny': str(per_slot)}
            total += slots * per_slot
        search = self.money(allocation['external_search']['allowance_cny'])
        total += search
        if total != self.money(allocation['combined_static_upper_cny']):
            self.blocked('attestation_static_allocation_math_mismatch')
        root = Path(head['model_root'])
        allocation_path = root / 'allocation.json'
        extras = []
        for path in sorted(root.glob('*/registration.json')):
            record = json.loads(path.read_bytes())
            amount = Decimal(0)
            for item in record['roles'].values():
                if type(item['slots']) is not int or item['slots'] < 1 or self.money(item['per_slot_cny']) <= 0:
                    self.blocked('attestation_invalid_extra_slot_cap')
                amount += item['slots'] * self.money(item['per_slot_cny'])
            if (record.get('version') != 'unallocated_buffer_reservation_v1'
                    or record.get('authorization_reference') != allocation.get('authorization_reference')
                    or record.get('parent_allocation_sha256') != self.registry.file_hash(allocation_path)
                    or self.money(record['prior_static_upper_cny']) != total
                    or self.money(record['cap_cny']) != amount
                    or self.money(record['combined_static_upper_cny']) != total + amount):
                self.blocked('attestation_unrecognized_extra_allocation')
            total += amount
            extras.append({'path': str(path), 'sha256': self.registry.file_hash(path), 'cap_cny': str(amount)})
        if total > cap:
            self.blocked('attestation_static_cap_exceeded')
        self.finance = {'prior_occupancy_upper_cny': allocation['prior_budget_occupancy_upper_cny'],
                        'main_roles': roles, 'external_search_allowance_cny': str(search),
                        'registered_extras': extras, 'combined_static_upper_cny': str(total),
                        'user_cap_cny': '500', 'new_allowance_cny': '0'}
        financial_paths = [Path(self.registry.paths()['registry']), Path(head['record_path']), allocation_path,
                           Path(head['search_root']) / 'allocation.json'] + [Path(e['path']) for e in extras]
        self.financial_files = {str(path): self.registry.file_hash(path) for path in financial_paths}
        self.extra_registration_paths = {item['path'] for item in extras}

    def check_fresh(self, allow_closing=False):
        if (self.phase != 'active' and not (allow_closing and self.phase == 'closing')) or os.getpid() != self.owner_pid:
            self.blocked('attestation_not_in_live_migration_lease')
        if self.file_identity(self.lease_path) != self.lease_identity:
            self.blocked('attestation_migration_lock_file_replaced')
        if self.head_signature(self.registry._chain_head()) != self.head:
            self.blocked('attestation_activation_changed')
        if {str(path) for path in Path(self.head['model_root']).glob('*/registration.json')} != self.extra_registration_paths:
            self.blocked('attestation_extra_allocation_set_changed')
        for name, expected in self.financial_files.items():
            if self.registry.file_hash(Path(name)) != expected:
                self.blocked('attestation_financial_document_changed:' + name)

    def attested_history(self, history):
        with self._mutex:
            self.check_fresh()
            if digest(history) != self.history_digest:
                self.blocked('attestation_history_manifest_changed')
            self.calls += 1

    def retired_check(self, request_key):
        with self._mutex:
            self.check_fresh()
            if hashlib.sha256(request_key.encode()).hexdigest() in self.retired:
                from provablerag.models import UnknownCharge
                raise UnknownCharge('retired_request_key_must_not_be_reissued')
            self.retired_checks += 1
            return self.original_retired(request_key)

    def report(self, status, error=None):
        if self.report_path is None:
            return
        value = {'version': 'session_history_attestation_v1', 'status': status, 'phase': self.phase,
                 'observed_utc': datetime.now(timezone.utc).isoformat(),
                 'head': getattr(self, 'head', None), 'finance': getattr(self, 'finance', None),
                 'history_digest': getattr(self, 'history_digest', None), 'full_history_checks': self.full_checks,
                 'per_request_attestation_checks': self.calls, 'per_request_retired_checks': self.retired_checks,
                 'history_mutation_detection': 'full start/end hashes; no continuous sealed-file stat checking',
                 'error_type': type(error).__name__ if error else None}
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

    def bind_pool(self, proxy):
        if self.phase != 'active':
            self.blocked('attestation_pool_bound_outside_live_session')
        self.bound_pools.append(proxy)

    def pools_quiescent(self):
        return all(pool.quiescent() for pool in self.bound_pools)

    def drain_pools(self):
        interruption = None
        for pool in self.bound_pools:
            try:
                drain_and_close(pool)
            except BaseException as error:
                interruption = interruption or error
        if interruption is not None:
            raise interruption

    def __enter__(self):
        if self.phase != 'new':
            self.blocked('attestation_session_not_reusable')
        self.original_history = self.registry._verify_history
        self.original_retired = self.registry.assert_not_retired_request
        self.lease_path = Path(self.registry.PACKAGE) / 'online_runs/.authorization_migration.lock'
        self._lease = self.registry.writer_lock(self.lease_path)
        self._lease.__enter__()
        LIVE_LEASES[id(self)] = self
        try:
            self.lease_identity = self.file_identity(self.lease_path)
            self.phase = 'starting'
            self.registry._verify_history = self.initial_history
            before = self.registry._chain_head()
            if before['generation'] < 4:
                self.blocked('attestation_requires_successor_allocation')
            allocation = self.registry.active_allocation(before['model_root'], 'model')
            after = self.registry._chain_head()
            if self.head_signature(before) != self.head_signature(after):
                self.blocked('attestation_activation_changed_during_startup')
            plan, audit = self.registry._transition_documents(after['generation'], after['record']['plan_sha256'])
            if self.history is None or digest(self.history) != digest(audit['history_files']):
                self.blocked('attestation_startup_history_not_verified')
            self.head = self.head_signature(after)
            self.history_digest = digest(self.history)
            self.retired = frozenset(audit['retired_request_key_hashes'])
            self.allocation = deepcopy(allocation)
            self.freeze_finance(allocation, after)
            self.phase = 'active'
            self.registry._verify_history = self.attested_history
            self.registry.assert_not_retired_request = self.retired_check
            self.report('open')
            return self
        except BaseException:
            error = sys.exc_info()
            self.registry._verify_history = self.original_history
            self.registry.assert_not_retired_request = self.original_retired
            self.phase = 'closed'
            self._lease.__exit__(*error)
            LIVE_LEASES.pop(id(self), None)
            raise

    def __exit__(self, exc_type, exc, traceback):
        failure = None
        try:
            self.phase = 'closing'
            deferred_interruption = None
            try:
                self.drain_pools()
            except BaseException as error:
                if not self.pools_quiescent():
                    raise
                deferred_interruption = error
            self.check_fresh(allow_closing=True)
            self.full_history(self.history)
            self.report('closed_interrupted' if deferred_interruption else 'closed', deferred_interruption or exc)
            failure = deferred_interruption
        except BaseException as error:
            failure = error
            self.report('failed', error)
        finally:
            if not self.pools_quiescent():
                self.phase = 'failed_lease_retained'
                self.report('failed_lease_retained', failure or exc)
            else:
                self.registry._verify_history = self.original_history
                self.registry.assert_not_retired_request = self.original_retired
                self.phase = 'closed'
                self._lease.__exit__(exc_type, exc, traceback)
                LIVE_LEASES.pop(id(self), None)
        if failure is not None:
            raise failure from exc
        if self.phase == 'failed_lease_retained':
            self.blocked('attestation_cleanup_interrupted_lease_retained')
        return False


class DrainingPool:
    def __init__(self, pool):
        from provablerag.models import ModelBlocked
        if pool.registered_allocation_hash is None:
            raise ModelBlocked('attestation_requires_registered_funded_pool')
        self._pool = pool
        self._condition = threading.Condition()
        self._active = 0
        self._closing = self._closed = False

    def __getattr__(self, name):
        return getattr(self._pool, name)

    def generate(self, *args, **kwargs):
        from provablerag.models import RunBlocked
        with self._condition:
            if self._closing:
                raise RunBlocked('attested_pool_is_draining')
            self._active += 1
        try:
            return self._pool.generate(*args, **kwargs)
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

    def close(self):
        interruption = None
        with self._condition:
            self._closing = True
            while self._active:
                try:
                    self._condition.wait(timeout=1)
                except BaseException as error:
                    interruption = interruption or error
            if not self._closed:
                self._pool.close()
                self._closed = True
        if interruption is not None:
            raise interruption

    def quiescent(self):
        with self._condition:
            return self._closing and self._closed and self._active == 0


def drain_and_close(proxy):
    interruption = None
    while True:
        try:
            proxy.close()
            if proxy.quiescent():
                break
        except BaseException as error:
            interruption = interruption or error
    if interruption is not None:
        raise interruption


@contextmanager
def attested_funded_pool(config, role, report_path=None):
    from provablerag import budget_registry
    from provablerag.funding import funded_pool
    with HistoryAttestationSession(budget_registry, report_path=report_path) as attestation:
        paid_root = Path(attestation.head['model_root'])
        pool = funded_pool(config, paid_root, role)
        if pool.registered_allocation_hash != budget_registry.digest(attestation.allocation):
            pool.close()
            attestation.blocked('attestation_registered_pool_allocation_mismatch')
        proxy = DrainingPool(pool)
        attestation.bind_pool(proxy)
        try:
            yield proxy
        finally:
            drain_and_close(proxy)
