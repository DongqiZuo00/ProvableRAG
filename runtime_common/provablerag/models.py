from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import hashlib
import json
import os
import time
import urllib.request
import urllib.error


class ModelBlocked(RuntimeError):
    pass


class RunBlocked(BaseException):
    pass


class UnknownCharge(ModelBlocked):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    endpoint: str
    exact_model_id: str
    api_key_env: str
    authorization_reference: str
    max_total_tokens: int = 16000
    max_requests: int = 32
    temperature: float | None = 0
    reasoning_effort: str | None = None
    timeout_seconds: float = 60
    output_limit_field: str = "max_completion_tokens"
    accepted_response_model_ids: tuple[str, ...] = ()
    thinking: str | None = None
    response_format: dict | None = None
    currency: str = "CNY"
    max_spend: float | None = None
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None
    pricing_source: str | None = None


@dataclass(frozen=True)
class UsageRecord:
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    cache_tokens: int | None
    source: str = "provider_response; missing fields remain null"

    @property
    def billable_tokens(self):
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class ModelResponse:
    text: str
    provider_request_id: str | None
    exact_model_id: str | None
    requested_model_id: str
    usage: UsageRecord
    finish_reason: str | None
    duration_monotonic: float


def parse_response(raw, requested_model, request_id, duration):
    usage = raw.get("usage") or {}
    detail = usage.get("completion_tokens_details") or {}
    prompt_detail = usage.get("prompt_tokens_details") or {}
    choice = raw["choices"][0]
    text = choice["message"].get("content")
    if not isinstance(text, str):
        raise ValueError("missing_text_response")
    return ModelResponse(text, request_id or raw.get("id"), raw.get("model"), requested_model,
                         UsageRecord(usage.get("prompt_tokens"), usage.get("completion_tokens"),
                                     detail.get("reasoning_tokens"), prompt_detail.get("cached_tokens", usage.get("prompt_cache_hit_tokens"))),
                         choice.get("finish_reason"), duration)


class ChatModel:
    def __init__(self, config: ModelConfig, journal_path):
        self.config = config
        self.journal_path = Path(journal_path)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)

    def _records(self):
        if not self.journal_path.exists():
            return []
        return [json.loads(line) for line in self.journal_path.read_text(encoding="utf-8").splitlines() if line]

    def _append(self, record):
        with self.journal_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _billing_contract(self):
        cfg=self.config
        return {"provider": cfg.provider, "endpoint": cfg.endpoint, "currency": cfg.currency,
                "input_price_per_million": cfg.input_price_per_million,
                "output_price_per_million": cfg.output_price_per_million,
                "pricing_source": cfg.pricing_source, "max_spend": cfg.max_spend}

    def _completed_cost(self, intent, terminal):
        usage=terminal["response"]["usage"]
        if usage["input_tokens"] is None or usage["output_tokens"] is None:return None
        return (usage["input_tokens"]*self.config.input_price_per_million
                + usage["output_tokens"]*self.config.output_price_per_million)/1_000_000

    def generate(self, messages, *, request_key, stage, max_output_tokens=2048):
        cfg = self.config
        if not cfg.provider or not cfg.exact_model_id or "REQUIRED" in cfg.exact_model_id:
            raise ModelBlocked("blocked_model_identity")
        if not cfg.endpoint.startswith("https://") or not cfg.authorization_reference:
            raise ModelBlocked("blocked_endpoint_or_spending_authorization")
        if (cfg.max_spend is None or cfg.max_spend <= 0 or cfg.input_price_per_million is None
                or cfg.output_price_per_million is None or not cfg.pricing_source):
            raise ModelBlocked("blocked_pricing_or_monetary_cap")
        if cfg.input_price_per_million < 0 or cfg.output_price_per_million < 0:
            raise ModelBlocked("invalid_negative_price")
        if cfg.output_limit_field not in {"max_tokens", "max_completion_tokens"}:
            raise ModelBlocked("unsupported_output_limit_field")
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        payload = {"model": cfg.exact_model_id, "messages": messages, cfg.output_limit_field: max_output_tokens}
        reasoning_enabled = cfg.reasoning_effort not in (None, "none")
        if cfg.temperature is not None and not (cfg.exact_model_id.startswith("gpt-") and reasoning_enabled):
            payload["temperature"] = cfg.temperature
        if cfg.reasoning_effort is not None:
            payload["reasoning_effort"] = cfg.reasoning_effort
        if cfg.thinking is not None:
            if cfg.thinking not in {"enabled", "disabled"}:
                raise ModelBlocked("invalid_thinking_mode")
            payload["thinking"] = {"type": cfg.thinking}
        if cfg.response_format:
            payload["response_format"] = cfg.response_format
        payload_hash = digest(payload)
        records = self._records()
        contract = self._billing_contract()
        if any(r.get("billing_contract", contract) != contract for r in records):
            raise ModelBlocked("journal_billing_contract_conflict")
        matched = [r for r in records if r["request_key"] == request_key]
        if matched:
            if any(r["payload_hash"] != payload_hash for r in matched):
                raise ModelBlocked("idempotency_key_payload_conflict")
            complete = [r for r in matched if r["status"] == "completed"]
            if complete:
                result = complete[-1]["response"]
                return ModelResponse(**{**result, "usage": UsageRecord(**result["usage"])})
            raise UnknownCharge("unknown_charge: reconcile provider request before any retry")
        intents = [r for r in records if r["status"] == "intent"]
        terminals = {r["request_key"]: r for r in records if r["status"] != "intent"}
        spent = 0
        spent_cost = Decimal(0)
        for intent in intents:
            terminal = terminals.get(intent["request_key"])
            if not terminal or terminal["status"] != "completed":
                raise UnknownCharge("unreconciled_request_in_journal")
            u = terminal["response"]["usage"]
            if u["input_tokens"] is None or u["output_tokens"] is None:
                raise ModelBlocked("usage_unavailable_requires_reconciliation")
            spent += u["input_tokens"] + u["output_tokens"]
            settled_cost=self._completed_cost(intent,terminal)
            if settled_cost is None:raise ModelBlocked("completed_cost_unavailable_requires_reconciliation")
            spent_cost += Decimal(str(settled_cost))
        reserve = len(json.dumps(messages, ensure_ascii=False).encode()) + 256 + max_output_tokens
        reserve_cost = ((reserve - max_output_tokens) * cfg.input_price_per_million + max_output_tokens * cfg.output_price_per_million) / 1_000_000
        if len(intents) >= cfg.max_requests or spent + reserve > cfg.max_total_tokens:
            raise ModelBlocked("unresolved_budget")
        if spent_cost + Decimal(str(reserve_cost)) > Decimal(str(cfg.max_spend)):
            raise ModelBlocked("unresolved_monetary_budget")
        key = os.environ.get(cfg.api_key_env)
        if not key:
            raise ModelBlocked("blocked_credentials:" + cfg.api_key_env)
        event = {"request_key": request_key, "payload_hash": payload_hash, "stage": stage,
                 "start_utc": datetime.now(timezone.utc).isoformat(), "status": "intent",
                 "provider": cfg.provider, "requested_model_id": cfg.exact_model_id,
                 "reservation_tokens": reserve, "reservation_source": "utf8_bytes_plus_256",
                 "reservation_cost_upper_bound": reserve_cost, "currency": cfg.currency,
                 "pricing_source": cfg.pricing_source, "pricing_cache_policy": "charge_all_input_at_cache_miss_rate_conservative",
                 "billing_contract": contract,
                 "authorization_reference": cfg.authorization_reference, "payload": payload}
        self._append(event)
        started = time.monotonic()
        request = urllib.request.Request(cfg.endpoint, data=json.dumps(payload).encode(), method="POST",
                                         headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        received = False
        raw_text = None
        raw = None
        rid = None
        try:
            with urllib.request.urlopen(request, timeout=cfg.timeout_seconds) as response:
                rid = response.headers.get("x-request-id")
                raw_text = response.read().decode("utf-8", errors="replace")
            sanitized = raw_text.replace(key, "[REDACTED_API_KEY]")
            self._append({**event, "status": "response_received", "provider_request_id": rid,
                          "raw_response_text": sanitized, "credential_echo_redacted": sanitized != raw_text,
                          "duration_monotonic": time.monotonic() - started})
            received = True
            raw = json.loads(sanitized)
            parsed = parse_response(raw, cfg.exact_model_id, rid, time.monotonic() - started)
            accepted = {cfg.exact_model_id, *cfg.accepted_response_model_ids}
            if parsed.exact_model_id not in accepted:
                self._append({**event, "status": "model_identity_mismatch", "raw_response": raw,
                              "duration_monotonic": time.monotonic() - started})
                raise ModelBlocked("model_identity_mismatch")
            completed={**event, "status": "completed", "response": asdict(parsed), "raw_response": raw,
                       "budget_overrun_tokens": max(0, spent + (parsed.usage.billable_tokens or 0) - cfg.max_total_tokens)}
            completed["cost_upper_bound"]=self._completed_cost(event,completed)
            self._append(completed)
            if parsed.usage.billable_tokens is not None and spent + parsed.usage.billable_tokens > cfg.max_total_tokens:
                raise ModelBlocked("unresolved_budget_provider_overrun")
            return parsed
        except ModelBlocked:
            raise
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace").replace(key, "[REDACTED_API_KEY]")
            self._append({**event, "status": "http_error_response", "http_status": exc.code,
                          "provider_request_id": exc.headers.get("x-request-id"), "raw_response_text": error_body,
                          "duration_monotonic": time.monotonic() - started})
            raise UnknownCharge("http_error_requires_reconciliation:" + str(exc.code)) from None
        except Exception as exc:
            usage = raw.get("usage") if isinstance(raw, dict) else None
            reason=getattr(exc,"reason",None)
            reason_errno=getattr(reason,"errno",None)
            transport_error={"transport_reason_type":type(reason).__name__ if reason is not None else None,
                             "transport_reason_errno":reason_errno if isinstance(reason_errno,int) and not isinstance(reason_errno,bool) else None}
            self._append({**event, "status": "known_response_parse_error" if received else "unknown_charge",
                          "provider_request_id": rid, "reported_usage": usage,
                          "error_type": type(exc).__name__,
                          **transport_error,
                          "duration_monotonic": time.monotonic() - started})
            raise UnknownCharge(("known_response_parse_error:" if received else "unknown_charge:") + type(exc).__name__) from None


class QuestionBudgetModel:
    def __init__(self, model, limit_tokens, initial_tokens=0, *, budget_path):
        if initial_tokens is None:
            raise ModelBlocked("initial_usage_unavailable")
        if limit_tokens < 0 or initial_tokens < 0:
            raise ValueError("invalid_question_budget")
        self.model = model
        self.limit_tokens = limit_tokens
        self.initial_tokens = initial_tokens
        self.budget_path = Path(budget_path)
        self.budget_path.parent.mkdir(parents=True, exist_ok=True)
        contract = {"status": "contract", "limit_tokens": limit_tokens, "initial_tokens": initial_tokens,
                    "paid_ledger": str(model.journal_path.resolve())}
        rows = self._read()
        if rows and rows[0] != contract:
            raise ModelBlocked("question_budget_contract_conflict")
        if not rows:
            self._append(contract)

    @property
    def config(self):
        return self.model.config

    def _read(self):
        return [json.loads(s) for s in self.budget_path.read_text(encoding="utf-8").splitlines() if s] if self.budget_path.exists() else []

    def _append(self, row):
        with self.budget_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def usage(self):
        rows = self._read()
        states = {r["request_key"]: r["status"] for r in rows if r["status"] in {"intent", "not_sent"}}
        keys = {k for k, state in states.items() if state == "intent"}
        paid_rows = self.model._records_for(keys) if hasattr(self.model, '_records_for') else self.model._records()
        paid_keys = {r["request_key"] for r in paid_rows}
        terminal = {r["request_key"]: r for r in paid_rows if r["status"] != "intent"}
        tokens = self.initial_tokens
        for key in keys:
            if key not in paid_keys:
                self._append({"status": "not_sent", "request_key": key, "reason": "no_underlying_durable_intent"})
                continue
            record = terminal.get(key)
            if not record or record["status"] != "completed":
                raise UnknownCharge("question_request_unreconciled:" + key)
            usage = record["response"]["usage"]
            if usage["input_tokens"] is None or usage["output_tokens"] is None:
                raise ModelBlocked("question_usage_unavailable")
            tokens += usage["input_tokens"] + usage["output_tokens"]
        return tokens

    def generate(self, messages, *, request_key, stage, max_output_tokens=2048):
        attribution_payload_hash = digest({"messages": messages, "stage": stage, "max_output_tokens": max_output_tokens,
                                           "requested_model_id": getattr(self.config, "exact_model_id", None)})
        if any(r.get("request_key") == request_key and r.get("payload_hash") not in (None, attribution_payload_hash) for r in self._read()):
            raise ModelBlocked("question_request_payload_conflict")
        spent = self.usage()
        states = {r["request_key"]: r["status"] for r in self._read() if r["status"] in {"intent", "not_sent"}}
        keys = {k for k, state in states.items() if state == "intent"}
        reserve = len(json.dumps(messages, ensure_ascii=False).encode()) + 256 + max_output_tokens
        if request_key not in keys:
            if spent + reserve > self.limit_tokens:
                raise ModelBlocked("unresolved_budget_question")
            self._append({"status": "intent", "request_key": request_key, "stage": stage,
                          "payload_hash": attribution_payload_hash,
                          "reservation_tokens": reserve, "reservation_source": "utf8_bytes_plus_256"})
        try:
            response = self.model.generate(messages, request_key=request_key, stage=stage, max_output_tokens=max_output_tokens)
        except ModelBlocked:
            request_rows = self.model._records_for([request_key]) if hasattr(self.model, '_records_for') else self.model._records()
            if not any(r["request_key"] == request_key for r in request_rows):
                self._append({"status": "not_sent", "request_key": request_key})
            raise
        if response.usage.billable_tokens is None:
            raise ModelBlocked("question_usage_unavailable")
        actual = self.usage()
        if actual > self.limit_tokens:
            self._append({"status": "provider_overrun", "request_key": request_key,
                          "actual_attributed_tokens": actual, "overrun_tokens": actual-self.limit_tokens})
            raise ModelBlocked("unresolved_budget_question_provider_overrun")
        return response
