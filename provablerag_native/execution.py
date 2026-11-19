from copy import deepcopy
from pathlib import Path
import time

from .budget import BudgetExhausted
from .compiler import compile_operations, operation_identity
from .config import Config
from .checked_computation import inputs_available, run as run_checked_computation
from .rule_execution import apply_rule
from .planning import _resolve, ground_axiom_facts
from .util import append, canonical, closure, digest, ground, is_ground, unify, write


def checked(operation, bindings, supplied_facts=None, question=None):
    return run_checked_computation(operation, bindings, supplied_facts, question)


class Executor:
    def __init__(self, spec, candidates, prover, acquirer, out_dir, config=None, *, proposal_only=False):
        self.spec, self.candidates = deepcopy(spec), deepcopy(candidates)
        self.prover, self.acquirer = prover, acquirer
        self.proposal_only = proposal_only
        self.node_formulas = {node["node_id"]: node["formula"] for node in self.spec["nodes"]}
        self.root = Path(out_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = config or Config()
        self.theory_hash = digest(spec["theory"])
        self.answer_hash = digest(spec["answer"])
        self.candidate_hash = digest(self.candidates)
        self.plans, self.solver_calls = [], 0
        self.solver_requests, self.compatibility_checks = 0, 0
        self.solver_seconds = 0.0
        self.prover.budget_limit_seconds = self.config.solver_total_seconds
        self.operations = 0
        self.blocked_resources = set()
        cumulative_counter = getattr(self.acquirer, "cumulative_search_count", None)
        physical_counter = getattr(self.acquirer, "physical_search_count", None)
        self.search_calls = (cumulative_counter() if callable(cumulative_counter)
                             else physical_counter() if callable(physical_counter) else 0)
        self.physical_search_calls = 0
        self.retrieval_attempts = 0
        self.acquirer.max_search_calls = self.config.max_search_calls
        self.events = 0
        self.accepted_bank = []
        self.axiom_facts=ground_axiom_facts(spec)
        self.mutex = [(edge["source"], edge["target"]) for edge in self.spec["relations"].get("edges", [])
                      if edge["kind"] == "mutex"]

    def event(self, kind, **data):
        self.events += 1
        append(self.root / "events.jsonl", {"event": self.events, "kind": kind, **data})

    def prove(self, method, *args):
        self.assert_fixed()
        if method == "compatible":
            self.compatibility_checks += 1
            result = self.prover.compatible(*args, f"compatible_{self.compatibility_checks:05d}")
            self.assert_fixed()
            self.event("compatibility", result=result)
            return result
        self.solver_requests += 1
        refresh = getattr(self.prover, "refresh_partial_reservations", None)
        if callable(refresh):
            refresh()
        if self.solver_budget_occupancy_seconds >= self.config.solver_total_seconds:
            result = {"status": "TIMEOUT", "proof_found": False, "solver_started": False,
                      "compile_status": "not_attempted", "error": "question_solver_budget",
                      "unknown_wall_reservation_seconds": getattr(self.prover, "unknown_wall_reservation_seconds", 0.0),
                      "solver_timing_complete": getattr(self.prover, "solver_timing_complete", True)}
            self.event("solver", method=method, result=result)
            return result
        remaining = self.config.solver_total_seconds - self.solver_budget_occupancy_seconds
        old_wall = getattr(self.prover, "wall_seconds", self.config.solver_wall_seconds)
        self.prover.wall_seconds = min(old_wall, self.config.solver_wall_seconds, remaining)
        before_calls = getattr(self.prover, "calls", None)
        before_wall = getattr(self.prover, "total_wall_seconds", None)
        start = time.monotonic()
        try:
            result = getattr(self.prover, method)(*args, f"{method}_{self.solver_requests:05d}")
        finally:
            self.prover.wall_seconds = old_wall
        after_calls = getattr(self.prover, "calls", None)
        after_wall = getattr(self.prover, "total_wall_seconds", None)
        actual_calls = (after_calls - before_calls if before_calls is not None and after_calls is not None
                        else int(bool(result.get("solver_started"))))
        self.solver_calls += actual_calls
        duration = (after_wall - before_wall if before_wall is not None and after_wall is not None
                    else result.get("duration_monotonic", time.monotonic() - start) if actual_calls else 0.0)
        self.solver_seconds += duration
        self.assert_fixed()
        self.event("solver", method=method, result=result)
        return result

    @property
    def solver_budget_occupancy_seconds(self):
        return self.solver_seconds + getattr(self.prover, "unknown_wall_reservation_seconds", 0.0)

    def assert_fixed(self):
        if digest(self.spec["theory"]) != self.theory_hash:
            raise RuntimeError("fixed_theory_mutated")
        if digest(self.spec["answer"]) != self.answer_hash:
            raise RuntimeError("fixed_answer_definition_mutated")
        if digest(self.candidates) != self.candidate_hash:
            raise RuntimeError("fixed_candidate_pool_mutated")

    def prepare(self):
        write(self.root / "spec.json", self.spec)
        write(self.root / "candidates.json", self.candidates)
        for candidate in self.candidates:
            proof = self.prove("planning", self.spec, candidate)
            compilation = None
            if proof.get("proof_found") and proof.get("compile_status") == "compiled":
                compilation = compile_operations(self.spec, candidate, proof)
            eligible = bool(compilation and compilation["status"] == "compiled")
            operations = {**compilation["operations"], **compilation["rule_input_operations"]} if eligible else {}
            active = set(compilation["node_dependencies"]) | set(operations) if eligible else set()
            plan = {"candidate": candidate, "proof": proof, "eligible": eligible, "operations": sorted(operations),
                    "operation_specs": operations, "proof_compilation": compilation,
                    "operation_delta": {"added_required_suppliers": sorted(set(operations) - set(candidate["operations"])),
                                        "removed_unused": sorted(set(candidate["operations"]) - set(operations))},
                    "active_nodes": sorted(active), "depth": proof.get("depth")}
            self.plans.append(plan)
        write(self.root / "plans.json", self.plans)

    def route_compatible(self, node_ids, facts):
        active = set(node_ids) | set(facts)
        return not any(left in active and right in active for left, right in self.mutex)

    def plan_nodes(self, plan):
        if "active_nodes" in plan:
            return set(plan["active_nodes"])
        candidate = plan["candidate"]
        return set(candidate.get("producer_map", {})) | set(plan["operations"]) | set(candidate.get("goal_node_ids", []))

    def rank(self, facts, excluded_nodes=()):
        return sorted((p for p in self.plans if p["eligible"]
                       and not self.plan_nodes(p) & set(excluded_nodes)
                       and self.route_compatible(self.plan_nodes(p), facts)),
                      key=lambda p: (p["depth"], p["candidate"]["stable_order"]))

    def effective_bindings(self, plan, facts, bindings):
        aliases=plan["proof_compilation"].get("bindings",{})
        concrete={}
        for slot,value in bindings.items():
            term=_resolve(slot,aliases)
            if term.startswith('?'):
                if term in concrete and concrete[term]!=value:return None
                concrete[term]=value
            elif term!=value:
                return None
        for slot in aliases:
            term=_resolve(slot,aliases)
            if not term.startswith('?') or term in concrete:
                concrete[slot]=concrete.get(term,term)
        for oid,op in plan["operation_specs"].items():
            if oid in facts:
                concrete=unify(op["formula"],facts[oid]["formula"],concrete)
                if concrete is None:return None
        return concrete

    def supplied(self, facts, bindings):
        records = dict(facts)
        for given in self.spec["theory"].get("given_facts", []):
            records.setdefault(given["id"], {"fact_id": given["id"], "formula": deepcopy(given["formula"]),
                "given": True, "scope": given.get("scope", "static"), "source_text": given.get("source_text", ""),
                "premise_ids": [], "evidence_ids": [], "theory_hash": self.theory_hash})
        for fid,(index,formula) in self.axiom_facts.items():
            records[fid]={"fact_id":fid,"formula":formula,"given":True,"scope":"static","axiom_index":index,
                          "premise_ids":[],"evidence_ids":[],"theory_hash":self.theory_hash}
        return records

    @staticmethod
    def operation_inputs(operation, supplied, bindings):
        records = dict(supplied)
        for item in operation.get("derived_inputs", []):
            ref = item["node_id"]
            given_id = item.get("given_id")
            if given_id is not None:
                source = records.get(given_id)
                if source is None or not source.get("given"):
                    return None
                records[ref] = source
            else:
                source = records.get(ref)
                if (source is None or not source.get("rule_execution")
                        or source["rule_id"] not in item["rule_ids"]):
                    return None
            if (source.get("scope", "static") != operation.get("scope", "static")
                    or unify(item["formula"],source["formula"],bindings) is None):
                return None
        return records

    def reuse_bank(self, state, branch_id):
        facts, bindings, feedback = (state[k] for k in ("facts", "bindings", "feedback"))
        excluded_nodes = set(state.get("excluded_nodes", []))
        nodes = {n["node_id"]: n for n in self.spec["nodes"]}
        changed = True
        while changed:
            changed = False
            supplied = self.supplied(facts, bindings)
            groups = {}
            for record in self.accepted_bank:
                oid = record["fact_id"]
                if oid in facts or oid in excluded_nodes or oid not in nodes or record.get("theory_hash") != self.theory_hash:
                    continue
                if not self.route_compatible([oid], facts):
                    continue
                if record.get("scope", "static") != nodes[oid].get("scope", "static"):
                    continue
                updated = unify(nodes[oid]["formula"], record["formula"], bindings)
                if updated is None:
                    continue
                excluded = False
                for entry in feedback.values():
                    previous = entry.get("previous", {})
                    previous_atom = previous.get("formula")
                    relevant = entry.get("operation_id") == oid
                    if not entry.get("operation_id") and previous_atom:
                        relevant = unify(nodes[oid]["formula"], previous_atom) is not None
                    blocked_atoms=entry.get("alternative_atoms",[])
                    blocked=(record["formula"] in blocked_atoms if blocked_atoms else
                             record.get("value") in entry.get("exclude_values", []))
                    if relevant and blocked:
                        excluded = True
                        break
                if excluded:
                    continue
                dependencies = set(record.get("input_origins", {}))
                sources = dict(supplied)
                for alias, origin in record.get("input_origins", {}).items():
                    if origin.get("given") and origin.get("fact_id") in supplied:
                        sources[alias] = supplied[origin["fact_id"]]
                if not set(record.get("input_fact_ids", [])) <= set(facts) or not dependencies <= set(sources):
                    continue
                expected = record.get("input_fact_formulas", {})
                if any(dep not in expected or canonical(sources[dep]["formula"]) != canonical(expected[dep])
                       for dep in dependencies):
                    continue
                origins = record.get("input_origins", {})
                if any(sources[dep].get('scope','static') != origins[dep].get('scope','static') for dep in dependencies):
                    continue
                current_origins={dep:self.fact_origin(sources[dep]) for dep in dependencies}
                groups.setdefault(oid, {}).setdefault(canonical(record["formula"]), (record, current_origins))
            for oid, alternatives in groups.items():
                ambiguous_routes = [other for left, right in self.mutex for other in
                                    ([right] if left == oid else [left] if right == oid else [])
                                    if other in groups]
                if ambiguous_routes:
                    self.event("bank_reuse_mutex_alternatives", branch=branch_id, operation=oid,
                               conflicting_nodes=ambiguous_routes)
                    continue
                if len(alternatives) != 1:
                    self.event("bank_reuse_ambiguous", branch=branch_id, operation=oid,
                               candidates=[r[0]["formula"] for r in alternatives.values()])
                    continue
                record, current_origins = next(iter(alternatives.values()))
                if not self.route_compatible([oid], facts):
                    continue
                updated = unify(nodes[oid]["formula"], record["formula"], bindings)
                if updated is None:
                    continue
                compatible = self.prove("compatible", self.spec, {**facts, oid: record})
                if compatible.get("status") != "compatible":
                    continue
                restored = deepcopy(record)
                restored["bindings"]=updated
                restored["input_origins"]=current_origins
                restored["reuse"] = {"source_branch": record.get("source_branch"), "target_branch": branch_id,
                                    "original_input_origins":record.get("input_origins",{}),
                                    "reason": "same_ground_fact_and_compatible_input_provenance"}
                facts[oid] = restored
                bindings.clear()
                bindings.update(updated)
                append(self.root / "facts.jsonl", {"branch": branch_id, **restored})
                self.event("fact_reused", branch=branch_id, fact=restored)
                result = self.completion(facts, bindings, branch_id)
                if result:
                    return result
                changed = True

    @staticmethod
    def fact_instance_key(oid, record):
        return digest([oid,{k:v for k,v in record.items() if k!='reuse'}])

    @staticmethod
    def fact_origin(record):
        return {k:record.get(k) for k in ("fact_id","scope","given","rule_id","premise_ids","evidence_ids")}

    @staticmethod
    def bound_operation(operation, bindings):
        actual=deepcopy(operation)
        actual['formula']=ground(actual['formula'],bindings)
        for derived in actual.get('derived_inputs',[]):
            derived['formula']=ground(derived['formula'],bindings)
        return actual

    def remember_operation(self, oid, operation, record, bindings, branch_id):
        actual=self.bound_operation(operation,bindings)
        if actual['formula']!=record['formula']:
            raise RuntimeError('accepted_operation_formula_does_not_match_fact:'+oid)
        key=self.fact_instance_key(oid,record)
        self.event('accepted_operation_provenance',branch=branch_id,operation_id=oid,
                   fact_instance_key=key,operation=actual)

    def answer_candidates(self, facts, bindings, proposed=None):
        if proposed is not None:
            yield proposed
            return
        answer_predicate = self.spec["answer"]["predicate"]
        templates = [r["formula"] for r in closure(self.spec, facts)]
        templates.extend(ground(r["conclusion"], bindings) for r in self.spec["theory"]["admitted_rules"])
        seen = set()
        for atom in templates:
            if atom["predicate"] == answer_predicate and is_ground(atom) and canonical(atom) not in seen:
                seen.add(canonical(atom))
                yield atom
        discovery = self.prove("runtime", self.spec, facts, None)
        def constants(expr):
            if 'predicate' in expr:
                yield from (a for a in expr['arguments'] if not a.startswith('?'))
            elif expr.get('op')=='eq':
                yield from (expr[k] for k in ('left','right') if not expr[k].startswith('?'))
            else:
                for child in expr.values():
                    if isinstance(child,dict):
                        yield from constants(child)
                    elif isinstance(child,list):
                        for item in child:
                            if isinstance(item,dict):
                                yield from constants(item)
        named=list(discovery.get("named_witness_candidates", []))
        theory=self.spec['theory']
        for expr in [*(f['formula'] for f in facts.values()),*(g['formula'] for g in theory.get('given_facts',[])),
                     *(a for r in theory['admitted_rules'] for a in [r['conclusion'],*r['premises']]),*theory.get('axioms',[])]:
            named.extend(constants(expr))
        for value in dict.fromkeys(named):
            atom = {"predicate":answer_predicate,"arguments":[value]}
            if canonical(atom) not in seen:
                yield atom

    def completion(self, facts, bindings, branch_id, candidate_atom=None):
        attempts = 0
        for atom in self.answer_candidates(facts, bindings, candidate_atom):
            if len(atom["arguments"]) != 1 or not is_ground(atom):
                continue
            attempts += 1
            proof = self.prove("runtime", self.spec, facts, atom)
            if proof.get("concrete_goal_proved") and proof["answer_template"] == atom:
                break
        else:
            self.event("completion_check", branch=branch_id, facts=len(facts), concrete_candidates=attempts, proved=False)
            return None
        value = atom["arguments"][0]
        used = proof["used_premise_ids"]
        given_ids = {row["id"] for row in self.spec["theory"].get("given_facts", [])}
        given_sources = set(used) & given_ids
        rule_ids = set(proof["used_rule_ids"])
        axiom_ids=set(proof["used_axiom_indices"])
        support_ids = set()
        pending = list(set(used) & facts.keys())
        while pending:
            fid = pending.pop()
            if fid in support_ids:
                continue
            support_ids.add(fid)
            record = facts[fid]
            if record.get("rule_execution"):
                rule_ids.add(record["rule_id"])
            parents = set(record.get("input_fact_ids", [])) | set(record.get("premise_ids", []))
            parents.update(p["fact_id"] for p in record.get("input_origins", {}).values() if p.get("fact_id"))
            pending.extend((parents & facts.keys()) - support_ids)
            given_sources.update(parents & given_ids)
            axiom_ids.update(self.axiom_facts[parent][0] for parent in parents if parent in self.axiom_facts)
            given_sources.update((record.get("computation") or {}).get("given_input_source_ids", []))
        return {"answer_status": "answered", "final_answer": value,
                "proof_verified": True,
                "completion_basis": "concrete_prover9",
                "answer_formula": atom, "provenance": {"used_fact_ids": used,
                    "execution_input_fact_ids": sorted(support_ids),
                    "given_input_source_ids": sorted(given_sources),
                    "evidence_ids": sorted({e for fid in support_ids if fid in facts for e in facts[fid].get("evidence_ids", [])}),
                    "rule_ids": sorted(rule_ids),
                    "axiom_indices": sorted(axiom_ids),
                    "proof_path": proof.get("raw_output_path")}}

    def run(self):
        self.prepare()
        frontier = [{"facts": {}, "bindings": {}, "attempts": {}, "feedback": {}, "excluded_nodes": []}]
        branches = []
        for branch_id in range(self.config.max_binding_branches):
            if not frontier:
                break
            state = frontier.pop(0)
            facts, bindings, attempts, feedback = (state[x] for x in ("facts", "bindings", "attempts", "feedback"))
            budget_blocked = set()
            self.event("branch_start", branch=branch_id, bindings=bindings, fact_ids=list(facts))
            result = self.completion(facts, bindings, branch_id) or self.reuse_bank(state, branch_id)
            if result:
                return self.finish(result, branches + [state])
            while self.operations < self.config.max_acquisition_operations:
                if self.solver_budget_occupancy_seconds >= self.config.solver_total_seconds:
                    break
                ready = None
                supplied = self.supplied(facts, bindings)
                for plan in self.rank(facts, state.get("excluded_nodes", [])):
                    assignment=self.effective_bindings(plan,facts,bindings)
                    if assignment is None:
                        continue
                    for oid in plan["operations"]:
                        if oid in facts:
                            continue
                        if not self.route_compatible([oid], facts):
                            continue
                        op = plan.get("operation_specs", plan["candidate"]["operations"])[oid]
                        if not all(d in facts for d in op.get("requires", [])):
                            continue
                        inputs = self.operation_inputs(op, supplied, assignment)
                        if inputs is None:
                            continue
                        if op["kind"] == "checked":
                            if not inputs_available(op, assignment, inputs, self.spec["question"]):
                                continue
                        elif any(v not in assignment for v in op.get("required_bindings", [])):
                            continue
                        key = digest([operation_identity(self.bound_operation(op, assignment)),
                                      {d: supplied[d]["formula"] for d in op.get("requires", [])}])
                        limit = self.config.attempts_per_binding if op["kind"] == "retrieve" else 1
                        if key in budget_blocked or attempts.get(key, 0) >= limit:
                            continue
                        ready = (oid, op, key, inputs, assignment)
                        break
                    if ready:
                        break
                if not ready:
                    break
                oid, op, key, supplied, assignment = ready
                before = deepcopy(state)
                if op["kind"] == "retrieve":
                    before_responses = len(getattr(self.acquirer, "responses", ()))
                    physical_counter = getattr(self.acquirer, "physical_search_count", None)
                    cumulative_counter = getattr(self.acquirer, "cumulative_search_count", None)
                    before_search = physical_counter() if callable(physical_counter) else None
                    try:
                        candidate = self.acquirer.acquire(self.spec["question"], op, assignment, supplied, feedback.get(key))
                    except BudgetExhausted as exc:
                        candidate = {"status": "budget_blocked", "reason": str(exc)}
                        budget_blocked.add(key)
                        self.blocked_resources.add(str(exc))
                    finally:
                        delta = physical_counter() - before_search if before_search is not None else 0
                        self.physical_search_calls += delta
                        if callable(cumulative_counter):
                            self.search_calls = cumulative_counter()
                        else:
                            self.search_calls += delta
                    if self.search_calls > self.config.max_search_calls:
                        raise RuntimeError("physical_search_budget_overrun")
                    if candidate.get("status") == "budget_blocked":
                        spent = delta > 0 or len(getattr(self.acquirer, "responses", ())) > before_responses
                        self.event("operation_budget_blocked", branch=branch_id, operation=oid,
                                   producer=op["kind"], reason=candidate["reason"], partially_dispatched=spent,
                                   status="partial_operation" if spent else "not_dispatched_budget")
                        if not spent:
                            continue
                    self.retrieval_attempts += 1
                elif op["kind"] == "rule":
                    candidate = apply_rule(self.spec, op, assignment, supplied)
                else:
                    candidate = checked(op, assignment, supplied, self.spec["question"])
                self.operations += 1
                attempts[key] = attempts.get(key, 0) + 1
                if candidate.get("status") == "accepted":
                    atom = candidate.get("formula")
                    actual = unify(op["formula"], atom, assignment) if atom and is_ground(atom) else None
                    updated = None
                    if actual is not None:
                        updated = unify(self.node_formulas[oid], atom, bindings)
                    if updated is None:
                        candidate = {**candidate, "status": "rejected", "reason": "inconsistent_shared_binding"}
                    elif not self.route_compatible([oid], facts):
                        candidate = {**candidate, "status": "rejected", "reason": "route_mutex_conflict"}
                    else:
                        source_ids = (set(op.get("requires", [])) | set(candidate.get("premise_ids", []))
                                      | {item["node_id"] for item in op.get("derived_inputs", [])})
                        record = {"fact_id": oid, "formula": atom, "scope": op.get("scope", "static"),
                                  "evidence_ids": candidate.get("evidence_ids", []), "premise_ids": candidate.get("premise_ids", op.get("requires", [])),
                                  "value": candidate.get("value", ""), "bindings": updated, "theory_hash": self.theory_hash,
                                  "source_branch": branch_id,
                                  "selected_prerequisite_group": plan["proof_compilation"]["selected_prerequisite_groups"].get(oid),
                                  "input_fact_ids": list(op.get("requires", [])),
                                  "input_fact_formulas": {d: supplied[d]["formula"] for d in source_ids},
                                  "input_origins": {d:self.fact_origin(supplied[d]) for d in source_ids},
                                  "verification": candidate.get("verification", "independent_node_verifier"),
                                  "computation": candidate.get("computation")}
                        if op["kind"] == "rule":
                            record.update(rule_execution=True, rule_id=op["rule_id"])
                        compatible = self.prove("compatible", self.spec, {**facts, oid: record})
                        if compatible.get("status") != "compatible":
                            candidate = {**candidate, "status": "rejected", "reason": "theory_compatibility:" + compatible.get("status", "unknown")}
                        else:
                            if any(oid in pair for pair in self.mutex) and len(frontier) + branch_id + 1 < self.config.max_binding_branches:
                                alternative_route = deepcopy(before)
                                alternative_route["excluded_nodes"] = sorted(set(alternative_route.get("excluded_nodes", [])) | {oid})
                                frontier.append(alternative_route)
                                self.event("route_alternative_saved", branch=branch_id, excluded_node=oid,
                                           retained_fact_ids=list(alternative_route["facts"]))
                            if op["kind"] == "retrieve" and len(frontier) + branch_id + 1 < self.config.max_binding_branches:
                                old = before["feedback"].get(key, {})
                                values = list(dict.fromkeys([*old.get("alternative_values", []), candidate.get("value")]))
                                atoms = [*old.get("alternative_atoms", []), atom]
                                before["feedback"][key] = {"reason": "alternative_binding", "operation_id": oid,
                                    "exclude_values": values, "alternative_values": values, "alternative_atoms": atoms,
                                    "previous": candidate}
                                frontier.append(before)
                            facts[oid] = record
                            bindings.clear()
                            bindings.update(updated)
                            self.accepted_bank.append(record)
                            self.remember_operation(oid,op,record,actual,branch_id)
                            append(self.root / "facts.jsonl", {"branch": branch_id, **record})
                            self.event("fact_update", branch=branch_id, fact=record)
                            write(self.root / "checkpoint.json", {"status": "running", "branch": branch_id, "state": state, "frontier": frontier, "operations": self.operations, "search_calls": self.search_calls})
                            result = self.completion(facts, bindings, branch_id) or self.reuse_bank(state, branch_id)
                            if result:
                                self.event("acquisition", branch=branch_id, operation=oid, attempt=self.operations, candidate=candidate)
                                return self.finish(result, branches + [state])
                if candidate.get("status") != "accepted":
                    previous = feedback.get(key, {})
                    feedback[key] = {**previous,"reason": candidate.get("reason", candidate.get("status")), "operation_id": oid, "previous": candidate,
                                     "exclude_values": previous.get("exclude_values", []) + ([candidate["value"]] if candidate.get("value") else [])}
                self.event("acquisition", branch=branch_id, operation=oid, attempt=self.operations, candidate=candidate)
                self.assert_fixed()
            branches.append(deepcopy(state))
        exhausted_budget = (self.blocked_resources or self.operations >= self.config.max_acquisition_operations
                            or self.search_calls >= self.config.max_search_calls
                            or self.solver_budget_occupancy_seconds >= self.config.solver_total_seconds)
        status = ("unresolved_budget" if exhausted_budget else "unresolved_no_verified_plan"
                  if not any(p["eligible"] for p in self.plans) else "unresolved_exhausted")
        return self.finish({"answer_status": status, "final_answer": None}, branches)

    def finish(self, result, branches):
        self.assert_fixed()
        if self.proposal_only and branches:
            state = branches[-1]
            try:
                review = self.acquirer.propose_answer(self.spec["question"], state["facts"],
                                                      self.spec["theory"].get("given_facts", []))
            except BudgetExhausted as exc:
                self.blocked_resources.add(str(exc))
                review = {"status":"unknown","reason":str(exc)}
            confirmed = None
            if review["status"] == "accepted":
                proposed = review["proposal"]["answer"].strip()
                if result.get("proof_verified") and proposed in (result["final_answer"],result["answer_formula"]["arguments"][0]):
                    confirmed = result
                else:
                    atom = {"predicate":self.spec["answer"]["predicate"],"arguments":[proposed]}
                    if is_ground(atom):
                        confirmed = self.completion(state["facts"],state["bindings"],len(branches)-1,candidate_atom=atom)
            result = {**(confirmed or {"answer_status":"unresolved_proposal","final_answer":None}),
                      "proposal_review":review,"runner":"proposal_only_with_ground_proof"}
        result = {**result, "method": "ProvableRAG", "theory_hash": self.theory_hash,
                  "candidate_hash": self.candidate_hash, "blocked_resources": sorted(self.blocked_resources),
                  "acquisition_operations": self.operations,
                  "search_calls": self.search_calls, "solver_calls": self.solver_calls,
                  "physical_search_calls_this_process": self.physical_search_calls, "retrieval_attempts": self.retrieval_attempts,
                  "solver_requests": self.solver_requests, "compatibility_checks": self.compatibility_checks,
                  "new_physical_solver_calls": getattr(self.prover, "physical_calls", self.solver_calls),
                  "replayed_solver_calls": getattr(self.prover, "replayed_calls", 0),
                  "unknown_wall_reservation_seconds": getattr(self.prover, "unknown_wall_reservation_seconds", 0.0),
                  "solver_timing_complete": getattr(self.prover, "solver_timing_complete", True),
                  "solver_budget_occupancy_seconds": self.solver_budget_occupancy_seconds,
                  "solver_wall_seconds": self.solver_seconds, "branches_explored": len(branches)}
        write(self.root / "result.json", result)
        write(self.root / "checkpoint.json", {"status": "completed", "result": result, "branches": branches})
        return result
