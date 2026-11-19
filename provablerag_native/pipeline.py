from dataclasses import asdict
import hashlib
import json
import time
from pathlib import Path

from . import VERSION
from .budget import BudgetExhausted
from .config import Config
from .execution import Executor
from .token_counting import counter_for_config
from .util import append, digest, write


class ModelFormatError(ValueError):
    pass


class QuestionModel:
    def __init__(self, model, config, out_dir, namespace):
        self.model, self.profile, self.root, self.namespace = model, config, Path(out_dir), namespace
        self.config = model.config
        self.counter = getattr(model, "token_counter", None) or counter_for_config(model.config)
        self.tokens = {"planning": 0, "execution": 0}
        self.input_tokens = {"planning": 0, "execution": 0}
        self.output_tokens = {"planning": 0, "execution": 0}
        self.calls, self.responses = 0, {}
        self.shared_calls = 0

    def seed_preprocessing(self, receipt):
        usage = receipt["model_usage"]
        if usage["total_tokens"]["execution"] != 0 or usage["token_counter"]["identity"] != self.counter.identity:
            raise ValueError("invalid_shared_preprocessing_accounting")
        for name, target in (("total_tokens", self.tokens), ("input_tokens", self.input_tokens), ("output_tokens", self.output_tokens)):
            for phase in target:
                amount = usage[name][phase]
                if type(amount) is not int or amount < 0:
                    raise ValueError("invalid_shared_token_count")
                target[phase] = amount
        if any(self.tokens[p] != self.input_tokens[p]+self.output_tokens[p] for p in self.tokens):
            raise ValueError("shared_usage_does_not_reconcile")
        self.shared_calls = self.calls = usage["model_calls"]
        self.responses = {k: None for k in usage["request_keys"]}
        if self.tokens["planning"] > self.profile.planning_tokens or self.calls > self.profile.max_model_calls or len(self.responses) != self.calls:
            raise ValueError("shared_preprocessing_exceeds_frozen_limits")
        if self.profile.max_total_tokens is not None and sum(self.tokens.values()) > self.profile.max_total_tokens:
            raise ValueError("shared_preprocessing_exceeds_total_limit")

    def execution_allowance(self):
        remaining = self.profile.execution_tokens - self.tokens["execution"]
        if self.profile.max_total_tokens is not None:
            remaining = min(remaining, self.profile.max_total_tokens - sum(self.tokens.values()))
        return remaining, self.profile.max_model_calls - self.calls

    def generate(self, messages, *, request_key, stage, max_output_tokens=2048):
        bucket = "planning" if stage in {"graph", "formalization", "planning"} else "execution"
        limit = self.profile.planning_tokens if bucket == "planning" else self.profile.execution_tokens
        estimate, bound = self.counter.input_bound(messages)
        remaining = limit - self.tokens[bucket]
        available = remaining - bound
        total_remaining = None
        if self.profile.max_total_tokens is not None:
            total_remaining = self.profile.max_total_tokens - sum(self.tokens.values())
            available = min(available, total_remaining - bound)
        output = min(max_output_tokens, available)
        floor = 1
        wire = digest([stage, messages, output, asdict(self.model.config)])
        key = self.namespace + ":" + request_key + ":wire:" + wire
        if self.calls >= self.profile.max_model_calls or output < floor:
            if self.calls >= self.profile.max_model_calls:
                resource = "model_calls"
            elif remaining < bound + floor:
                resource = bucket + "_token_budget"
            else:
                resource = "total_token_budget"
            append(self.root / "budget.jsonl", {"request_key": key, "stage": stage, "bucket": bucket,
                "input_estimate": estimate, "input_bound": bound, "remaining": remaining,
                "total_remaining": total_remaining, "calls_used": self.calls, "resource": resource,
                "minimum_output": floor, "requested_output": max_output_tokens, "status": "not_dispatched_budget"})
            raise BudgetExhausted(resource)
        response = self.model.generate(messages, request_key=key, stage=stage, max_output_tokens=output)
        usage = response.usage
        if usage.input_tokens is None or usage.output_tokens is None:
            from provablerag.models import RunBlocked
            raise RunBlocked("provider_usage_missing_reconcile_before_more_requests")
        self.calls += 1
        self.tokens[bucket] += usage.input_tokens + usage.output_tokens
        self.input_tokens[bucket] += usage.input_tokens
        self.output_tokens[bucket] += usage.output_tokens
        self.responses[key] = asdict(response)
        append(self.root / "budget.jsonl", {"request_key": key, "stage": stage, "bucket": bucket,
            "input_estimate": estimate, "input_bound": bound, "output_reserved": output,
            "provider_usage": asdict(usage), "response_model": response.exact_model_id,
            "finish_reason": response.finish_reason, "actual_bucket_total": self.tokens[bucket]})
        if usage.input_tokens > bound or self.tokens[bucket] > limit or self.profile.max_total_tokens is not None and sum(self.tokens.values()) > self.profile.max_total_tokens:
            from provablerag.models import RunBlocked
            raise RunBlocked("official_token_bound_exceeded_preserve_and_recalibrate")
        return response

    def report(self):
        return {"model_calls": self.calls, "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "total_tokens": self.tokens, "end_to_end_tokens": sum(self.tokens.values()),
                "token_counter": self.counter.metadata(), "request_keys": list(self.responses),
                "shared_preprocessing_calls": self.shared_calls, "local_invocation_calls": self.calls-self.shared_calls,
                "cost_interpretation": "attributed provider usage; unique physical expenditure is in canonical billing journals"}


def run_question(question, model, retriever, solver_binary, out_dir, config=None, *, namespace=VERSION, fixed_spec=None, fixed_candidates=None, shared_preprocessing=None, proposal_only=False):
    from provablerag.datasets import Question
    from .formalization import build_spec, review_fixed_spec, SpecificationError
    from .planning import enumerate_candidates
    from .proving import PaperProver
    from .acquisition import EvidenceAcquirer
    config = config or Config()
    public = asdict(question) if hasattr(question, "__dataclass_fields__") else question
    public = asdict(Question.from_public(public))
    receipt = None
    if fixed_spec is None and (fixed_candidates is not None or shared_preprocessing is not None):
        raise ValueError("replay_requires_a_fixed_specification")
    if fixed_spec is not None:
        if fixed_candidates is not None and shared_preprocessing is None:
            raise ValueError("fixed_candidates_require_measured_shared_preprocessing_receipt")
        source_question = fixed_spec["question"]
        if source_question.get("id", public["id"]) != public["id"] or source_question["question"] != public["question"]:
            raise ValueError("fixed_spec_public_question_differs")
    if shared_preprocessing is not None:
        receipt = json.loads(Path(shared_preprocessing).read_bytes()) if isinstance(shared_preprocessing, (str, Path)) else shared_preprocessing
        expected = {"public_question_hash": digest(public), "spec_hash": digest(fixed_spec), "candidates_hash": digest(fixed_candidates), "model_config_hash": digest(asdict(model.config))}
        if any(receipt.get(k) != v for k, v in expected.items()):
            raise ValueError("fixed_candidate_receipt_mismatch")
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    runtime_hash = digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))})
    manifest = {"protocol": VERSION, "question": public, "config": asdict(config), "proposal_only":proposal_only,
                "namespace": namespace, "fixed_spec_hash": digest(fixed_spec) if fixed_spec is not None else None,
                "fixed_candidates_hash": digest(fixed_candidates) if fixed_candidates is not None else None,
                "shared_preprocessing_hash": digest(receipt) if receipt is not None else None,
                "model_config": asdict(model.config), "runtime_source_hash": runtime_hash,
                "solver_sha256": hashlib.sha256(Path(solver_binary).read_bytes()).hexdigest(),
                "retriever": {"version": retriever.version, "snapshot_hash": retriever.snapshot_hash}}
    manifest = json.loads(json.dumps(manifest))
    manifest_path = root / "question_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_bytes()) != manifest:
        raise ValueError("question_resume_manifest_changed")
    write(manifest_path, manifest)
    write(root / "public_question.json", public)
    write(root / "config.json", asdict(config))
    wrapper = QuestionModel(model, config, root, namespace + ":" + public["id"])
    if receipt is not None:
        wrapper.seed_preprocessing(receipt)
    executor = None
    start = time.monotonic()
    online_start = None
    try:
        if fixed_spec is None:
            spec = build_spec(public, wrapper, root / "formalization", config)
        else:
            spec = review_fixed_spec(public, fixed_spec, wrapper, root / "formalization", config)
        online_start = time.monotonic()
        if fixed_candidates is None:
            enumeration = {}
            candidates = enumerate_candidates(spec, config.max_candidates, audit=enumeration,
                                              max_expansions=config.max_planning_expansions)
        else:
            candidates = fixed_candidates
            enumeration = {"status":"fixed_candidates", "candidate_count":len(candidates)}
        write(root/"candidate_enumeration.json", {**enumeration, "candidate_limit":config.max_candidates,
                                                  "candidates_sha256":digest(candidates)})
        if fixed_spec is None:
            write(root / "shared_preprocessing.json", {"public_question_hash": digest(public), "spec_hash": digest(spec),
                "candidates_hash": digest(candidates), "model_config_hash": digest(asdict(model.config)), "model_usage": wrapper.report()})
        prover = PaperProver(solver_binary, root / "prover", wall_seconds=config.solver_wall_seconds)
        acquirer = EvidenceAcquirer(wrapper, retriever, root / "acquisition.jsonl", top_k=config.top_k,
                                    replay_from_start=True, max_search_calls=config.max_search_calls,
                                    compact_citations=config.compact_citations,
                                    node_formulas={n["node_id"]: n["formula"] for n in spec["nodes"]})
        executor = Executor(spec, candidates, prover, acquirer, root / "execution", config, proposal_only=proposal_only)
        result = executor.run()
    except BudgetExhausted as exc:
        result = {"answer_status": "unresolved_budget", "final_answer": None, "reason": str(exc)}
    except (SpecificationError, ModelFormatError, json.JSONDecodeError) as exc:
        result = {"answer_status": "formalization_unresolved" if executor is None else "unresolved_model_format", "final_answer": None, "reason": str(exc)}
    if executor is not None and "solver_calls" not in result:
        checkpoint = root / "execution/checkpoint.json"
        saved = json.loads(checkpoint.read_bytes()) if checkpoint.exists() else {}
        branches = saved.get("branches", [saved["state"]] if "state" in saved else [])
        result = executor.finish(result, branches)
    end = time.monotonic()
    result["phase_usage"] = {
        "graph_formalization_seconds": None if fixed_spec is not None else (online_start or end)-start,
        "after_graph_seconds": end-online_start if online_start is not None else 0.0,
        "after_graph_tokens": wrapper.tokens["execution"],
        "graph_formalization_tokens": wrapper.tokens["planning"],
        "fresh_graph": fixed_spec is None,
        "boundary": "Offline: build graph and serialize fixed theory. Online: candidate enumeration, proof search, evidence acquisition and completion; never include judge."}
    result.update({"question_id": public["id"], "model_usage": wrapper.report(), "method": "ProvableRAG",
                   "protocol": VERSION,
                   "budget_interpretation": "planning_and_execution_separately_bounded; both_charged_end_to_end"})
    if receipt is not None:
        result["shared_preprocessing_cost"] = {"status": "measured_usage_attributed", "receipt_sha256": digest(receipt), "measured_fresh_end_to_end": False,
                                             "warning": "cache replay latency is not a fresh end-to-end measurement"}
    if executor is not None:
        result.setdefault("acquisition_operations", executor.operations)
        result.setdefault("search_calls", executor.search_calls)
        result.setdefault("solver_wall_seconds", executor.solver_seconds)
    result.setdefault("proof_verified", False)
    write(root / "result.json", result)
    return result
