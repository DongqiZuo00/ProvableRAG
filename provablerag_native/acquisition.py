from copy import deepcopy
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path

from .util import append, digest, is_ground, plain, unify
from .budget import BudgetExhausted


from . import published_prompts

SHARED = "\n" + published_prompts.SHARED
QUERY = published_prompts.QUERY
PROPOSE = published_prompts.NODE_CANDIDATE
VERIFY = (
    'Independently check the actual proposal for the active retrieval node. The original '
    'sentences supplied under passages must establish the relation for this hop. They '
    'may come directly from this retrieval or from the provenance of a cited premise. '
    'Use cited accepted premises to resolve bindings and context. A premise atom, the '
    'question, the proposed value, or compatibility_only facts cannot by themselves '
    'establish a new retrieved relation. If the originals do not establish it, return unknown. '
    'Check factual support, entity identity, scope and compatibility with established '
    'facts, including direction, qualifiers and shared bindings. Map a supported proposal '
    'to the complete ordered ground arguments of its node. These arguments must faithfully '
    'express this very binding or factual statement. An alias, description or paraphrase '
    'is admissible when the cited evidence establishes the same meaning. Never substitute '
    'a different answer. Preserve fixed terms exactly; a fixed term does not by itself '
    'establish the relation. Ambiguous or insufficient evidence is unknown. '
    'Return {"status":"accepted|rejected|unknown","arguments":["ordered ground argument"],'
    '"support_supported":true,"identity_supported":true,"scope_supported":true,'
    '"compatibility_supported":true,"reason":"brief explanation of the evidence for this judgment"}.'
)
FACT_FIELDS = ("fact_id", "formula", "scope", "evidence_ids", "premise_ids", "rule_id")
CHECKS = ("support_supported", "identity_supported", "scope_supported", "compatibility_supported")
ANSWER_VERIFY = (
    'Independently verify the whole proposed answer using only the supplied question, '
    'question givens, verified nodes and cited original sentences. Check every requested '
    'relation, entity, qualifier and comparison. Check the proposed supporting facts and '
    'evidence triples against the supplied sources; a proposed triple is not evidence. '
    'Return {"status":"accepted|rejected|unknown",'
    '"support_supported":true,"identity_supported":true,"scope_supported":true,'
    '"compatibility_supported":true,"reason":"brief reason"}. Missing support is unknown.'
)


def ordered_slots(operation, original, bindings):
    slots = []
    for index, term in enumerate(operation["arguments"]):
        if term.startswith("?") and term not in bindings:
            slots.append({"open":term})
            continue
        fixed = bindings.get(term,term)
        slot = {"fixed":fixed}
        source = original["arguments"][index]
        if source.startswith("?") and bindings.get(source) == fixed:
            slot["bound_from"] = source
        slots.append(slot)
    return slots


def _question(question):
    q = plain(question)
    if not isinstance(q, dict) or not isinstance(q.get("question"), str) or not q["question"].strip():
        raise ValueError("public_question_required")
    return {k: q[k] for k in ("id", "question", "dataset") if k in q}


def _ids(value):
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        return None
    return list(dict.fromkeys(value))


def fact_wire(fact_id, record):
    if "formula" not in record:
        raise ValueError("fact_formula_required")
    formula = record["formula"]
    if (not isinstance(formula, dict) or not isinstance(formula.get("predicate"), str)
            or not isinstance(formula.get("arguments"), list)
            or any(not isinstance(v, str) or not v.strip() or v.startswith("?") for v in formula["arguments"])):
        raise ValueError("fact_atom_required")
    result = {"fact_id": fact_id, **{k: deepcopy(record[k]) for k in FACT_FIELDS if k in record and k not in {"fact_id", "formula"}}}
    result["formula"] = {"predicate": formula["predicate"], "arguments": list(formula["arguments"])}
    return result


class EvidenceAcquirer:
    version = "evidence_acquisition_v2"

    def __init__(self, model, retriever, log_path, top_k=5, replay_from_start=False, max_search_calls=None, compact_citations=False, node_formulas=None):
        if type(top_k) is not int or top_k < 1:
            raise ValueError("positive_top_k_required")
        if max_search_calls is not None and (type(max_search_calls) is not int or max_search_calls < 0):
            raise ValueError("nonnegative_search_cap_required")
        self.model, self.retriever = model, retriever
        self.compact_citations = compact_citations
        self.log_path, self.top_k = Path(log_path), top_k
        self.node_formulas = dict(node_formulas or {})
        self.responses = []
        self.replay_from_start = bool(replay_from_start)
        self.evidence_store = {}
        self.observed_sentences = set()
        self.sources = self.evidence_store
        self.passage_sentences = {}
        self.search_cache = {}
        self.search_calls = 0
        self.historical_search_calls = 0
        self._new_physical_searches = 0
        self.max_search_calls = max_search_calls
        self.query_history = {}
        self.recorded_queries = {}
        self.failed_candidates = {}
        if self.log_path.exists():
            for line in self.log_path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                if record.get("role") == "retrieval":
                    passages = self._register(record["passages"])
                    self.search_cache[self._search_key(record["query"])] = passages
                    self.historical_search_calls += int(record.get("search_calls", 1))
                    if record.get("query_context_hash"):
                        self.recorded_queries[record["query_context_hash"]] = (record["query"],record.get("cached_only",False))
                    if not self.replay_from_start:
                        self.query_history.setdefault(record["input_key"], []).append(record["query"])
                elif not self.replay_from_start and record.get("role") == "node_result" and record["result"]["status"] != "accepted":
                    failed = record.get("proposal")
                    if failed is not None:
                        self.failed_candidates.setdefault(record["input_key"], []).append(failed)

    def physical_search_count(self):
        return self.search_calls

    def cumulative_search_count(self):
        return self.historical_search_calls + self._new_physical_searches

    def _search_key(self, query):
        return digest([query, self.top_k, getattr(self.retriever, "snapshot_hash", None),
                       getattr(self.retriever, "version", None)])

    def _expand_evidence(self, evidence_ids, supplied=None):
        return list(dict.fromkeys(s for eid in evidence_ids
            for s in self.passage_sentences.get(eid, [eid]) if supplied is None or s in supplied))

    def _original_passages(self, sentence_ids):
        documents = {}
        for sid in sentence_ids:
            sentence = self.evidence_store[sid]
            pid = sentence["passage_id"]
            document = documents.setdefault(pid, {"id":pid, "title":sentence["title"], "sentences":[]})
            document["sentences"].append({"sentence_id":sid, "text":sentence["text"]})
        return list(documents.values())

    def _register(self, passages):
        normalized = []
        for passage in passages:
            pid, title = passage["id"], passage.get("title", "")
            if not isinstance(pid, str) or not pid or not isinstance(title, str):
                raise ValueError("invalid_original_passage")
            sentences = []
            for sentence in passage["sentences"]:
                sid, text = str(sentence["sentence_id"]), sentence["text"]
                if not sid.startswith(pid + ":"):
                    sid = pid + ":" + sid
                if not isinstance(text, str) or not text:
                    raise ValueError("empty_original_sentence")
                record = {"sentence_id": sid, "text": text, "title": title, "passage_id": pid}
                if sid in self.evidence_store and self.evidence_store[sid] != record:
                    raise ValueError("evidence_id_content_changed")
                self.evidence_store[sid] = record
                sentences.append({"sentence_id": sid, "text": text})
            normalized.append({"id": pid, "title": title, "sentences": sentences})
            self.passage_sentences[pid] = [s["sentence_id"] for s in sentences]
        return normalized

    @staticmethod
    def _messages(prompt, public):
        return [{"role": "system", "content": prompt + SHARED},
                    {"role": "user", "content": "PUBLIC INPUT\n" + json.dumps(public, ensure_ascii=False, separators=(",", ":"))}]

    def _ask(self, role, prompt, public, max_output_tokens):
        messages = self._messages(prompt, public)
        key = digest([self.version, role, messages])
        response = self.model.generate(messages, request_key="paper_acquire:" + key,
            stage="execution",
            max_output_tokens=max_output_tokens)
        self.responses.append(response)
        raw = asdict(response) if is_dataclass(response) else {"text": response.text}
        append(self.log_path, {"role": role, "request_key": key, "messages": messages, "response": raw})
        if getattr(response, "finish_reason", None) == "length":
            return None, "truncated_" + role
        try:
            value = json.loads(response.text)
        except (ValueError, TypeError):
            return None, "invalid_json_" + role
        if not isinstance(value, dict):
            return None, "object_required_" + role
        return value, None

    def _context(self, question, operation, bindings, facts):
        requires = list(operation.get("requires", operation.get("required_inputs", [])))
        derived = operation.get("derived_inputs", [])
        if not isinstance(derived, list) or any(not isinstance(item, dict)
                or not isinstance(item.get("node_id"), str) or not isinstance(item.get("formula"), dict)
                for item in derived):
            raise ValueError("invalid_selected_derived_inputs")
        for item in derived:
            node_id = item["node_id"]
            record = facts.get(node_id)
            admitted = record and (
                (record.get("rule_execution") and record.get("rule_id") in item.get("rule_ids", []))
                or (record.get("given") and record.get("fact_id") == item.get("given_id")))
            if (not admitted or not is_ground(record["formula"])
                    or unify(item["formula"], record["formula"], bindings) is None):
                raise ValueError("acquisition_unready_derived_premise:" + node_id)
            requires.append(node_id)
        requires = list(dict.fromkeys(requires))
        required = {k: fact_wire(k, facts[k]) for k in requires if k in facts}
        missing = set(requires) - set(required)
        if missing:
            raise ValueError("acquisition_unready_missing_premises:" + ",".join(sorted(missing)))
        needed = set(operation["formula"]["arguments"])
        for node_id in [operation["node_id"], *requires]:
            if node_id in self.node_formulas:
                needed.update(self.node_formulas[node_id]["arguments"])
        for item in derived:
            needed.update(item["formula"]["arguments"])
        public = {"question": _question(question)["question"],
                  "node": {k: deepcopy(operation[k]) for k in ("node_id", "need", "scope", "formula") if k in operation},
                  "ordered_slots": ordered_slots(operation["formula"],
                        self.node_formulas.get(operation["node_id"],operation["formula"]), bindings),
                  "bindings": {v: value for v, value in bindings.items() if v in needed},
                  "required_facts": required}
        return public

    @staticmethod
    def _semantic(verdict, checks=CHECKS):
        if verdict is None or verdict.get("status") not in {"accepted", "rejected", "unknown"}:
            return "unknown", "invalid_independent_verdict"
        status = verdict["status"]
        if status == "accepted" and any(verdict.get(k) is not True for k in checks):
            return "unknown", "independent_acceptance_missing_semantic_check"
        return status, str(verdict.get("reason", ""))

    def _verification_input(self, public, value, premises, passages):
        request = {key: public[key] for key in ("question", "node", "ordered_slots", "bindings")}
        request.update(proposal={"value": value},
            required_facts={fid: public["required_facts"][fid] for fid in premises},
            compatibility_only={fid: fact for fid, fact in public["required_facts"].items() if fid not in premises},
            passages=passages)
        return request

    def _proposal_input(self, public, passages):
        view = {**public, "passages": passages, "selected_mode": "retrieve"}
        if self.compact_citations:
            from .citations import proposal_view
            return proposal_view(view)
        return view, {}

    def _input_cost(self, prompt, data):
        return self.model.counter.input_bound(self._messages(prompt, data))[1]

    def _reserve(self, public, cached_query):
        available, calls = self.model.execution_allowance()
        needed_calls = 2 if cached_query else 3
        if calls < needed_calls:
            raise BudgetExhausted("model_calls")
        verifier_floor = self._input_cost(VERIFY, self._verification_input(public, "", [], [])) + 1
        proposal_floor = self._input_cost(PROPOSE, self._proposal_input(public, [])[0]) + 1
        query_cost = 0 if cached_query else self._input_cost(QUERY, public)
        minimum = verifier_floor + proposal_floor + query_cost + int(not cached_query)
        append(self.log_path, {"role":"operation_reservation", "remaining":available, "calls_left":calls,
            "needed_calls":needed_calls, "minimum":minimum, "admitted":available >= minimum})
        if available < minimum:
            raise BudgetExhausted("whole_retrieval_operation_tokens")
        return min(160, available - verifier_floor - proposal_floor - query_cost), verifier_floor

    def _fit_observation(self, public, passages, verifier_floor):
        available, _ = self.model.execution_allowance()
        input_allowance = available - verifier_floor - 1

        def fits(selected):
            return self._input_cost(PROPOSE, self._proposal_input(public, selected)[0]) <= input_allowance

        if fits(passages):
            return passages
        selected = []
        for passage in passages:
            kept = []
            for sentence in passage["sentences"]:
                candidate = {**passage, "sentences":[*kept, sentence]}
                if fits([*selected, candidate]):
                    kept.append(sentence)
            if kept:
                selected.append({**passage, "sentences":kept})
        shown = [s["sentence_id"] for p in selected for s in p["sentences"]]
        append(self.log_path, {"role":"operation_observation_window", "shown_sentences":len(shown),
            "retrieved_sentences":sum(len(p["sentences"]) for p in passages), "shown_ids":shown})
        return selected

    def acquire(self, question, operation, bindings, facts, feedback=None):
        public = self._context(question, operation, bindings, facts)
        input_key = digest([self.version, _question(question), public])
        history = self.query_history.setdefault(input_key, [])
        failed = self.failed_candidates.setdefault(input_key, [])
        public["verification_feedback"] = {"previous_queries": list(dict.fromkeys(history)),
            "failed_candidates": deepcopy(failed), "current": deepcopy(feedback or {})}
        prior = (feedback or {}).get("previous", {})
        first_alternative = bool(feedback and feedback.get("reason") == "alternative_binding")
        excluded = (feedback or {}).get("alternative_values", (feedback or {}).get("exclude_values", [])
                                       if first_alternative else [])
        excluded_atoms = (feedback or {}).get("alternative_atoms", [])
        if excluded or excluded_atoms:
            public["verification_feedback"] = {"previous_queries":list(dict.fromkeys(history)),
                "current": {"reason":feedback["reason"],"previous_value":prior.get("value")},
                "failed_candidates":[p.get("value") for p in failed]}
            public["alternative_binding"] = {"excluded_values":excluded, "excluded_atoms":excluded_atoms,
                "open_slots":[{"index":i,"variable":s["open"]} for i,s in enumerate(public["ordered_slots"]) if "open" in s],
                "instruction":"Fill the open slot with one different cited referent satisfying this same relation. "
                              "Use only citation handles from the current passages. Leave empty if no other witness."}

        def finish(status, reason, *, proposal=None, query=None, formula=None, evidence_ids=None, premise_ids=None, verdict=None, **metadata):
            result = {"status": status, "formula": formula, "value": proposal.get("value", "") if proposal else "",
                      "evidence_ids": evidence_ids or [], "premise_ids": premise_ids or [], "reason": reason,
                      "semantic_verification": "independent_context_model_judgment", "semantic_guarantee": False}
            result.update(metadata)
            if query is not None:
                result["query"] = query
            if verdict is not None:
                result["verification"] = verdict
            append(self.log_path, {"role": "node_result", "input_key": input_key, "operation": operation["node_id"],
                                   "proposal": proposal, "result": result})
            if status != "accepted" and proposal is not None:
                failed.append(deepcopy(proposal))
            return result

        query_context_hash = digest(public)
        recorded = self.recorded_queries.get(query_context_hash) if self.replay_from_start else None
        preferred = prior.get("query") if first_alternative and prior.get("status") == "accepted" else None
        preferred_cached = preferred in history and self._search_key(preferred) in self.search_cache
        search_cap_reached = (self.max_search_calls is not None
                              and self.cumulative_search_count() >= self.max_search_calls)
        previous_cached = next((q for q in reversed(history) if self._search_key(q) in self.search_cache), None)
        cached_query = preferred if preferred_cached else (previous_cached if search_cap_reached else None)
        if recorded:
            cached_query = recorded[0] if recorded[1] else None
        elif search_cap_reached and not self.search_cache:
            raise BudgetExhausted("physical_search")
        cached_only = cached_query is not None
        query_output, verifier_floor = self._reserve(public, cached_only)
        if cached_only:
            query = cached_query
        else:
            query_data, error = self._ask("query", QUERY, public, query_output)
            if error:
                return finish("unknown", error)
            query = query_data.get("query")
        if not isinstance(query, str) or not query.strip():
            return finish("unknown", "empty_query")
        query = query.strip()
        if recorded and query != recorded[0]:
            raise RuntimeError("replayed_query_differs_from_recorded_query")
        if query not in history:
            history.append(query)
        search_key = self._search_key(query)
        reused = search_key in self.search_cache
        if reused:
            passages, physical_calls = deepcopy(self.search_cache[search_key]), 0
        else:
            if self.max_search_calls is not None and self.cumulative_search_count() >= self.max_search_calls:
                append(self.log_path, {"role": "search_budget", "query": query,
                    "input_key": input_key, "max_search_calls": self.max_search_calls,
                    "cumulative_search_calls": self.cumulative_search_count(), "not_dispatched": True})
                return finish("unknown", "search_budget", query=query, budget_exhausted=True,
                              resource="physical_search", cumulative_search_calls=self.cumulative_search_count())
            search = self.retriever.search_record(query, self.top_k)
            physical_calls = int(search.get("physical_search_calls", search.get("search_calls", 1)))
            passages = self._register(search["passages"])
            self.search_cache[search_key] = deepcopy(passages)
        self.search_calls += physical_calls
        self._new_physical_searches += physical_calls
        append(self.log_path, {"role": "retrieval", "input_key": input_key, "query": query,
            "query_context_hash":query_context_hash, "cached_only":cached_only,
            "search_calls": physical_calls, "passages": passages,
            "application_replay": reused, "physical_search_increment": physical_calls,
            "retriever_version": getattr(self.retriever, "version", None),
            "snapshot_hash": getattr(self.retriever, "snapshot_hash", None)})
        passages = self._fit_observation(public, passages, verifier_floor)
        self.observed_sentences.update(s["sentence_id"] for p in passages for s in p["sentences"])
        input_evidence = {sid for fact in public["required_facts"].values()
                          for sid in fact.get("evidence_ids", []) if sid in self.evidence_store}
        supplied = input_evidence | {s["sentence_id"] for p in passages for s in p["sentences"]} | {p["id"] for p in passages}
        view, aliases = self._proposal_input(public, passages)
        if self.compact_citations:
            append(self.log_path, {"role": "proposal_citations", "input_key": input_key, "aliases": aliases})
        available, _ = self.model.execution_allowance()
        prop_output = min(512, available - self._input_cost(PROPOSE, view) - verifier_floor)
        if prop_output < 1:
            raise BudgetExhausted("whole_retrieval_operation_tokens")
        proposal, error = self._ask("proposal", PROPOSE, view, prop_output)
        if error:
            return finish("unknown", error, query=query)
        if self.compact_citations:
            references = _ids(proposal.get("evidence_ids"))
            if references is None or any(ref not in aliases and ref not in input_evidence for ref in references):
                return finish("rejected", "candidate_citation_not_supplied", proposal=proposal, query=query)
            proposal["evidence_ids"] = [aliases.get(ref, ref) for ref in references]
        value = proposal.get("value")
        if not isinstance(value, str) or not value.strip():
            return finish("unknown", "no_candidate_value", proposal=proposal, query=query)
        evidence, premises = _ids(proposal.get("evidence_ids")), _ids(proposal.get("premise_ids"))
        if evidence is None or premises is None or not (evidence or premises):
            return finish("unknown", "candidate_requires_supplied_citations", proposal=proposal, query=query)
        if not set(evidence) <= supplied or not set(premises) <= set(public["required_facts"]):
            return finish("rejected", "candidate_citation_not_supplied", proposal=proposal, query=query)
        proposed = {"value": value, "evidence_ids": evidence, "premise_ids": premises}
        if any(digest(old) == digest(proposed) for old in failed):
            return finish("unknown", "exhausted_candidate_not_repeated", proposal=proposed, query=query)
        cited_sentences = self._expand_evidence(evidence, supplied)
        inherited = self._expand_evidence(
            [eid for fid in premises for eid in public["required_facts"][fid].get("evidence_ids", [])])
        cited_sentences = list(dict.fromkeys([*cited_sentences, *(sid for sid in inherited if sid in self.evidence_store)]))
        if not cited_sentences:
            return finish("unknown", "retrieval_requires_original_evidence", proposal=proposed, query=query,
                          premise_ids=premises)
        verification_input = self._verification_input(public, value, premises, self._original_passages(cited_sentences))
        verdict, error = self._ask("verification", VERIFY, verification_input, 768)
        status, reason = self._semantic(verdict)
        if error:
            return finish("unknown", error, proposal=proposed, query=query)
        formula = None
        if status == "accepted":
            values = verdict.get("arguments")
            if (not isinstance(values, list) or len(values) != len(operation["formula"]["arguments"])
                    or any(not isinstance(v, str) or not v.strip() or v.startswith("?") for v in values)):
                status, reason = "unknown", "independent_ground_mapping_missing"
            else:
                mapped = {"predicate": operation["formula"]["predicate"], "arguments": values}
                if mapped in excluded_atoms:
                    status, reason = "rejected", "excluded_ground_atom_repeated"
                elif unify(operation["formula"], mapped, bindings) is None:
                    status, reason = "rejected", "independent_mapping_changes_fixed_binding"
                elif not is_ground(mapped):
                    status, reason = "unknown", "non_ground_independent_mapping"
                else:
                    formula = mapped
        return finish(status, reason, proposal=proposed, query=query, formula=formula,
                      evidence_ids=cited_sentences, premise_ids=premises, verdict=verdict)

    def attributed_usage(self):
        def total(field):
            values = [getattr(getattr(r, "usage", None), field, None) for r in self.responses]
            return sum(values) if all(v is not None for v in values) else None
        return {"input_tokens": total("input_tokens"), "output_tokens": total("output_tokens"),
                "reasoning_tokens": total("reasoning_tokens"), "model_calls": len(self.responses),
                "search_calls": self.search_calls, "cumulative_search_calls": self.cumulative_search_count(),
                "usage_mode": "all actual proposal and independent-verification responses; replay uses original usage, physical increments reported separately"}

    def propose_answer(self, question, facts, givens):
        verified_nodes = {fid:fact_wire(fid, record) for fid, record in facts.items()}
        source_ids = sorted(self.observed_sentences |
                            {sid for r in facts.values() for sid in self._expand_evidence(r.get("evidence_ids", []))})
        if any(sid not in self.evidence_store for sid in source_ids):
            return {"status":"unknown","reason":"missing_original_evidence"}
        observed = self._original_passages(source_ids)
        documents = {p["id"] for p in observed}
        public = {"question":_question(question)["question"], "verified_nodes":verified_nodes,
                  "question_givens":deepcopy(givens), "passages":observed}
        proposal, error = self._ask("answer_proposal", published_prompts.ANSWER, public, 512)
        if error or not isinstance(proposal.get("answer"),str) or not proposal["answer"].strip():
            return {"status":"unknown","reason":error or "empty_answer_candidate","proposal":proposal}
        references = _ids(proposal.get("evidence_ids"))
        supporting = proposal.get("supporting_facts")
        triples = proposal.get("evidences")
        if (references is None or not set(references) <= set(source_ids) | set(documents)
                or not isinstance(supporting, list) or not isinstance(triples,list)
                or any(not isinstance(t,list) or len(t)!=3 or not all(isinstance(v,str) for v in t) for t in triples)):
            return {"status":"rejected","reason":"invalid_answer_evidence","proposal":proposal}
        cited = set(self._expand_evidence(references, set(source_ids)))
        for entry in supporting:
            if not isinstance(entry,dict) or not isinstance(entry.get("sentence_ids"),list):
                return {"status":"rejected","reason":"invalid_answer_support","proposal":proposal}
            eid = entry.get("evidence_id")
            if not isinstance(eid,str) or eid not in documents and eid not in source_ids:
                return {"status":"rejected","reason":"answer_support_not_supplied","proposal":proposal}
            pid = eid if eid in documents else self.evidence_store[eid]["passage_id"]
            for index in entry["sentence_ids"]:
                if type(index) not in (int, str):
                    return {"status":"rejected","reason":"answer_support_not_supplied","proposal":proposal}
                sid = str(index) if str(index).startswith(pid + ":") else pid + ":" + str(index)
                if sid not in source_ids or eid != pid and sid != eid:
                    return {"status":"rejected","reason":"answer_support_not_supplied","proposal":proposal}
                cited.add(sid)
        verifier_input = {**public, "proposal":proposal, "passages":self._original_passages(sorted(cited))}
        verdict, error = self._ask("answer_verification", ANSWER_VERIFY, verifier_input, 512)
        status, reason = self._semantic(verdict)
        return {"status":"unknown" if error else status,"reason":error or reason,"proposal":proposal,"verification":verdict}
