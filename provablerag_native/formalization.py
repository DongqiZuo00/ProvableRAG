from copy import deepcopy
from dataclasses import asdict, is_dataclass
import json

from . import published_prompts
from .util import digest, write


class SpecificationError(ValueError):
    pass


SHARED = "\n" + published_prompts.SHARED
NODES_PROMPT = published_prompts.NODES + SHARED
RELATIONS_PROMPT = published_prompts.RELATIONS + SHARED

BRIDGE_PROMPT = '''Serialize the fixed graph and question as atomic node
formulas, a fixed background theory and a unary answer predicate. Preserve
the original request, node IDs, scopes, dependencies and goal options. Use
variables for unknowns and share them only for the same entity or value.
The theory may contain given facts and admitted inference rules. Do not put
facts to be retrieved, hypothetical outcomes or guessed answers in the theory.
Define the answer condition Q(a), not just its predicate name. If the answer
is a value projected from linked information needs, give the question-definition
rule: the joint premises with their shared variables imply Q(the requested
variable). Keep its antecedents conditional, never observations. If Q itself
already expresses that condition, no redundant projection rule is needed.
Declaring a new unary answer symbol without relating it to the question
does not define Q. This definition may refer to existing nodes without adding
a new graph node.
Emit a question-definition rule in theory.admitted_rules, or in the
top-level question_definition field using exactly the same rule schema;
both serializations are subject to the identical admission checks.
Prerequisite and exclusion edges constrain plans; neither supplies a fact.
Proposed logical edges are not themselves verified inference rules.
Serialize each justified direction of IMPLIES or EQUIVALENT as an
independently admitted scoped rule. Leave unsupported directions unused.
Return {"node_formulas":{"n1":{"predicate":"p","arguments":["?x"]}},
"theory":{"predicates":{"p":1,"answer":1},"given_facts":[],
"admitted_rules":[],"axioms":[]},"answer":{"predicate":"answer","constant_labels":{}},
"computations":{}}. Use the predicate names and arities that fit the question.
An atom has a predicate and ordered argument terms (?variable or literal).
Use a stable id for each given fact and admitted rule. Given facts must be
stated in the question. A given fact has a ground formula and source_text
quoting the text that states it. A rule has
premises and a conclusion. For a question definition, identify the rule as
source_kind question_definition; for an explicit conditional, question_explicit.
Quote the statement for an explicit conditional. For a definition, a
source_text rationale is optional. Use tested_rule_library only for a supplied
pinned rule. Keep the answer constant in its proved surface spelling.
List sources of extra axioms in theory.axiom_sources as records containing
index, source_kind, and source_text. Quote question_explicit axioms; explain
structural denials as question_definition. Declare predicate arities. Axioms
may include atoms, equality
(left/right), not (arg), and/or (args), implies (left/right), and forall/exists
(variables/body), keyed by op. Include only axioms supplied by or required
to define the question. Do not turn facts to retrieve into axioms.
No computation is required. If a proposed reason node uses a supported
primitive, identify it by node ID with kind and ordered inputs from its
selected predecessors or question givens. An input is a term or a position
in that node's head arguments, or {"node":"n2","position":1} for the
zero-based argument index in a prerequisite's formula. The named node must
actually supply this slot on the selected route. Keep a computation distinct
from an admitted rule. compare_date, compare_number and compare_year test
strict left < right. Operations equal and not_equal can compare complete
dates or numbers after normalization. Identical literals also compare equal.
Distinct text labels alone cannot establish an entity mismatch.
If the question cannot be expressed in this representation, return
{"status":"formalization_unresolved","reason":"..."}.
Preserve the original entities, the direction and roles of each relation, the
requested answer role, and each condition on time, comparison, negation or
quantification. If multiple premises refer to the same unknown, use the same
variable in those positions. If they may differ, keep the terms distinct.
For a property shared by several participants, require it for each of them,
joined on the same property. For an asymmetric relation, retain the ordered
roles from the question.
Read each proposed answer rule as a conditional. If its premises could all
hold while a necessary participant, relation or qualifier in the question
is missing, revise the rule before returning it.
Capture these constraints in the formulas. A graph step or static graph scope
does not encode an object-level time condition.
'''+SHARED

COMPUTATIONS = {
    "compare_date": {"arity": 2, "inputs": [0, 1], "input_types":["date","date"]},
    "compare_year": {"arity": 2, "inputs": [0, 1], "input_types":["year","year"]},
    "compare_number": {"arity": 2, "inputs": [0, 1], "input_types":["number","number"]},
    "year": {"arity": 2, "inputs": [0], "input_types":["date"]},
    "equal": {"arity": 2, "inputs": [0, 1], "input_types":["canonical_value","canonical_value"]},
    "not_equal": {"arity": 2, "inputs": [0, 1], "input_types":["canonical_value","canonical_value"]},
}
SOURCE_KINDS = {"question_definition", "question_answer_definition",
                "question_explicit", "tested_rule_library", "deterministic_contract",
                "checked_computation"}


def computation_input_terms(formula,computation,nodes=None):
    kind=computation.get('kind')
    if kind not in COMPUTATIONS:raise SpecificationError('unknown_computation_kind')
    declared=computation.get('inputs',COMPUTATIONS[kind]['inputs'])
    expected=len(COMPUTATIONS[kind]['inputs'])
    if not isinstance(declared,list) or len(declared)!=expected:
        raise SpecificationError('computation_input_count:'+str(kind))
    terms=[]
    for item in declared:
        if type(item) is int:
            if not 0<=item<len(formula['arguments']):raise SpecificationError('computation_input_position_out_of_range')
            terms.append(formula['arguments'][item])
        elif isinstance(item,str):
            _term(item);terms.append(item)
        elif isinstance(item,dict) and set(item)=={"node","position"}:
            source,index=item["node"],item["position"]
            if not isinstance(source,str) or source not in (nodes or {}):
                raise SpecificationError("unknown_computation_source_node:"+str(source))
            arguments=nodes[source]["formula"]["arguments"]
            if type(index) is not int or not 0 <= index < len(arguments):
                raise SpecificationError("invalid_computation_source_slot:"+source)
            terms.append(arguments[index])
        else:raise SpecificationError('computation_input_must_be_term_or_position')
    if kind=='year' and len(formula['arguments'])<2:
        raise SpecificationError('year_requires_output_head_argument_1')
    if kind=='year' and formula['arguments'][1].startswith('?') and formula['arguments'][1] in terms:
        raise SpecificationError('year_input_cannot_be_its_output_variable')
    return terms




def public_question(question):
    if is_dataclass(question):
        question = asdict(question)
    if not isinstance(question, dict) or set(question)-{"id", "question", "dataset", "metadata"}:
        raise SpecificationError("public_question_has_unapproved_fields")
    metadata = question.get("metadata", {})
    if not isinstance(metadata, dict) or set(metadata)-{"dataset", "split", "type", "level"}:
        raise SpecificationError("public_metadata_has_unapproved_fields")
    if not isinstance(question.get("question"), str) or not question["question"].strip():
        raise SpecificationError("empty_question")
    return {"id": str(question.get("id", digest(question["question"])[:20])),
            "question": question["question"],
            "dataset": question.get("dataset", metadata.get("dataset", "unknown"))}


def _term(term):
    if not isinstance(term, str) or not term or term == "?":
        raise SpecificationError("nonempty_string_term_required")


def validate_atom(atom, predicates=None, *, ground=False):
    if not isinstance(atom, dict) or set(atom)!={"predicate", "arguments"}:
        raise SpecificationError("atomic_formula_required")
    if not isinstance(atom["predicate"], str) or not atom["predicate"]:
        raise SpecificationError("invalid_predicate")
    args = atom["arguments"]
    if not isinstance(args, list):
        raise SpecificationError("arguments_array_required")
    for term in args:
        _term(term)
        if ground and term.startswith("?"):
            raise SpecificationError("ground_fact_required")
    if predicates is not None and predicates.get(atom["predicate"])!=len(args):
        raise SpecificationError("predicate_arity_mismatch:"+atom["predicate"])


def validate_formula(formula, predicates):
    if isinstance(formula, dict) and "predicate" in formula:
        validate_atom(formula, predicates)
        return
    if not isinstance(formula, dict):
        raise SpecificationError("structured_formula_required")
    op = formula.get("op")
    if op=="eq" and set(formula)=={"op", "left", "right"}:
        _term(formula["left"]); _term(formula["right"])
    elif op=="not" and set(formula)=={"op", "arg"}:
        validate_formula(formula["arg"], predicates)
    elif op in {"and", "or"} and set(formula)=={"op", "args"}:
        if not isinstance(formula["args"], list) or not formula["args"]:
            raise SpecificationError("nonempty_formula_group_required")
        for item in formula["args"]:
            validate_formula(item, predicates)
    elif op=="implies" and set(formula)=={"op", "left", "right"}:
        validate_formula(formula["left"], predicates); validate_formula(formula["right"], predicates)
    elif op in {"forall", "exists"} and set(formula)=={"op", "variables", "body"}:
        variables = formula["variables"]
        if not isinstance(variables, list) or not variables or len(set(variables))!=len(variables):
            raise SpecificationError("invalid_quantifier_variables")
        for term in variables:
            _term(term)
            if not term.startswith("?"):
                raise SpecificationError("quantifier_requires_variables")
        validate_formula(formula["body"], predicates)
    else:
        raise SpecificationError("unsupported_formula_shape:"+str(op))


def validate_graph(nodes, relations):
    if not isinstance(nodes, list):
        raise SpecificationError("nodes_array_required")
    ids = [n.get("node_id") for n in nodes]
    if len(set(ids))!=len(ids) or any(not isinstance(n, str) or not n for n in ids):
        raise SpecificationError("duplicate_or_missing_node_ids")
    for node in nodes:
        if not isinstance(node.get("need"), str) or not node["need"].strip():
            raise SpecificationError("node_need_required")
        if node.get("scope", "static")!="static":
            raise SpecificationError("unsupported_scope")
        if not isinstance(node.get("modes"), list) or not node["modes"] or set(node["modes"])-{"retrieve", "reason"}:
            raise SpecificationError("invalid_acquisition_modes")
    if not isinstance(relations, dict):
        raise SpecificationError("relations_object_required")
    for edge in relations.get("edges", []):
        if edge.get("source") not in ids or edge.get("target") not in ids or edge.get("kind") not in {"requires", "implies", "equivalent", "mutex"}:
            raise SpecificationError("invalid_graph_edge")
        if edge.get("scope","static") != "static":
            raise SpecificationError("edge_scope_does_not_match_static_QA_nodes")
    prerequisites = relations.get("prerequisites", [])
    if not isinstance(prerequisites, list) or len(prerequisites)!=len(ids) or {r.get("node") for r in prerequisites}!=set(ids):
        raise SpecificationError("prerequisites_must_cover_fixed_nodes_once")
    for row in prerequisites:
        if not isinstance(row.get("groups"), list):
            raise SpecificationError("prerequisite_groups_required")
        for group in row["groups"]:
            if not isinstance(group, list) or len(set(group))!=len(group) or any(ref not in ids for ref in group):
                raise SpecificationError("invalid_prerequisite_group")
        producers = row.get("producer_groups", {})
        if not isinstance(producers, dict) or set(producers) - {"retrieve", "checked", "rule"}:
            raise SpecificationError("invalid_group_producer")
        for indexes in producers.values():
            if not isinstance(indexes, list) or any(type(i) is not int or not 0 <= i < len(row["groups"]) for i in indexes):
                raise SpecificationError("invalid_producer_group_index")
    goals = relations.get("goal_options")
    if not isinstance(goals, list):
        raise SpecificationError("goal_options_required")
    for group in goals:
        if not isinstance(group, list) or len(set(group))!=len(group) or any(ref not in ids for ref in group):
            raise SpecificationError("invalid_goal_group")


def validate_spec(spec, *, trusted_rules=None):
    if spec.get("schema_version")!="paper_native_v1":
        raise SpecificationError("unknown_spec_schema")
    question = public_question(spec["question"])
    validate_graph(spec["nodes"], spec["relations"])
    theory = spec["theory"]
    generated = spec.get("method_provenance", {}).get("formalization") in {
        "disclosed_engineering_bridge","reviewed_fixed_input"}
    if trusted_rules is None:
        trusted_rules = {r["source_id"]:r["rule_hash"] for r in spec.get("admission_records",[])
                         if r.get("admission")=="pinned_tested_library" and "rule_hash" in r}
    predicates = theory.get("predicates")
    if not isinstance(predicates, dict) or not predicates or any(not isinstance(n, str) or type(a) is not int or a<0 for n,a in predicates.items()):
        raise SpecificationError("predicate_declarations_required")
    answer = spec["answer"]["predicate"]
    if predicates.get(answer)!=1:
        raise SpecificationError("unary_answer_predicate_required")
    for node in spec["nodes"]:
        validate_atom(node["formula"], predicates)
    records, ids = [], set()
    for given in theory.get("given_facts", []):
        validate_atom(given["formula"], predicates, ground=True)
        if not given.get("id") or given["id"] in ids:
            raise SpecificationError("duplicate_source_id")
        ids.add(given["id"])
        source = given.get("source_text")
        if generated and (not isinstance(source, str) or not source.strip() or source not in question["question"]):
            raise SpecificationError("given_fact_source_not_in_question")
        records.append({"source_id":given["id"], "kind":"question_explicit",
                        "source_text":source,"formula":given["formula"],
                        "admission":"model_translation_of_explicit_question_text" if generated else "supplied_fixed_theory_fact",
                        "mechanical_source_present":bool(isinstance(source,str) and source and source in question["question"]),"independent_semantic_certificate":False})
    for rule in theory.get("admitted_rules", []):
        if not rule.get("id") or rule["id"] in ids:
            raise SpecificationError("duplicate_source_id")
        ids.add(rule["id"])
        if not isinstance(rule.get("premises"), list) or not rule["premises"]:
            raise SpecificationError("rule_requires_premises")
        for atom in rule["premises"]+[rule["conclusion"]]:
            validate_atom(atom, predicates)
        kind, source = rule.get("source_kind"), rule.get("source_text", "")
        if kind is None:
            if not generated:
                kind = "supplied_fixed_rule"
            elif rule["conclusion"]["predicate"] == answer:
                kind = "question_definition"
        if not isinstance(kind, str) or not isinstance(source, str) or (generated and kind not in SOURCE_KINDS):
            raise SpecificationError("rule_admission_source_required")
        if generated and kind in {"checked_computation","deterministic_contract"}:
            raise SpecificationError("model_rule_cannot_self_certify_as_checked_or_deterministic")
        if generated and kind=="question_explicit" and (not source.strip() or source not in question["question"]):
            raise SpecificationError("rule_source_not_in_question")
        if kind=="tested_rule_library" and (trusted_rules or {}).get(rule["id"])!=digest(rule):
            raise SpecificationError("unregistered_tested_rule")
        records.append({"source_id":rule["id"],"kind":kind,"source_text":source,
                        "rule_hash":digest(rule),
                        "question_id":question["id"],"question_text":question["question"],
                        "question_text_hash":digest(question["question"]),
                        "source_text_role":"definition_rationale" if kind in {"question_definition","question_answer_definition"} else "source_record",
                        "source_is_question_substring":bool(source and source in question["question"]),
                        "admission":("pinned_tested_library" if kind=="tested_rule_library" else
                                     "explicit_formalization_assumption" if generated else "supplied_fixed_theory_rule"),
                        "independent_semantic_certificate":False,
                        "externally_pinned":kind=="tested_rule_library"})
    axioms = theory.get("axioms", [])
    sources = theory.get("axiom_sources", [])
    if not isinstance(sources,list) or any(not isinstance(s,dict) for s in sources):
        raise SpecificationError("invalid_axiom_sources")
    by_index = {s.get("index"):s for s in sources if type(s.get("index")) is int}
    if len(by_index) != len(sources) or any(i < 0 or i >= len(axioms) for i in by_index):
        raise SpecificationError("invalid_axiom_source_index")
    for index, axiom in enumerate(axioms):
        validate_formula(axiom, predicates)
        if generated and "predicate" in axiom:
            raise SpecificationError("extra_axiom_requires_admissible_source:serialize_positive_atoms_as_sourced_givens")
        body = axiom
        while body.get("op") == "forall":
            body = body["body"]
        denial = body.get("op") == "not" or (
            body.get("op") == "implies" and body["right"].get("op") == "not"
        )
        source = by_index.get(index, {})
        text = source.get("source_text", "")
        kind = source.get("source_kind")
        if not generated:
            admission = "supplied_fixed_theory_axiom"
        elif kind == "question_explicit" and isinstance(text,str) and text.strip() and text in question["question"]:
            admission = "model_translation_of_explicit_question_text"
        elif kind == "question_definition" and denial and isinstance(text,str) and text.strip():
            admission = "explicit_formalization_assumption"
        else:
            raise SpecificationError("extra_axiom_requires_admissible_source:"+str(index))
        records.append({"source_id":"axiom:"+str(index),"kind":kind or "supplied_fixed_axiom",
                        "formula":axiom,"source_text":text,"admission":admission,
                        "independent_semantic_certificate":False})
    nodes = {n["node_id"]:n for n in spec["nodes"]}
    for nid, computation in spec.get("computations", {}).items():
        if nid not in nodes or not isinstance(computation, dict):
            raise SpecificationError("invalid_checked_computation")
        try:
            computation_input_terms(nodes[nid]['formula'],computation,nodes)
        except SpecificationError as exc:
            records.append({'source_id':'computation:'+nid,'kind':'invalid_checked_producer',
                            'reason':str(exc),'independent_semantic_certificate':False})
    return records




def review_fixed_spec(question, spec, model, out_dir, config=None):
    q = public_question(question)
    cfg = asdict(config) if is_dataclass(config) else dict(config or {})
    try:
        if q["question"] != spec["question"]["question"] or q["id"] != spec["question"].get("id", q["id"]):
            raise SpecificationError("fixed_spec_public_question_differs")
        validate_spec(spec, trusted_rules=cfg.get("trusted_rules"))
        result = deepcopy(spec)
        write(str(out_dir)+"/spec.json", result)
        return result
    except (ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
        raise SpecificationError(str(error)) from error


def build_spec(question, model, out_dir, config=None):
    q = public_question(question)
    cfg = asdict(config) if is_dataclass(config) else dict(config or {})
    repairs = cfg.get("max_formalization_repairs", 0)
    if type(repairs) is not int or repairs < 0:
        raise SpecificationError("nonnegative_formalization_repair_limit_required")
    attempts = repairs + 1
    prompts = {"nodes":NODES_PROMPT, "relations":RELATIONS_PROMPT, "bridge":BRIDGE_PROMPT}

    def call(role, payload, validate):
        feedback = []
        for attempt in range(attempts if role == "bridge" else 1):
            public = {**payload,
                      **({"verification_feedback":feedback[-1]} if feedback else {})}
            messages = [{"role":"system","content":prompts[role]},
                        {"role":"user","content":"PUBLIC INPUT\n"+json.dumps(public,ensure_ascii=False,separators=(",",":"))}]
            key = q["id"]+":native:"+role+":"+digest(messages)+":"+str(attempt)
            base = str(out_dir)+"/"+role+"_"+str(attempt)
            write(base+"_request.json", {"request_key":key,"stage":"graph" if role!="bridge" else "formalization", "messages":messages,
                                       "role_origin":"graph_construction" if role!="bridge" else "disclosed_formalization_bridge"})
            response = model.generate(messages, request_key=key, stage="graph" if role!="bridge" else "formalization",
                                      max_output_tokens=int(cfg.get("formalization_output_tokens", 3072 if role=="bridge" else 1536)))
            raw = asdict(response) if is_dataclass(response) else {"text":response.text}
            write(base+"_response.json", raw)
            obj = None
            try:
                if getattr(response, "finish_reason", None) == "length":
                    raise SpecificationError("truncated_model_response: return a compact complete JSON object")
                obj = json.loads(response.text)
                if not isinstance(obj, dict):
                    raise SpecificationError("json_object_required")
                if obj.get("status")=="formalization_unresolved":
                    raise SpecificationError("model_declined:"+str(obj.get("reason", "unspecified")))
                return validate(obj)
            except (ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
                entry = {"reason":str(error),"instruction":"Repair only the stated error; feedback is not evidence. Preserve original question constraints."}
                feedback.append(entry)
                write(base+"_rejection.json", feedback[-1])
        raise SpecificationError(role+"_invalid_after_bounded_repairs:"+feedback[-1]["reason"])

    def nodes_checked(obj):
        if set(obj) != {"nodes"} or not isinstance(obj["nodes"], list):
            raise SpecificationError("node_role_must_return_only_nodes")
        result = []
        for i, node in enumerate(obj["nodes"]):
            if set(node)-{"need","step","modes","scope"}:
                raise SpecificationError("node_role_must_only_propose_information_needs")
            result.append({"node_id":"n"+str(i+1),"need":node["need"],"step":node.get("step",i+1),
                           "modes":node["modes"],"scope":node.get("scope","static")})
        validate_graph(result, {"edges":[],"prerequisites":[{"node":n["node_id"],"groups":[]} for n in result],
                                "goal_options":[[n["node_id"] for n in result]]})
        return result

    nodes = call("nodes", {"question":q["question"]}, nodes_checked)
    def relations_checked(obj):
        if set(obj)!={"edges","prerequisites","goal_options"}:
            raise SpecificationError("relation_role_must_not_modify_nodes")
        validate_graph(nodes,obj)
        return obj
    relations = call("relations", {"question":q["question"],"nodes":nodes}, relations_checked)
    def bridge_checked(obj):
        mapping = obj["node_formulas"]
        if set(mapping)!={n["node_id"] for n in nodes}:
            raise SpecificationError("node_formula_mapping_must_cover_fixed_nodes")
        theory = deepcopy(obj["theory"])
        if "question_definition" in obj:
            rule = obj["question_definition"]
            if not isinstance(rule,dict) or rule.get("source_kind") not in {
                None,"question_definition","question_answer_definition"
            } or rule.get("conclusion",{}).get("predicate") != obj["answer"]["predicate"]:
                raise SpecificationError("malformed_explicit_question_definition")
            if rule not in theory["admitted_rules"]:
                theory["admitted_rules"].append(deepcopy(rule))
        result = {"schema_version":"paper_native_v1", "question":q,
                  "nodes":[{**n,"formula":deepcopy(mapping[n["node_id"]])} for n in nodes],
                  "relations":deepcopy(relations),"theory":theory,
                  "computations":deepcopy(obj.get("computations",{})),"answer":deepcopy(obj["answer"]),
                  "method_provenance":{"graph":"two_stage_graph","formalization":"disclosed_engineering_bridge",
                                       "prompt_source":"published_prompts",
                                       "theory_assumption":"given_facts_and_admitted_rules_fixed_for_question",
                                       "semantic_guarantee":False}}
        result["admission_records"] = validate_spec(result, trusted_rules=cfg.get("trusted_rules"))
        result["spec_hash"] = digest(result)
        return result
    spec = call("bridge", {"question":q["question"],"fixed_nodes":nodes,"fixed_relations":relations,
                           "available_computations":COMPUTATIONS,"trusted_rule_library":cfg.get("trusted_rule_library",[])}, bridge_checked)
    write(str(out_dir)+"/spec.json",spec)
    return spec
