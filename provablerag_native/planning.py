from copy import deepcopy

from .formalization import validate_spec, SpecificationError, computation_input_terms, COMPUTATIONS
from .util import canonical, digest
import re


def checked_input_metadata(spec,operation,operations,node_dependencies,producer_map=None):
    nodes={n['node_id']:n for n in spec['nodes']};nid=operation['node_id']
    if nid not in nodes:raise SpecificationError('checked_operation_has_no_fixed_node')
    computation=spec.get('computations',{}).get(nid,operation.get('computation',{}))
    terms=computation_input_terms(nodes[nid]['formula'],computation,nodes)
    declarations=computation.get('inputs',COMPUTATIONS[computation['kind']]['inputs'])
    scope=operation.get('scope','static');ancestors=set();visiting=set()
    def visit(ref):
        if ref==nid or ref in visiting:raise SpecificationError('checked_input_dependency_cycle')
        if ref in ancestors:return
        visiting.add(ref)
        for parent in node_dependencies.get(ref,operations.get(ref,{}).get('requires',[])):visit(parent)
        visiting.remove(ref);ancestors.add(ref)
    for ref in node_dependencies.get(nid,operation.get('requires',[])):visit(ref)
    givens={g['id']:g for g in spec['theory'].get('given_facts',[])}
    givens.update({fid:{'id':fid,'formula':formula} for fid,(_,formula) in ground_axiom_facts(spec).items()})
    input_sources={};operand_sources=[];given_ids=set();question_literals=[]
    for term,declaration in zip(terms,declarations):
        selector=declaration if isinstance(declaration,dict) else None
        sources=[]
        for ref in sorted(ancestors):
            if selector and ref!=selector['node']:continue
            if ref not in nodes or nodes[ref].get('scope','static')!=scope:continue
            producer=(producer_map or {}).get(ref,{})
            if ref in operations or producer.get('kind')=='rule':
                source_computation=spec.get('computations',{}).get(ref)
                source_kind=producer.get('kind',operations.get(ref,{}).get('kind'))
                slots=range(len(nodes[ref]['formula']['arguments']))
                if source_kind=='checked' and source_computation:
                    slots=[1] if source_computation['kind']=='year' else []
                for index in slots:
                    if selector and index!=selector['position']:continue
                    value=nodes[ref]['formula']['arguments'][index]
                    if value==term:sources.append({'node_id':ref,'argument_index':index,'scope':scope})
            if producer.get('kind')=='given' and producer.get('given_id') in givens:
                gid=producer['given_id'];given=givens[gid]
                if given.get('scope','static')!=scope:continue
                for index,value in enumerate(nodes[ref]['formula']['arguments']):
                    if selector and index!=selector['position']:continue
                    if value==term:
                        sources.append({'node_id':gid,'argument_index':index,'scope':scope,
                                        'source_kind':'given','selected_node':ref});given_ids.add(gid)
        if selector and not sources:
            raise SpecificationError('unavailable_selected_source_slot:'+selector['node'])
        if not selector and not term.startswith('?'):
            for gid,given in givens.items():
                if given.get('scope','static')!=scope:continue
                for index,value in enumerate(given['formula']['arguments']):
                    if value==term:
                        sources.append({'node_id':gid,'argument_index':index,'scope':scope,'source_kind':'given'});given_ids.add(gid)
            if not sources and re.search(r'(?<!\w)'+re.escape(term)+r'(?!\w)',spec['question']['question']):
                question_literals.append(term)
        unique={canonical(row):row for row in sources}
        operand_sources.append(list(unique.values()))
        combined={canonical(row):row for row in input_sources.get(term,[])}
        input_sources[term]=list((combined|unique).values())
        if not sources and term not in question_literals:
            raise SpecificationError('checked_input_has_no_selected_supplier:'+term)
    return {'input_terms':terms,'input_types':COMPUTATIONS[computation['kind']]['input_types'],
            'input_sources':input_sources,'operand_sources':operand_sources,'given_source_ids':sorted(given_ids),
            'input_dependency_ids':sorted(ref for ref in ancestors if ref in operations),
            'selected_input_node_ids':sorted(ancestors),
            'question_literal_inputs':list(dict.fromkeys(question_literals)),
            'required_bindings':[t for t in dict.fromkeys(terms) if t.startswith('?')]}


def _resolve(term, substitutions):
    visited = set()
    while term in substitutions and substitutions[term]!=term:
        if term in visited:
            raise SpecificationError("cyclic_variable_substitution")
        visited.add(term)
        term = substitutions[term]
    return term


def _atom(atom, substitutions):
    return {"predicate":atom["predicate"],"arguments":[_resolve(t,substitutions) for t in atom["arguments"]]}


def fixed_equalities(axioms):
    aliases={}
    pending=list(axioms)
    while pending:
        expr=pending.pop()
        if expr.get("op")=="and":
            pending.extend(expr["args"])
        if expr.get("op")!="eq":
            continue
        left,right=(expr[k] for k in ("left","right"))
        if left.startswith("?") or right.startswith("?"):
            continue
        left,right=(_resolve(t,aliases) for t in (left,right))
        if left!=right:
            first,second=sorted((left,right))
            aliases[second]=first
    return {t:_resolve(t,aliases) for t in aliases}


def ground_axiom_facts(spec):
    registered=[n["node_id"] for n in spec["nodes"]]+[g["id"] for g in spec["theory"].get("given_facts",[])]
    prefix="@axiom:"
    while any(name.startswith(prefix) for name in registered):
        prefix="@"+prefix
    return {prefix+str(i):(i,formula) for i,formula in enumerate(spec['theory'].get('axioms',[]))
            if 'predicate' in formula and all(not t.startswith('?') for t in formula['arguments'])}


def _unify(a, b, substitutions, question_variables=(), *, equalities=None):
    if a["predicate"]!=b["predicate"] or len(a["arguments"])!=len(b["arguments"]):
        return None
    result = dict(substitutions)
    for left,right in zip(a["arguments"],b["arguments"]):
        left,right = _resolve(left,result),_resolve(right,result)
        if left==right or equalities and _resolve(left,equalities)==_resolve(right,equalities):
            continue
        if left.startswith("?") and right.startswith("?"):
            if left in question_variables and right in question_variables:
                first,second=sorted((left,right))
                result[second]=first
                continue
            if left in question_variables:
                result[right] = left
                continue
        if left.startswith("?"):
            result[left] = right
        elif right.startswith("?"):
            result[right] = left
        else:
            return None
    return result


def inference_directions(spec):
    nodes = {node["node_id"]: node for node in spec["nodes"]}
    variables = {arg for node in nodes.values() for arg in node["formula"]["arguments"]
                 if arg.startswith("?")}
    equalities=fixed_equalities(spec["theory"].get("axioms",[]))
    output = []
    for index, edge in enumerate(spec["relations"].get("edges", [])):
        if edge["kind"] not in {"implies", "equivalent"}:
            continue
        ends = [(edge["source"], edge["target"])]
        if edge["kind"] == "equivalent":
            ends.append((edge["target"], edge["source"]))
        for source, target in ends:
            rule_ids = []
            for rule in spec["theory"]["admitted_rules"]:
                if len(rule["premises"]) != 1:
                    continue
                body, head = rule["premises"][0], rule["conclusion"]
                prefix = "?edge_rule_"
                while any(v.startswith(prefix) for v in variables):
                    prefix += "_"
                fresh = {v: prefix + v[1:] for v in body["arguments"] + head["arguments"]
                         if v.startswith("?")}
                sub = _unify(_atom(head, fresh), nodes[target]["formula"], {}, variables,equalities=equalities)
                if sub is not None and _unify(_atom(body, fresh), nodes[source]["formula"], sub, variables,equalities=equalities) is not None:
                    rule_ids.append(rule["id"])
            output.append({"edge_index": index, "source": source, "target": target,
                           "kind": edge["kind"], "admitted_rule_ids": rule_ids})
    return output


def enumerate_candidates(spec, max_candidates=16, *, audit=None, max_expansions=None):
    return list(iter_candidates(spec, max_candidates, audit=audit, max_expansions=max_expansions))


def iter_candidates(spec, max_candidates=None, *, audit=None, max_expansions=None):
    for name, limit in (("candidate",max_candidates),("expansion",max_expansions)):
        if limit is not None and (type(limit) is not int or limit < 0):
            raise ValueError("nonnegative_"+name+"_limit_required")
    validate_spec(spec)
    nodes = {n["node_id"]:n for n in spec["nodes"]}
    order = list(nodes)
    groups = {r["node"]:r["groups"] for r in spec["relations"]["prerequisites"]}
    producer_groups = {r["node"]:r.get("producer_groups",{}) for r in spec["relations"]["prerequisites"]}
    mandatory = {n:[] for n in nodes}
    mutex = []
    for edge in spec["relations"].get("edges",[]):
        if edge["kind"]=="requires":
            mandatory[edge["target"]].append(edge["source"])
        elif edge["kind"]=="mutex":
            mutex.append((edge["source"],edge["target"]))
    rules = spec["theory"]["admitted_rules"]
    edge_suppliers = {}
    for direction in inference_directions(spec):
        for rid in direction["admitted_rule_ids"]:
            key = (direction["target"], rid)
            edge_suppliers.setdefault(key, []).append(direction["source"])
    computations = spec.get("computations",{})
    givens = [*spec["theory"].get("given_facts",[]),
              *({"id":fid,"formula":formula} for fid,(_,formula) in ground_axiom_facts(spec).items())]
    equalities=fixed_equalities(spec["theory"].get("axioms",[]))
    answer_predicate = spec["answer"]["predicate"]
    question_variables = {t for n in nodes.values() for t in n["formula"]["arguments"] if t.startswith("?")}
    seen = set()
    expansions = 0
    if audit is None:
        audit = {}
    audit.update(status="complete", expansion_limit=max_expansions, expansions=0, candidate_count=0)
    if max_candidates == 0:
        audit["status"] = "candidate_limit"
        return
    class ExpansionLimit(Exception):
        pass

    def compatible(state):
        active = set(state["producers"])
        return not any(a in active and b in active for a,b in mutex)

    def expand_many(refs, trail, state):
        if not refs:
            yield state
            return
        first,*rest = refs
        for branch in expand(first,trail,state):
            yield from expand_many(rest,trail,branch)

    def add_rule_inputs(body, index, trail, state, refs, preferred):
        if index==len(body):
            yield state, refs
            return
        needed = _atom(body[index],state["substitutions"])
        for given in givens:
            sub = _unify(needed,given["formula"],state["substitutions"],question_variables,equalities=equalities)
            if sub is not None:
                branch = deepcopy(state); branch["substitutions"] = sub
                branch["given_ids"].add(given["id"])
                yield from add_rule_inputs(body,index+1,trail,branch,refs,preferred)
        for ref in dict.fromkeys([*preferred, *order]):
            sub = _unify(needed,nodes[ref]["formula"],state["substitutions"],question_variables,equalities=equalities)
            if sub is None:
                continue
            branch = deepcopy(state); branch["substitutions"] = sub
            for grounded in expand(ref,trail,branch):
                yield from add_rule_inputs(body,index+1,trail,grounded,refs+[ref],preferred)

    def expand(nid, trail, state):
        nonlocal expansions
        expansions += 1
        if max_expansions is not None and expansions>max_expansions:
            raise ExpansionLimit
        if nid in trail:
            return
        if nid in state["producers"]:
            yield state
            return
        node = nodes[nid]
        for gi,extra in enumerate(groups[nid] or [[]]):
            permitted=producer_groups[nid]
            def permitted_for(kind):
                return kind not in permitted or gi in permitted[kind]
            base = deepcopy(state)
            base["groups"][nid] = gi if groups[nid] else None
            base["dependencies"][nid] = list(dict.fromkeys(mandatory[nid]+extra))
            for ready in expand_many(base["dependencies"][nid],trail+(nid,),base):
                formula = _atom(node["formula"],ready["substitutions"])
                if "retrieve" in node["modes"] and permitted_for("retrieve") and formula["predicate"]!=answer_predicate:
                    branch = deepcopy(ready); branch["producers"][nid] = {"kind":"retrieve"}
                    if compatible(branch):
                        yield branch
                if nid in computations and "reason" in node["modes"] and permitted_for("checked") and formula["predicate"] != answer_predicate:
                    branch=deepcopy(ready);branch['producers'][nid]={'kind':'checked','computation':deepcopy(computations[nid])}
                    possible_ops={k:{} for k,p in branch['producers'].items() if p['kind'] in {'retrieve','checked'}}
                    provisional={'node_id':nid,'scope':node.get('scope','static'),'formula':formula,'computation':computations[nid]}
                    try:
                        checked_input_metadata(spec,provisional,possible_ops,branch['dependencies'],branch['producers'])
                    except SpecificationError:
                        pass
                    else:
                        if compatible(branch):yield branch
                for given in givens:
                    if given.get("scope","static") != node.get("scope","static"):
                        continue
                    sub = _unify(formula,given["formula"],ready["substitutions"],question_variables,equalities=equalities)
                    if sub is not None:
                        branch = deepcopy(ready); branch["substitutions"] = sub
                        branch["producers"][nid] = {"kind":"given","given_id":given["id"]}
                        branch["given_ids"].add(given["id"])
                        if compatible(branch):
                            yield branch
                if "reason" not in node["modes"] or not permitted_for("rule"):
                    continue
                for rule_index,rule in enumerate(rules):
                    local_variables = {t for atom in rule["premises"]+[rule["conclusion"]] for t in atom["arguments"] if t.startswith("?")}
                    fresh = {v:"?local_"+digest([nid,rule["id"],v])[:20] for v in local_variables}
                    head = _atom(rule["conclusion"],fresh)
                    sub = _unify(head,formula,ready["substitutions"],question_variables,equalities=equalities)
                    if sub is None:
                        continue
                    branch = deepcopy(ready); branch["substitutions"] = sub
                    body = [_atom(a,fresh) for a in rule["premises"]]
                    preferred = [*edge_suppliers.get((nid,rule["id"]), []), *ready["dependencies"][nid]]
                    for with_inputs,refs in add_rule_inputs(body,0,trail+(nid,),branch,[],preferred):
                        result = deepcopy(with_inputs)
                        result["producers"][nid] = {"kind":"rule","rule_id":rule["id"],"premise_node_ids":list(dict.fromkeys(refs)),
                            "rule_premises":[_atom(a,result["substitutions"]) for a in body],
                            "rule_conclusion":_atom(head,result["substitutions"])}
                        result["dependencies"][nid] = list(dict.fromkeys(result["dependencies"][nid]+refs))
                        if compatible(result):
                            yield result

    def to_candidate(state, goals):
        operations = {}
        producers = state["producers"]
        def suppliers(nid, trail=()):
            if nid in trail:
                raise SpecificationError("selected_dependency_cycle")
            if producers[nid]["kind"] in {"retrieve","checked"}:
                return [nid]
            return list(dict.fromkeys(s for p in state["dependencies"].get(nid,[]) for s in suppliers(p,trail+(nid,))))
        for nid in order:
            producer = producers.get(nid,{})
            if producer.get("kind") not in {"retrieve","checked"}:
                continue
            dependencies = state["dependencies"].get(nid,[])
            inputs = list(dict.fromkeys(s for ref in dependencies for s in suppliers(ref)))
            formula = _atom(nodes[nid]["formula"],state["substitutions"])
            operation = {"node_id":nid,"need":nodes[nid]["need"],"scope":nodes[nid].get("scope","static"),
                         "kind":producer["kind"],"formula":formula,"requires":inputs}
            if producer["kind"]=="checked":
                operation["computation"] = deepcopy(producer["computation"])
            derived = []
            for ref in dependencies:
                if producers[ref]["kind"]=="rule":
                    derived.append({"node_id":ref,"formula":_atom(nodes[ref]["formula"],state["substitutions"]),
                                    "rule_ids":[producers[ref]["rule_id"]],"supplier_ids":suppliers(ref)})
            if derived:
                operation["derived_inputs"] = derived
            operations[nid] = operation
        visited, visiting = set(), set()
        def visit(nid):
            if nid in visiting:
                raise SpecificationError("selected_operation_cycle")
            if nid in visited:
                return
            visiting.add(nid)
            for supplier in operations[nid]["requires"]:
                visit(supplier)
            visiting.remove(nid); visited.add(nid)
        for nid in operations:
            visit(nid)
        for oid,operation in operations.items():
            if operation['kind']=='checked':
                operation.update(checked_input_metadata(spec,operation,operations,state['dependencies'],producers))
        symbolic=dict(state['substitutions'])
        years={}
        for operation in operations.values():
            if operation.get("computation",{}).get("kind")=="year":
                source=_resolve(operation["input_terms"][0],symbolic)
                output=_resolve(operation["formula"]["arguments"][1],symbolic)
                if source in years:
                    slot=lambda t:{"predicate":"same_year","arguments":[t]}
                    joined=_unify(slot(output),slot(years[source]),symbolic,question_variables)
                    if joined is None:
                        return None
                    symbolic=joined
                else:
                    years[source]=output
        for operation in operations.values():
            if operation.get("computation",{}).get("kind") in {
                    "compare_date","earlier","compare_year","less_year","compare_number","less_number","not_equal"}:
                left,right=(_resolve(t,symbolic) for t in operation["input_terms"])
                if left==right:
                    return None
        result = {"anticipated_premises":{nid:op["formula"] for nid,op in operations.items()},
                  "operations":operations,"selected_rule_ids":list(dict.fromkeys(p["rule_id"] for p in producers.values() if p["kind"]=="rule")),
                  "goal_node_ids":list(goals),"selected_prerequisite_groups":state["groups"],
                  "producer_map":deepcopy(producers),"given_ids":sorted(state["given_ids"]),
                  "bindings":{t:_resolve(t,state["substitutions"]) for n in spec["nodes"] for t in n["formula"]["arguments"] if t.startswith("?") and _resolve(t,state["substitutions"])!=t}}
        return result

    def goal_states():
        try:
            for goals in spec["relations"]["goal_options"]:
                initial = {"producers":{},"groups":{},"dependencies":{},"substitutions":{},"given_ids":set()}
                for state in expand_many(goals,(),initial):
                    yield goals, state
        except ExpansionLimit:
            audit["status"] = "expansion_limit"

    for goals,state in goal_states():
        candidate = to_candidate(state,goals)
        if candidate is None:
            continue
        identity = deepcopy(candidate)
        identity.pop("selected_rule_ids")
        for producer in identity["producer_map"].values():
            producer.pop("rule_id",None)
            if "premise_node_ids" in producer:
                producer["premise_node_ids"] = sorted(producer["premise_node_ids"])
                producer["rule_premises"] = sorted(producer["rule_premises"],key=canonical)
        for operation in identity["operations"].values():
            for derived in operation.get("derived_inputs",[]):
                derived.pop("rule_ids",None)
        key = digest(identity)
        if key in seen:
            continue
        seen.add(key)
        candidate.update(candidate_id="candidate_"+key[:20],stable_order=len(seen)-1)
        audit.update(expansions=expansions,candidate_count=len(seen))
        yield candidate
        if max_candidates is not None and len(seen)>=max_candidates:
            audit["status"] = "candidate_limit"
            break
    audit.update(expansions=min(expansions,max_expansions) if max_expansions is not None else expansions,
                 candidate_count=len(seen))
