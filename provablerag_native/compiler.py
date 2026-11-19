from copy import deepcopy

from .planning import iter_candidates, _unify, _atom, checked_input_metadata, ground_axiom_facts
from .util import canonical, ground, unify


class CompileUnsupported(ValueError):
    pass


def operation_identity(operation):
    value=deepcopy(operation)
    value.pop('need',None)
    value['scope']=value.get('scope','static')
    value['requires']=sorted(set(value.get('requires',[])))
    return canonical(value)


def compile_operations(spec, candidate, proof, bindings=None):
    try:
        return _compile(spec,candidate,proof,bindings or {})
    except (CompileUnsupported,ValueError,KeyError,TypeError) as exc:
        return {'status':'unsupported','error':str(exc),'operations':{},
                'rule_steps':[],'node_mapping':[],
                'required_input_operation_ids':[],'extra_input_operation_ids':[]}


def _compile(spec,candidate,proof,bindings):
    if not proof.get('proof_found') or proof.get('compile_status')!='compiled':
        raise CompileUnsupported('no_replayed_positive_proof')
    nodes={n['node_id']:n for n in spec['nodes']}
    rules={r['id']:r for r in spec['theory']['admitted_rules']}
    givens={g['id']:ground(g['formula'],bindings) for g in spec['theory'].get('given_facts',[])}
    used_background_facts={fid:formula for fid,(index,formula) in ground_axiom_facts(spec).items()
                           if index in proof.get('used_axiom_indices',[])}
    givens.update(used_background_facts)
    answer=spec['answer']['predicate']
    question_variables={t for n in nodes.values() for t in n['formula']['arguments'] if t.startswith('?')}
    sub=dict(candidate.get('bindings',{}))
    equalities=proof.get('fixed_equalities',{})
    def key(atom):
        return canonical(_atom(atom,equalities))

    def merge_bindings(incoming):
        for term, value in incoming.items():
            slot = {'predicate': 'binding', 'arguments': [term]}
            supplied = {'predicate': 'binding', 'arguments': [value]}
            found = _unify(slot, supplied, sub, question_variables,equalities=equalities)
            if found is None:
                raise CompileUnsupported('supplier_route_conflicts_with_shared_binding:'+term)
            sub.update(found)

    def bind_operation(oid, operation):
        found = _unify(nodes[oid]['formula'], operation['formula'], sub, question_variables,equalities=equalities)
        if found is None:
            raise CompileUnsupported('input_operation_conflicts_with_proof_binding:'+oid)
        sub.update(found)

    merge_bindings(bindings)
    available=deepcopy(candidate.get('operations',{}))
    accepted=candidate.get('accepted_fact_records',{})
    blocked=candidate.get('blocked_operation_ids',{})
    anticipated=candidate.get('anticipated_premises',{})
    used=proof.get('used_premise_ids',[])
    known={key(a):'@given:'+gid for gid,a in givens.items()}
    sources={}
    used_leaves=[]
    for pid in used:
        if pid in givens:
            continue
        formula=anticipated.get(pid) or available.get(pid,{}).get('formula') or nodes.get(pid,{}).get('formula')
        if formula is None:
            raise CompileUnsupported('used_premise_has_no_registered_producer:'+str(pid))
        formula=_atom(formula,sub)
        if formula['predicate']==answer:
            raise CompileUnsupported('answer_assumed_in_proof_premises')
        known[key(formula)]=pid
        sources[pid]=formula
        used_leaves.append(pid)
    steps=[];mapping=[];actual_derived={};base_dependencies={}
    for step in proof.get('replay',[]):
        rid=step['rule_id']
        if rid not in rules:
            raise CompileUnsupported('proof_rule_not_in_fixed_theory:'+rid)
        rule=rules[rid]
        premises=[_atom(a,sub) for a in step['premises']]
        conclusion=_atom(step['conclusion'],sub)
        if any(key(a) not in known for a in premises):
            raise CompileUnsupported('rule_replay_uses_unavailable_premise:'+rid)
        choices=[({},set())]
        for pattern in rule['premises']:
            following=[]
            for assignment,seen in choices:
                for index,actual in enumerate(premises):
                    found=unify(_atom(pattern,equalities),_atom(actual,equalities),assignment)
                    if found is not None:
                        following.append((found,seen|{index}))
            choices=following
        valid=[assignment for assignment,seen in choices if seen==set(range(len(premises)))
               and key(ground(rule['conclusion'],assignment))==key(conclusion)]
        if not valid:
            raise CompileUnsupported('actual_rule_instance_does_not_replay:'+rid)
        matches=[]
        for nid,node in nodes.items():
            if 'reason' not in node['modes'] or nid in sources or nid in actual_derived:
                continue
            found=_unify(node['formula'],conclusion,sub,question_variables,equalities=equalities)
            if found is not None:
                matches.append((nid,found))
        if not matches and conclusion['predicate']!=answer:
            raise CompileUnsupported('actual_rule_conclusion_has_no_reason_node:'+canonical(conclusion))
        matches.sort(key=lambda pair:(
            _atom(nodes[pair[0]]['formula'],sub)!=conclusion,
            pair[0] not in candidate.get('producer_map',{}),
            list(nodes).index(pair[0])))
        if matches:
            nid,sub=matches[0]
            if nid in sources:
                raise CompileUnsupported('same_node_is_both_used_leaf_and_derived_instance:'+nid)
            if nid in actual_derived:
                raise CompileUnsupported('multiple_actual_rule_instances_for_one_graph_node:'+nid)
            actual_derived[nid]={'rule_id':rid,'formula':conclusion}
            refs=[known[key(a)] for a in premises if not known[key(a)].startswith('@given:')]
            base_dependencies[nid]=list(dict.fromkeys(refs))
            node_id=nid
        else:
            node_id='@answer:'+str(step.get('proof_line',len(steps)))
            refs=[known[key(a)] for a in premises if not known[key(a)].startswith('@given:')]
            base_dependencies[node_id]=list(dict.fromkeys(refs))
        known[key(conclusion)]=node_id
        row={**deepcopy(step),'conclusion':conclusion,'premises':premises,'node_id':node_id,
             'actual_rule_substitution':valid[0]}
        steps.append(row)
        mapping.append({'proof_line':step.get('proof_line'),'rule_id':rid,'node_id':node_id,
                        'formula':conclusion,'source':'actual_positive_rule_replay'})
    target=proof.get('answer_template')
    if target is None or target['predicate']!=answer or key(_atom(target,sub)) not in known:
        raise CompileUnsupported('proof_does_not_supply_known_answer_expression')
    relations=spec['relations']
    prerequisites={row['node']:row['groups'] for row in relations['prerequisites']}
    producer_groups={row['node']:row.get('producer_groups',{}) for row in relations['prerequisites']}
    mandatory={nid:[] for nid in nodes}
    mutex=[]
    for edge in relations.get('edges',[]):
        if edge['kind']=='requires':mandatory[edge['target']].append(edge['source'])
        elif edge['kind']=='mutex':mutex.append((edge['source'],edge['target']))
    operations={};dependencies=deepcopy(base_dependencies);active=set(actual_derived)
    completed=set();visiting=set();selected_groups={}
    producer_records={nid:{'kind':'rule','rule_id':row['rule_id'],
                           'premise_node_ids':list(base_dependencies.get(nid,[])),
                           'formula':deepcopy(row['formula'])} for nid,row in actual_derived.items()}
    selected_given_ids=(set(used)&set(givens))|set(used_background_facts)

    def check_mutex():
        branch_nodes = active | set(accepted)
        if any(a in branch_nodes and b in branch_nodes for a,b in mutex):
            raise CompileUnsupported('actual_proof_or_input_route_violates_mutex')

    def add_operation(oid,operation):
        operation=deepcopy(operation)
        operation['formula']=_atom(operation['formula'],sub)
        validate_operation_source(oid,operation)
        bind_operation(oid, operation)
        if oid in operations and canonical(operations[oid])!=canonical(operation):
            raise CompileUnsupported('incompatible_producers_for_input:'+oid)
        if oid in actual_derived:
            raise CompileUnsupported('input_producer_conflicts_with_actual_derived_node:'+oid)
        operations[oid]=operation
        active.add(oid)
        dependencies.setdefault(oid,[])
        dependencies[oid]=list(dict.fromkeys(dependencies[oid]+operation.get('requires',[])))
        check_mutex()
        for ref in list(operation.get('requires',[])):
            ensure_supplier(ref)
        ensure_inputs(oid)

    def validate_operation_source(oid,operation):
        if oid in sources and canonical(operation['formula'])!=canonical(_atom(sources[oid],sub)):
            raise CompileUnsupported('used_leaf_operation_differs_from_proof_source:'+oid)
        if oid in nodes and operation.get('scope','static')!=nodes[oid].get('scope','static'):
            raise CompileUnsupported('operation_scope_differs_from_fixed_node:'+oid)
        if 'accepted_fact_records' in candidate and oid not in accepted:
            raise CompileUnsupported('runtime_input_not_established:'+oid)
        if oid in accepted:
            record=accepted[oid]
            if record.get('rule_execution'):
                raise CompileUnsupported('rule_fact_is_not_an_acquisition_leaf:'+oid)
            if (operation['formula']!=record['formula']
                    or operation.get('scope','static')!=record.get('scope','static')):
                raise CompileUnsupported('operation_differs_from_accepted_ground_record:'+oid)
            kind='checked' if record.get('computation') is not None else 'retrieve'
            if operation.get('kind')!=kind:
                raise CompileUnsupported('operation_producer_kind_differs_from_accepted_record:'+oid)
            if kind=='checked' and operation.get('computation')!=spec.get('computations',{}).get(oid):
                raise CompileUnsupported('checked_operation_changes_fixed_computation:'+oid)

    def register_supplier_route(route):
        merge_bindings(route.get('bindings',{}))
        for nid,producer in route.get('producer_map',{}).items():
            if nid in actual_derived and (producer.get('kind')!='rule' or producer.get('rule_id')!=actual_derived[nid]['rule_id']):
                raise CompileUnsupported('supplier_route_changes_actual_proof_producer:'+nid)
        selected_given_ids.update(route.get('given_ids',[]))
        for nid,producer in route.get('producer_map',{}).items():
            active.add(nid);producer_records.setdefault(nid,deepcopy(producer))
            group=route.get('selected_prerequisite_groups',{}).get(nid)
            extra=prerequisites[nid][group] if group is not None else []
            refs=mandatory[nid]+extra+producer.get('premise_node_ids',[])
            dependencies[nid]=list(dict.fromkeys(dependencies.get(nid,[])+refs))
            selected_groups[nid]=group
            completed.add(nid)
        check_mutex()
        for oid,operation in route['operations'].items():
            operation=deepcopy(operation);operation['formula']=_atom(operation['formula'],sub)
            validate_operation_source(oid,operation)
            bind_operation(oid, operation)
            if oid in operations and canonical(operations[oid])!=canonical(operation):
                raise CompileUnsupported('supplier_operation_conflict:'+oid)
            operations[oid]=operation
            dependencies[oid]=list(dict.fromkeys(dependencies.get(oid,[])+operation.get('requires',[])))

    def ensure_supplier(nid):
        if nid in blocked:
            raise CompileUnsupported('runtime_operation_producer_unresolved:'+nid+':'+str(blocked[nid]))
        if nid.startswith('@given:'):
            return
        if nid in actual_derived or nid.startswith('@answer:'):
            ensure_inputs(nid)
            return
        if nid in operations:
            ensure_inputs(nid)
            return
        if nid in available:
            add_operation(nid,available[nid])
            return
        if nid not in nodes:
            raise CompileUnsupported('missing_required_input_node:'+nid)
        supplier_spec=deepcopy(spec)
        supplier_spec['relations']['goal_options']=[[nid]]
        routes=iter_candidates(supplier_spec)
        errors=[]
        for route in routes:
            snapshot=deepcopy((operations,dependencies,active,completed,selected_groups,producer_records,selected_given_ids,sub))
            try:
                register_supplier_route(route)
                check_cycles()
                return
            except CompileUnsupported as exc:
                if len(errors)<3:errors.append(str(exc))
                saved_ops,saved_deps,saved_active,saved_done,saved_groups,saved_producers,saved_given,saved_sub=snapshot
                sub.clear(); sub.update(saved_sub)
                operations.clear();operations.update(saved_ops)
                dependencies.clear();dependencies.update(saved_deps)
                active.clear();active.update(saved_active)
                completed.clear();completed.update(saved_done)
                selected_groups.clear();selected_groups.update(saved_groups)
                producer_records.clear();producer_records.update(saved_producers)
                selected_given_ids.clear();selected_given_ids.update(saved_given)
        raise CompileUnsupported('required_input_routes_incompatible:'+nid+':'+','.join(errors))

    def ensure_inputs(nid):
        if nid in completed:
            return
        if nid in visiting:
            raise CompileUnsupported('actual_proof_input_cycle:'+nid)
        visiting.add(nid)
        if nid not in nodes:
            for ref in dependencies.get(nid,[]):ensure_supplier(ref)
            completed.add(nid);visiting.remove(nid)
            return
        groups=prerequisites[nid] or [[]]
        preferred=candidate.get('selected_prerequisite_groups',{}).get(nid)
        kind=producer_records.get(nid,operations.get(nid,{})).get('kind')
        indices=list(producer_groups[nid].get(kind,range(len(groups))))
        if preferred is not None and preferred in indices:
            indices.remove(preferred);indices.insert(0,preferred)
        errors=[]
        for gi in indices:
            snapshot=deepcopy((operations,dependencies,active,completed,selected_groups,producer_records,selected_given_ids,sub))
            try:
                refs=list(dict.fromkeys(dependencies.get(nid,[])+mandatory[nid]+groups[gi]))
                dependencies[nid]=refs
                active.add(nid);check_mutex()
                for ref in refs:ensure_supplier(ref)
                selected_groups[nid]=gi if prerequisites[nid] else None
                completed.add(nid)
                check_cycles()
                visiting.remove(nid)
                return
            except CompileUnsupported as exc:
                errors.append(str(exc))
                saved_ops,saved_deps,saved_active,saved_done,saved_groups,saved_producers,saved_given,saved_sub=snapshot
                sub.clear(); sub.update(saved_sub)
                operations.clear();operations.update(saved_ops)
                dependencies.clear();dependencies.update(saved_deps)
                active.clear();active.update(saved_active)
                completed.clear();completed.update(saved_done)
                selected_groups.clear();selected_groups.update(saved_groups)
                producer_records.clear();producer_records.update(saved_producers)
                selected_given_ids.clear();selected_given_ids.update(saved_given)
        visiting.remove(nid)
        raise CompileUnsupported('no_admissible_actual_rule_input_group:'+nid+':'+','.join(errors[:3]))

    def check_cycles():
        entered=set();done=set()
        def visit(nid):
            if nid in entered:raise CompileUnsupported('compiled_execution_dependency_cycle:'+nid)
            if nid in done:return
            entered.add(nid)
            for ref in dependencies.get(nid,[]):visit(ref)
            entered.remove(nid);done.add(nid)
        for nid in dependencies:visit(nid)

    for pid in used_leaves:ensure_supplier(pid)
    for row in steps:ensure_inputs(row['node_id'])
    check_cycles();check_mutex()
    def acquisition_leaves(nid,trail=()):
        if nid in trail:raise CompileUnsupported('compiled_leaf_dependency_cycle')
        if nid in operations:return [nid]
        return list(dict.fromkeys(x for dep in dependencies.get(nid,[]) for x in acquisition_leaves(dep,trail+(nid,))))
    for oid,operation in operations.items():
        operation['formula']=_atom(operation['formula'],sub)
        refs=dependencies.get(oid,[])
        operation['requires']=list(dict.fromkeys(x for ref in refs for x in acquisition_leaves(ref)))
        if oid in operation['requires']:raise CompileUnsupported('operation_requires_itself:'+oid)
        derived={d['node_id']:d for d in operation.get('derived_inputs',[])}
        for ref in refs:
            if ref in nodes and ref not in operations:
                producer=producer_records.get(ref,{})
                derived[ref]={'node_id':ref,'formula':_atom(nodes[ref]['formula'],sub),
                              'rule_ids':[producer['rule_id']] if producer.get('rule_id') else [],
                              'supplier_ids':acquisition_leaves(ref)}
                if producer.get('kind') == 'given':
                    derived[ref]['given_id'] = producer['given_id']
        if derived:
            operation['derived_inputs']=list(derived.values())
            for record in operation['derived_inputs']:
                record['formula']=_atom(record['formula'],sub)
    for oid,operation in operations.items():
        if operation.get('kind')=='checked':
            operation.update(checked_input_metadata(spec,operation,operations,dependencies,producer_records))
            selected_given_ids.update(operation.get('given_source_ids',[]))
    rule_operations = {}
    def include_rule_input(nid):
        producer = producer_records.get(nid, {})
        if producer.get('kind') != 'rule' or nid in rule_operations:
            return
        refs = dependencies.get(nid, [])
        node = nodes[nid]
        if node['formula']['predicate'] == answer:
            raise CompileUnsupported('answer_cannot_supply_an_acquisition')
        given_nodes = {ref:producer_records[ref]['given_id'] for ref in refs
                       if producer_records.get(ref,{}).get('kind') == 'given'}
        sources = [given_nodes.get(s, s) for s in producer.get('premise_node_ids', [])]
        rule_operations[nid] = {'node_id':nid, 'kind':'rule', 'rule_id':producer['rule_id'],
            'formula':_atom(node['formula'],sub), 'scope':node.get('scope','static'),
            'requires':[ref for ref in refs if ref not in given_nodes],
            'rule_source_ids':list(dict.fromkeys([*sources, *sorted(selected_given_ids)]))}
        for ref in refs:
            include_rule_input(ref)
    for oid in operations:
        for ref in dependencies.get(oid, []):
            include_rule_input(ref)
    return {'status':'compiled','error':None,'operations':operations,'rule_steps':steps,'node_mapping':mapping,
            'rule_input_operations':rule_operations,
            'required_input_operation_ids':sorted(operations),
            'extra_input_operation_ids':sorted(set(operations)-set(used_leaves)),
            'used_proof_premise_ids':list(used),'selected_prerequisite_groups':selected_groups,
            'node_dependencies':dependencies,'answer_template':_atom(target,sub),
            'bindings':{term:_atom({'predicate':'binding','arguments':[term]},sub)['arguments'][0]
                        for term in question_variables if term in sub},
            'producer_records':producer_records,'given_input_source_ids':sorted(selected_given_ids),
            'theory_changed':False,'anticipated_premises_changed':False,
            'input_policy':'additional_suppliers_are_execution_inputs_not_conditional_assumptions'}
