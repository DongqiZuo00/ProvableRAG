from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import hashlib
from itertools import product
import json
import math
import os
import re
import subprocess
import time

from provablerag.proving import Atom, Rule, Theory, classify_exit, match, instantiate, unit
from .planning import fixed_equalities


class ProofInputError(ValueError):
    pass


def wire_atom(value):
    return {'predicate': value.predicate, 'arguments': list(value.arguments)}


def atom(value, predicates):
    if isinstance(value, Atom):
        value = asdict(value)
    if not isinstance(value, dict) or not isinstance(value.get('predicate'), str):
        raise ProofInputError('invalid_atom')
    name, terms = value['predicate'], value.get('arguments', [])
    if name not in predicates or not isinstance(terms, (list, tuple)) or len(terms) != predicates[name]:
        raise ProofInputError('undeclared_predicate_or_arity')
    if any(not isinstance(t, str) or not t or t == '?' for t in terms):
        raise ProofInputError('invalid_term')
    return Atom(name, tuple(terms))


def theory_from_spec(spec):
    raw = spec['theory']
    predicates = dict(raw['predicates'])
    if any(not isinstance(k, str) or not k or type(v) is not int or v < 0 for k, v in predicates.items()):
        raise ProofInputError('invalid_predicate_signature')
    answer = spec['answer']['predicate']
    if predicates.get(answer) != 1:
        raise ProofInputError('answer_predicate_must_be_unary')
    rules = []
    for item in raw.get('admitted_rules', []):
        premises = tuple(atom(a, predicates) for a in item['premises'])
        if not premises:
            raise ProofInputError('empty_rule_body_use_given_fact')
        rules.append(Rule(item['id'], premises, atom(item['conclusion'], predicates),
                          item.get('source_kind', 'question_definition')))
    givens = {item['id']: atom(item['formula'], predicates) for item in raw.get('given_facts', [])}
    if len(givens) != len(raw.get('given_facts', [])) or len({r.rule_id for r in rules}) != len(rules):
        raise ProofInputError('duplicate_source_id')
    if any(t.startswith('?') for a in givens.values() for t in a.arguments):
        raise ProofInputError('given_facts_must_be_ground')
    return Theory(predicates, tuple(rules), givens)


class Symbols:
    def __init__(self, predicates):
        self.predicates = {p: 'p' + str(i) for i, p in enumerate(predicates)}
        self.constants, self.variables = {}, {}

    def term(self, term):
        if not isinstance(term, str) or not term or term == '?':
            raise ProofInputError('invalid_term')
        table, prefix = (self.variables, 'x') if term.startswith('?') else (self.constants, 'c')
        return table.setdefault(term, prefix + str(len(table)))

    def atom(self, value):
        name = self.predicates[value.predicate]
        return name + ('(' + ','.join(self.term(t) for t in value.arguments) + ')' if value.arguments else '')

    def closure(self, expression, variables):
        return ''.join('all ' + self.term(v) + ' ' for v in sorted(variables)) + '(' + expression + ')'


def formula_variables(value):
    if 'predicate' in value:
        return {t for t in value.get('arguments', []) if isinstance(t, str) and t.startswith('?')}
    op = value.get('op')
    if op == 'eq':
        return {value[k] for k in ('left', 'right') if isinstance(value.get(k), str) and value[k].startswith('?')}
    if op == 'not':
        return formula_variables(value['arg'])
    if op in ('and', 'or'):
        return set().union(*(formula_variables(a) for a in value['args']))
    if op == 'implies':
        return formula_variables(value['left']) | formula_variables(value['right'])
    if op in ('forall', 'exists'):
        return formula_variables(value['body']) - set(value['variables'])
    raise ProofInputError('unsupported_structured_formula_operator')


def render_formula(value, symbols, predicates):
    if not isinstance(value, dict):
        raise ProofInputError('formula_must_be_structured_object')
    if 'predicate' in value:
        return symbols.atom(atom(value, predicates))
    op = value.get('op')
    if op == 'eq':
        return '(' + symbols.term(value['left']) + ' = ' + symbols.term(value['right']) + ')'
    if op == 'not':
        return '-(' + render_formula(value['arg'], symbols, predicates) + ')'
    if op in ('and', 'or'):
        args = value.get('args')
        if not isinstance(args, list) or not args:
            raise ProofInputError('empty_boolean_formula')
        return '(' + (' & ' if op == 'and' else ' | ').join(render_formula(a, symbols, predicates) for a in args) + ')'
    if op == 'implies':
        return '(' + render_formula(value['left'], symbols, predicates) + ' -> ' + render_formula(value['right'], symbols, predicates) + ')'
    if op in ('forall', 'exists'):
        variables = value.get('variables')
        if (not isinstance(variables, list) or not variables or len(set(variables)) != len(variables)
                or any(not isinstance(t, str) or not t.startswith('?') or t == '?' for t in variables)):
            raise ProofInputError('invalid_quantifier_variables')
        quantifier = 'all' if op == 'forall' else 'exists'
        return ''.join(quantifier + ' ' + symbols.term(t) + ' ' for t in variables) + '(' + render_formula(value['body'], symbols, predicates) + ')'
    raise ProofInputError('unsupported_structured_formula_operator')


def serialize(spec, premises, goal, *, planning, cpu_seconds, memory_mb):
    theory = theory_from_spec(spec)
    symbols = Symbols(theory.predicates)
    labels = {}
    lines = [f'assign(max_seconds,{int(cpu_seconds)}).', f'assign(max_megs,{int(memory_mb)}).']
    if planning:
        lines += ['clear(auto).', 'set(hyper_resolution).', 'clear(unit_deletion).']
    else:
        lines.append('clear(auto_denials).')
    lines += ['assign(max_weight,"1e12").', 'formulas(sos).']
    for i, rule in enumerate(theory.rules):
        formula = '(' + ' & '.join(symbols.atom(a) for a in rule.premises) + ') -> ' + symbols.atom(rule.conclusion)
        variables = {t for a in (*rule.premises, rule.conclusion) for t in a.arguments if t.startswith('?')}
        label = 'r' + str(i)
        labels[label] = ['rule', rule.rule_id]
        lines.append(symbols.closure(formula, variables) + ' # label(' + label + ').')
    for i, (fid, fact) in enumerate(theory.facts.items()):
        label = 'f' + str(i)
        labels[label] = ['fact', fid]
        lines.append(symbols.atom(fact) + ' # label(' + label + ').')
    for i, axiom in enumerate(spec['theory'].get('axioms', [])):
        label = 'a' + str(i)
        expression = render_formula(axiom, symbols, theory.predicates)
        expression = symbols.closure(expression, formula_variables(axiom))
        labels[label] = ['axiom', i]
        lines.append(expression + ' # label(' + label + ').')
    if set(theory.facts) & set(premises):
        raise ProofInputError('duplicate_fact_premise_id')
    if not planning:
        if any(t.startswith('?') for a in list(premises.values()) + ([goal] if goal is not None else []) for t in a.arguments):
            raise ProofInputError('runtime_requires_ground_facts_and_goal')
        for i, (fid, fact) in enumerate(premises.items()):
            label = 'h' + str(i)
            labels[label] = ['premise', fid]
            lines.append(symbols.atom(fact) + ' # label(' + label + ').')
    lines += ['end_of_list.', 'formulas(goals).']
    if planning:
        witness = '?__answer_witness'
        premise_variables = {t for a in premises.values() for t in a.arguments if t.startswith('?')}
        while witness in symbols.variables or witness in premise_variables:
            witness += '_'
        consequent = 'exists ' + symbols.term(witness) + ' (' + symbols.atom(Atom(spec['answer']['predicate'], (witness,))) + ')'
        expression = '(' + ' & '.join(symbols.atom(a) for a in premises.values()) + ') -> (' + consequent + ')' if premises else consequent
        expression = symbols.closure(expression, premise_variables)
    elif goal is None:
        witness = '?__runtime_answer'
        while witness in symbols.variables:
            witness += '_'
        expression = 'exists ' + symbols.term(witness) + ' (' + symbols.atom(Atom(spec['answer']['predicate'], (witness,))) + ')'
    else:
        expression = symbols.atom(goal)
    lines += [expression + ' # label(query).', 'end_of_list.', '']
    return '\n'.join(lines), theory, symbols, labels


def replay_core(theory, premises, answer_predicate, goal, proof, labels, symbols, axioms=()):
    by_line = {r['line']: r for r in proof}
    rules = {r.rule_id: r for r in theory.rules}
    leaves = {**theory.facts, **premises}
    inverse = {v: k for k, v in symbols.constants.items()}
    inverse_predicates = {v: k for k, v in symbols.predicates.items()}
    used_axioms = sorted({labels[tag][1] for row in proof for tag in row['labels']
                          if tag in labels and labels[tag][0]=='axiom'})
    equalities=fixed_equalities([axioms[i] for i in used_axioms])
    def norm(value):
        return Atom(value.predicate,tuple(equalities.get(t,t) for t in value.arguments))
    def clause_rules(row):
        tagged=[labels[t][1] for t in row['labels'] if t in labels and labels[t][0]=='rule']
        if tagged:return tagged
        if row['parents'] and row['justification'].startswith(('copy(','back_rewrite(')):
            return clause_rules(by_line[row['parents'][0]])
        return []
    query_leaves = [r for r in proof if 'query' in r['labels'] and r['justification'].startswith('deny(') and unit(r['formula'])]
    def premise_mappings(index, mapping, leaf_ids):
        if index == len(query_leaves):
            yield mapping, leaf_ids
            return
        row = query_leaves[index]
        raw = unit(row['formula'])
        for pid, source in premises.items():
            if symbols.predicates[source.predicate] != raw.predicate or len(source.arguments) != len(raw.arguments):
                continue
            mapped = dict(mapping)
            for external, original in zip(raw.arguments, source.arguments):
                if external in mapped and equalities.get(mapped[external],mapped[external]) != equalities.get(original,original):
                    break
                if external not in mapped and not original.startswith('?'):
                    break
                if original.startswith('?') and any(k != external and v == original for k, v in mapped.items()):
                    break
                mapped.setdefault(external,original)
            else:
                yield from premise_mappings(index+1, mapped, {**leaf_ids, row['line']: pid})

    def attempt(mapping, leaf_ids):
        derived = {}
        def decode(raw):
            if raw.predicate not in inverse_predicates or any(t not in mapping for t in raw.arguments):
                raise ValueError('unmapped_proof_term')
            return Atom(inverse_predicates[raw.predicate], tuple(mapping[t] for t in raw.arguments))
        for row in proof:
            raw = unit(row['formula'])
            if raw is None or row['justification'] == 'goal':
                continue
            if row['line'] in leaf_ids:
                pid = leaf_ids[row['line']]
                value = decode(raw)
                if norm(value) != norm(premises[pid]):
                    raise ValueError('conditional_premise_decode_mismatch')
                derived[row['line']] = {'atom': value, 'depth': 0, 'premises': {pid}, 'rules': set(), 'steps': []}
                continue
            source = [labels[l] for l in row['labels'] if l in labels and labels[l][0] in ('fact', 'premise')]
            positive_axioms=[labels[tag][1] for tag in row['labels'] if tag in labels and labels[tag][0]=='axiom']
            if row['justification']=='assumption' and len(positive_axioms)==1:
                index=positive_axioms[0]
                value=decode(raw)
                if 'predicate' not in axioms[index] or norm(value)!=norm(atom(axioms[index],theory.predicates)):
                    raise ValueError('unmapped_positive_background_axiom')
                derived[row['line']]={'atom':value,'depth':0,'premises':set(),'rules':set(),'steps':[]}
                continue
            if row['justification'] == 'assumption' and len(source) == 1:
                fid = source[0][1]
                value = decode(raw)
                if norm(value) != norm(leaves[fid]):
                    raise ValueError('fact_decode_mismatch')
                derived[row['line']] = {'atom': value, 'depth': 0, 'premises': {fid}, 'rules': set(), 'steps': []}
                continue
            if row['parents'] and row['justification'].startswith(('back_rewrite(','copy(')):
                parent=derived.get(row['parents'][0])
                actual=decode(raw)
                if parent is None or norm(parent['atom'])!=norm(actual):
                    raise ValueError('uncertified_rewrite_of_positive_parent')
                derived[row['line']]={**parent,'atom':actual}
                continue
            if not row['justification'].startswith('hyper('):
                if row['justification'].startswith(('clausify(', 'deny(')):
                    continue
                raise ValueError('unsupported_positive_inference:' + row['justification'])
            parent_rows = [by_line[n] for n in row['parents']]
            rule_parents = [(r,rid) for r in parent_rows for rid in clause_rules(r)
                            if r['line'] not in derived]
            if len(rule_parents) != 1:
                raise ValueError('hyper_requires_one_domain_rule')
            rule_row, rid = rule_parents[0]
            rule = rules[rid]
            parent_ids = [n for n in row['parents'] if n != rule_row['line']]
            if any(n not in derived for n in parent_ids):
                raise ValueError('unsupported_hyper_parent')
            parents = [derived[n] for n in parent_ids]
            actual = decode(raw)
            matches = [({}, set())]
            for pattern in rule.premises:
                following = []
                for sub, selected in matches:
                    for i, parent in enumerate(parents):
                        found = match(norm(pattern), norm(parent['atom']), sub)
                        if found is not None:
                            following.append((found, selected | {i}))
                matches = following
            valid = [sub for sub, used in matches if used == set(range(len(parents))) and norm(instantiate(rule.conclusion, sub)) == norm(actual)]
            if not valid:
                raise ValueError('actual_domain_rule_instance_replay_failed')
            depth = 1 + max((p['depth'] for p in parents), default=0)
            steps = []
            for parent in parents:
                for step in parent['steps']:
                    if step not in steps:
                        steps.append(step)
            steps.append({'proof_line': row['line'], 'rule_proof_line': rule_row['line'], 'parent_proof_lines': parent_ids,
                          'rule_id': rid, 'substitution': valid[0], 'premises': [wire_atom(p['atom']) for p in parents],
                          'conclusion': wire_atom(actual), 'depth': depth})
            derived[row['line']] = {'atom': actual, 'depth': depth,
                                    'premises': set().union(*(p['premises'] for p in parents)),
                                    'rules': {rid}.union(*(p['rules'] for p in parents)), 'steps': steps}
        root = next(r for r in proof if r['formula'] == '$F')
        witnesses = [derived[n] for n in root['parents'] if n in derived
                     and derived[n]['atom'].predicate == answer_predicate
                     and (goal is None or derived[n]['atom'] == goal)]
        if len(witnesses) != 1:
            raise ValueError('refutation_does_not_supply_positive_answer_witness')
        return {**witnesses[0],'axioms':used_axioms,'equalities':equalities}
    error = 'unmappable_conditional_premises'
    for mapping, leaf_ids in premise_mappings(0, dict(inverse), {}):
        try:
            return attempt(mapping, leaf_ids)
        except ValueError as exc:
            error = str(exc)
    raise ValueError(error)


def runtime_refutation(output):
    block = re.search(r'=+ PROOF =+(.*?)=+ end of proof =+', output, re.S | re.I)
    if not block:
        raise ValueError('missing_proof_block')
    by_line = {}
    def arguments(expr):
        depth = start = 0
        for index, ch in enumerate(expr):
            if ch in '([':
                depth += 1
            elif ch in ')]':
                depth -= 1
            elif ch == ',' and depth == 0:
                yield expr[start:index].strip()
                start = index + 1
        yield expr[start:].strip()

    pattern = r'^\s*(\d+)\s+(.+?)\.\s*\[(.*?)\]\.\s*$'
    for match in re.finditer(pattern, block[1], re.M | re.S):
        number, formula, justification = match.groups()
        formula = ' '.join(formula.split())
        parents = {}
        for inference in arguments(justification):
            if '(' not in inference:
                continue
            name, args = inference.split('(', 1)
            args = args[:-1]
            if name == 'rewrite' and args.startswith('[') and args.endswith(']'):
                args = args[1:-1]
            for argument in arguments(args):
                ref = re.fullmatch(r'(\d+)(?:\(.*\))?', argument)
                if ref:
                    parents[int(ref[1])] = None
        by_line[int(number)] = {'line': int(number), 'formula': formula.split(' #')[0],
                                'labels': re.findall(r'#\s*label\(([^)]+)\)', formula),
                                'justification': justification, 'parents': list(parents)}
    roots = [i for i,r in by_line.items() if r['formula'] == '$F']
    if len(roots) != 1:
        raise ValueError('missing_or_ambiguous_refutation')
    ancestry = set()
    pending = roots.copy()
    while pending:
        line = pending.pop()
        if line not in by_line:
            raise ValueError('missing_proof_parent')
        if line not in ancestry:
            ancestry.add(line)
            pending.extend(by_line[line]['parents'])
    return [r for i,r in by_line.items() if i in ancestry]


def _positive_patterns(value, predicates):
    if 'predicate' in value:
        return [atom(value, predicates)]
    if value.get('op') == 'and' and value.get('args'):
        return [a for child in value['args'] for a in _positive_patterns(child, predicates)]
    raise ValueError('not_positive_conjunction')


def _constraint(value, predicates):
    while value.get('op') == 'forall':
        value = value['body']
    if value.get('op') == 'and':
        return [c for child in value['args'] for c in _constraint(child, predicates)]
    body = []
    if value.get('op') == 'implies':
        body = _positive_patterns(value['left'], predicates)
        value = value['right']
    if value.get('op') != 'not':
        raise ValueError('axiom_outside_decidable_denial_fragment')
    denied = value['arg']
    if denied.get('op') == 'eq':
        left, right = denied['left'], denied['right']
        if not isinstance(left, str) or not isinstance(right, str):
            raise ValueError('invalid_equality_term')
        return [(body, (left, right))]
    return [(body + _positive_patterns(denied, predicates), None)]


def _model_axioms(axioms, theory, closure, tick):
    constants = {t for a in closure for t in a.arguments}
    symbols = Symbols(theory.predicates)
    for rule in theory.rules:
        for a in (*rule.premises, rule.conclusion):
            symbols.atom(a)
    for axiom in axioms:
        render_formula(axiom, symbols, theory.predicates)
    constants.update(symbols.constants)
    domain = sorted(constants) or [None]

    def holds(expr, env):
        tick()
        def term(t):
            return env[t] if t.startswith('?') else t
        if 'predicate' in expr:
            value = atom(expr, theory.predicates)
            return Atom(value.predicate, tuple(term(t) for t in value.arguments)) in closure
        op = expr['op']
        if op == 'eq':
            return term(expr['left']) == term(expr['right'])
        if op == 'not':
            return not holds(expr['arg'], env)
        if op in ('and', 'or'):
            reduce = all if op == 'and' else any
            return reduce(holds(part, env) for part in expr['args'])
        if op == 'implies':
            return not holds(expr['left'], env) or holds(expr['right'], env)
        variables = expr['variables']
        reduce = all if op == 'forall' else any
        return reduce(holds(expr['body'], {**env, **dict(zip(variables, values))})
                      for values in product(domain, repeat=len(variables)))

    for expr in axioms:
        free = sorted(formula_variables(expr))
        if not all(holds(expr, dict(zip(free, values))) for values in product(domain, repeat=len(free))):
            raise ValueError('no_model_certificate_for_background_axiom')


def finite_compatibility(spec, facts, max_atoms=None, max_steps=None):
    try:
        theory = theory_from_spec(spec)
        values = {fid: atom(record.get('formula', record), theory.predicates) for fid, record in facts.items()}
        if any(t.startswith('?') for a in list(theory.facts.values()) + list(values.values()) for t in a.arguments):
            raise ValueError('compatibility_requires_ground_facts')
        constraints, other_axioms = [], []
        axioms = spec['theory'].get('axioms', [])
        def conjuncts(expr):
            if expr.get('op') == 'and':
                for part in expr['args']:
                    yield from conjuncts(part)
            else:
                yield expr
        base=[part for expr in axioms for part in conjuncts(expr)]
        aliases=fixed_equalities(base)
        def representative(term):
            return aliases.get(term,term)
        def normalize(expr):
            if isinstance(expr,list):return [normalize(part) for part in expr]
            if not isinstance(expr,dict):return expr
            if 'predicate' in expr:
                return {**expr,'arguments':[representative(t) for t in expr['arguments']]}
            if expr.get('op')=='eq':
                return {**expr,'left':representative(expr['left']),'right':representative(expr['right'])}
            return {key:normalize(value) for key,value in expr.items()}
        axioms=[normalize(expr) for expr in axioms]
        def norm_atom(a):return Atom(a.predicate,tuple(representative(t) for t in a.arguments))
        theory=Theory(theory.predicates,
                     tuple(Rule(r.rule_id,tuple(norm_atom(a) for a in r.premises),norm_atom(r.conclusion),r.source_kind)
                           for r in theory.rules),{k:norm_atom(a) for k,a in theory.facts.items()})
        values={k:norm_atom(a) for k,a in values.items()}
        axiom_facts={norm_atom(atom(expr,theory.predicates)) for expr in base if 'predicate' in expr and not formula_variables(expr)}
        symbols=Symbols(theory.predicates)
        for rule in theory.rules:
            for a in (*rule.premises,rule.conclusion):symbols.atom(a)
        for expr in axioms:render_formula(expr,symbols,theory.predicates)
        for a in [*theory.facts.values(),*values.values()]:symbols.atom(a)
        domain=list(symbols.constants) or ['@domain_element']
        for value in axioms:
            try:
                constraints.extend(_constraint(value, theory.predicates))
            except ValueError:
                other_axioms.append(value)
    except (KeyError, TypeError, ValueError) as exc:
        return {'status': 'unknown', 'reason': str(exc), 'method': 'finite_positive_closure_and_universal_denials'}
    closure = set(theory.facts.values()) | set(values.values()) | axiom_facts
    origins = {a: {'facts': sorted(fid for fid, other in {**theory.facts, **values}.items() if other == a), 'rules': []} for a in closure}
    steps = 0
    def tick():
        nonlocal steps
        steps += 1
        if max_steps is not None and steps > max_steps:
            raise ValueError('finite_compatibility_step_limit')

    def joins(patterns):
        matches = [({}, [])]
        for pattern in patterns:
            next_matches = []
            for substitution, parents in matches:
                for fact in sorted(closure, key=lambda a: (a.predicate, a.arguments)):
                    tick()
                    found = match(pattern, fact, substitution)
                    if found is not None:
                        next_matches.append((found, parents + [fact]))
            matches = next_matches
        return matches
    try:
        changed = True
        while changed:
            changed = False
            for rule in theory.rules:
                additions = []
                for substitution, parents in joins(rule.premises):
                    remaining=list(dict.fromkeys(t for t in rule.conclusion.arguments
                                                 if t.startswith('?') and t not in substitution))
                    for assignment in product(domain,repeat=len(remaining)):
                        tick()
                        value = instantiate(rule.conclusion, substitution | dict(zip(remaining,assignment)))
                        if value not in closure:
                            additions.append((value, parents))
                for value, parents in additions:
                    if value in closure:
                        continue
                    closure.add(value)
                    origins[value] = {'facts': sorted(set().union(*(set(origins[p]['facts']) for p in parents))),
                                      'rules': sorted({rule.rule_id}.union(*(set(origins[p]['rules']) for p in parents)))}
                    changed = True
                    if max_atoms is not None and len(closure) > max_atoms:
                        raise ValueError('finite_compatibility_atom_limit')
        for index, (patterns, equality) in enumerate(constraints):
            for substitution, parents in joins(patterns):
                if equality is not None:
                    left, right = (substitution.get(t, t) for t in equality)
                    if left.startswith('?') or right.startswith('?'):
                        if left.startswith('?') and right.startswith('?'):
                            same = True
                        else:
                            same = True
                    else:
                        same = left == right
                    if not same:
                        continue
                return {'status': 'inconsistent', 'reason': 'universal_denial_violated', 'constraint_index': index,
                        'substitution': substitution, 'conflicting_atoms': [wire_atom(p) for p in parents],
                        'used_fact_ids': sorted(set().union(*(set(origins[p]['facts']) for p in parents))) if parents else [],
                        'used_rule_ids': sorted(set().union(*(set(origins[p]['rules']) for p in parents))) if parents else [],
                        'method': 'finite_positive_closure_and_universal_denials'}
        if other_axioms:
            _model_axioms(axioms, theory, closure, tick)
    except (KeyError, TypeError, ValueError) as exc:
        return {'status': 'unknown', 'reason': str(exc), 'method': 'finite_positive_closure_and_universal_denials'}
    return {'status': 'compatible', 'reason': 'constructed_finite_model_modulo_asserted_equalities',
            'closure_atoms': len(closure), 'checked_denials': len(constraints),
            'model_checked_axioms': len(axioms) if other_axioms else 0, 'steps': steps,
            'method': 'finite_positive_closure_and_universal_denials',
            'general_FOL_consistency_claimed': False}


class PaperProver:
    def __init__(self, binary, out_dir, wall_seconds=1, cpu_seconds=2, memory_mb=256, total_seconds=10):
        self.binary = str(Path(binary).resolve())
        self.out_dir = Path(out_dir).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.wall_seconds = float(wall_seconds)
        self.cpu_seconds, self.memory_mb = int(cpu_seconds), int(memory_mb)
        if not math.isfinite(self.wall_seconds) or self.wall_seconds <= 0 or self.cpu_seconds < 1 or self.memory_mb < 1:
            raise ValueError('invalid_prover_resource_limit')
        self.total_wall_seconds, self.calls = 0.0, 0
        self.physical_calls, self.replayed_calls = 0, 0
        self._accounted_calls = set()
        self.binary_sha256 = self._hash(Path(self.binary)) if Path(self.binary).is_file() else None
        self.budget_limit_seconds = float(total_seconds)
        if not math.isfinite(self.budget_limit_seconds) or self.budget_limit_seconds < 0:
            raise ValueError('invalid_question_solver_budget')
        self._unknown_attempts = {}
        self.refresh_partial_reservations()

    @property
    def unknown_wall_reservation_seconds(self):
        return sum(self._unknown_attempts.values())

    @property
    def budget_occupancy_seconds(self):
        return self.total_wall_seconds + self.unknown_wall_reservation_seconds

    @property
    def solver_timing_complete(self):
        return not self._unknown_attempts

    def refresh_partial_reservations(self):
        inputs = list(self.out_dir.glob('*.in')) + list(self.out_dir.glob('*__attempts/attempt_*.in'))
        for inp in inputs:
            prefix = inp.with_suffix('')
            if all(prefix.with_suffix(suffix).is_file() for suffix in
                   ('.in', '.out', '.err', '.symbols.json', '.manifest.json', '.result.json', '.receipt.json')):
                continue
            identity = str(prefix)
            if identity in self._unknown_attempts:
                continue
            limit = self.wall_seconds
            manifest_path = prefix.with_suffix('.manifest.json')
            if manifest_path.is_file():
                try:
                    stored = float(json.loads(manifest_path.read_text(encoding='utf-8'))['wall_seconds'])
                    if math.isfinite(stored) and stored > 0:
                        limit = stored
                except (OSError, ValueError, TypeError, KeyError):
                    pass
            self._unknown_attempts[identity] = limit

    def timing_record(self):
        return {'solver_timing_complete': self.solver_timing_complete,
                'known_solver_wall_seconds': self.total_wall_seconds,
                'unknown_wall_reservation_seconds': self.unknown_wall_reservation_seconds,
                'unknown_solver_attempts': len(self._unknown_attempts),
                'solver_budget_occupancy_seconds': self.budget_occupancy_seconds}

    def compatible(self, spec, facts, call_id=None):
        result = finite_compatibility(spec, facts)
        if call_id is not None:
            self._safe_id(call_id)
            path = self.out_dir / (call_id + '.compatibility.json')
            if path.exists():
                if json.loads(path.read_text(encoding='utf-8')) != result:
                    raise ValueError('compatibility_call_id_content_changed')
            else:
                self._write_new(path, result)
            result['artifact_path'] = str(path)
        return result

    def planning(self, spec, candidate, call_id):
        theory = theory_from_spec(spec)
        premises = {pid: atom(value, theory.predicates) for pid, value in candidate['anticipated_premises'].items()}
        if any(a.predicate == spec['answer']['predicate'] for a in premises.values()):
            return self._invalid('answer_predicate_in_anticipated_premises')
        return self._prove(spec, premises, None, call_id, True)

    def runtime(self, spec, facts, goal, call_id):
        theory = theory_from_spec(spec)
        premises = {fid: atom(value.get('formula', value), theory.predicates) for fid, value in facts.items()}
        target = atom(goal, theory.predicates) if goal is not None else None
        if target is not None and target.predicate != spec['answer']['predicate']:
            return self._invalid('runtime_goal_must_use_fixed_answer_predicate')
        if any(a.predicate == spec['answer']['predicate'] for a in premises.values()):
            return self._invalid('answer_predicate_in_acquired_facts')
        compatibility = finite_compatibility(spec, facts)
        if compatibility['status'] == 'inconsistent':
            return {**self._invalid('inconsistent_fact_state', status='INCONSISTENT'), 'compatibility': compatibility}
        return self._prove(spec, premises, target, call_id, False, compatibility=compatibility)

    @staticmethod
    def _safe_id(call_id):
        if not isinstance(call_id, str) or not re.fullmatch(r'[a-zA-Z0-9_-]+', call_id):
            raise ValueError('unsafe_call_id')

    @staticmethod
    def _write_new(path, value):
        with path.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

    @staticmethod
    def _hash(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _manifest(self, script, symbols, wall_seconds):
        current_binary = self._hash(Path(self.binary)) if Path(self.binary).is_file() else None
        if current_binary != self.binary_sha256:
            raise ValueError('pinned_solver_binary_changed')
        return {'version': 'paper_native_solver_receipt_v1', 'binary_sha256': current_binary,
                'adapter_sha256': self._hash(Path(__file__)),
                'input_sha256': hashlib.sha256(script.encode()).hexdigest(),
                'symbols_sha256': hashlib.sha256(json.dumps(symbols, sort_keys=True).encode()).hexdigest(),
                'wall_seconds': wall_seconds, 'cpu_seconds': self.cpu_seconds, 'memory_mb': self.memory_mb}

    def _recover(self, prefix, script, symbols, manifest):
        required = ('.in', '.symbols.json', '.manifest.json', '.out', '.err', '.result.json', '.receipt.json')
        if not all(prefix.with_suffix(suffix).is_file() for suffix in required):
            return None
        stored = json.loads(prefix.with_suffix('.manifest.json').read_text(encoding='utf-8'))
        if stored != manifest:
            return None
        if prefix.with_suffix('.in').read_text(encoding='utf-8') != script:
            raise ValueError('solver_input_integrity_mismatch')
        if json.loads(prefix.with_suffix('.symbols.json').read_text(encoding='utf-8')) != symbols:
            raise ValueError('solver_symbol_integrity_mismatch')
        receipt = json.loads(prefix.with_suffix('.receipt.json').read_text(encoding='utf-8'))
        hashes = {suffix: self._hash(prefix.with_suffix(suffix)) for suffix in required if suffix != '.receipt.json'}
        if receipt.get('hashes') != hashes or receipt.get('manifest') != manifest:
            raise ValueError('solver_completion_receipt_integrity_mismatch')
        result = json.loads(prefix.with_suffix('.result.json').read_text(encoding='utf-8'))
        if not result.get('solver_started') or result.get('exit_code') is None:
            return None
        identity = str(prefix)
        if identity not in self._accounted_calls:
            self._accounted_calls.add(identity)
            self.calls += 1
            self.total_wall_seconds += result['duration_monotonic']
            self.replayed_calls += 1
        result.update(replayed=True, solver_started_this_process=False, original_result_path=str(prefix.with_suffix('.result.json')))
        result.update(self.timing_record())
        return result

    def _select_attempt(self, call_id, script, symbols, manifest):
        original = self.out_dir / call_id
        attempts = self.out_dir / (call_id + '__attempts')
        prefixes = [original]
        if attempts.is_dir():
            prefixes += sorted({p.with_suffix('') for p in attempts.glob('attempt_*.in')})
        for prefix in prefixes:
            recovered = self._recover(prefix, script, symbols, manifest)
            if recovered is not None:
                return prefix, recovered
        if original.with_suffix('.in').exists() and original.with_suffix('.in').read_text(encoding='utf-8') != script:
            raise ValueError('solver_call_id_input_changed')
        if not any(original.with_suffix(suffix).exists() for suffix in ('.in', '.out', '.err', '.result.json')):
            return original, None
        attempts.mkdir(exist_ok=True)
        index = 1
        while any((attempts / ('attempt_' + str(index).zfill(5))).with_suffix(suffix).exists()
                  for suffix in ('.in', '.out', '.err', '.result.json', '.manifest.json')):
            index += 1
        return attempts / ('attempt_' + str(index).zfill(5)), None

    @staticmethod
    def _invalid(error, status='PARSE_ERROR'):
        return {'status': status, 'proof_found': False, 'proof_parsed': False, 'compile_status': 'not_attempted',
                'used_rule_ids': [], 'used_premise_ids': [], 'depth': None, 'answer_template': None,
                'replay': [], 'proof_dag': [], 'raw_input_path': None, 'raw_output_path': None,
                'duration_monotonic': 0.0, 'exit_code': None, 'solver_started': False, 'error': error}

    def _prove(self, spec, premises, goal, call_id, planning, compatibility=None):
        self._safe_id(call_id)
        self.refresh_partial_reservations()
        result = self._invalid(None)
        if compatibility is not None:
            result['compatibility'] = compatibility
        try:
            script, theory, symbols, labels = serialize(spec, premises, goal, planning=planning,
                                                       cpu_seconds=self.cpu_seconds, memory_mb=self.memory_mb)
        except (KeyError, TypeError, ValueError) as exc:
            result['error'] = str(exc)
            return result
        symbol_record = {'predicates': symbols.predicates, 'constants': symbols.constants,
                         'variables': symbols.variables, 'labels': labels}
        remaining = self.budget_limit_seconds - self.budget_occupancy_seconds
        if remaining <= 0:
            result.update(status='TIMEOUT', error='question_solver_budget_including_unknown_wall', **self.timing_record())
            return result
        effective_wall = min(self.wall_seconds, remaining)
        manifest = self._manifest(script, symbol_record, effective_wall)
        prefix, recovered = self._select_attempt(call_id, script, symbol_record, manifest)
        if recovered is not None:
            return recovered
        inp, out, err = (prefix.with_suffix(suffix) for suffix in ('.in', '.out', '.err'))
        with inp.open('x', encoding='utf-8') as stream:
            stream.write(script)
        result.update(raw_input_path=str(inp), raw_output_path=str(out))
        self._write_new(prefix.with_suffix('.symbols.json'), symbol_record)
        self._write_new(prefix.with_suffix('.manifest.json'), manifest)
        start = time.monotonic()
        try:
            with out.open('x', encoding='utf-8') as stdout, err.open('x', encoding='utf-8') as stderr:
                process = subprocess.Popen([self.binary, '-f', str(inp)], stdout=stdout, stderr=stderr,
                                           cwd=self.out_dir)
                result['solver_started'] = True
                self.calls += 1
                self.physical_calls += 1
                self._accounted_calls.add(str(prefix))
                try:
                    code = process.wait(timeout=effective_wall)
                    result.update(exit_code=code, status=classify_exit(code))
                except subprocess.TimeoutExpired:
                    process.kill()
                    result.update(exit_code=process.wait(), status='TIMEOUT', error='external_wall_watchdog')
        except OSError as exc:
            result.update(status='SOLVER_ERROR', error=str(exc))
        duration = time.monotonic() - start
        self.total_wall_seconds += duration
        result['duration_monotonic'] = duration
        output = out.read_text(encoding='utf-8', errors='replace') if out.exists() else ''
        stderr_text = err.read_text(encoding='utf-8', errors='replace') if err.exists() else ''
        if result['exit_code'] == 1:
            result['status'] = classify_exit(1, output, stderr_text)
        if result['status'] == 'SOLVER_ERROR' and re.search(r'out of memory|cannot allocate memory|malloc.*fail|failed to map segment', output + '\n' + stderr_text, re.I):
            result.update(status='MEMORY_LIMIT', error='os_address_space_limit_allocation_failure')
        result['proof_found'] = result['status'] == 'PROVED' and 'THEOREM PROVED' in output
        if result['status'] == 'PROVED' and not result['proof_found']:
            result.update(status='SOLVER_ERROR', error='exit_zero_without_theorem')
        result['diagnostics'] = {'internal_cpu_seconds': self.cpu_seconds, 'external_wall_seconds': effective_wall,
                                 'solver_max_megs': self.memory_mb,
                                 'planning_goal': ('universal_H_implies_exists_answer' if planning else
                                                   'runtime_ground_witness_search' if goal is None else 'concrete_answer'),
                                 'proof_check': ('executable_rule_instances' if planning else 'query_in_refutation_ancestry'),
                                 'FOL_axioms_preserved': True}
        if result['proof_found']:
            try:
                proof = runtime_refutation(output)
                result.update(proof_dag=proof, proof_parsed=True)
                if planning:
                    replay = replay_core(theory,premises,spec['answer']['predicate'],goal,proof,labels,symbols,
                                         spec['theory'].get('axioms',[]))
                    result.update(compile_status='compiled', used_rule_ids=sorted(replay['rules']),
                                  used_axiom_indices=replay['axioms'],fixed_equalities=replay['equalities'],
                                  used_premise_ids=sorted(replay['premises']), depth=replay['depth'],
                                  answer_template=wire_atom(replay['atom']), replay=replay['steps'])
                else:
                    query = next((r for r in proof if r['justification'] == 'goal' and 'query' in r['labels']), None)
                    if query is None:
                        result.update(status='INCONSISTENT', proof_found=False, error='refutation_independent_of_goal')
                    elif goal is not None and query['formula'] != symbols.atom(goal):
                        raise ValueError('concrete_query_mismatch')
                    else:
                        sources = {tuple(labels[tag]) for row in proof for tag in row['labels'] if tag in labels}
                        result.update(used_rule_ids=sorted(i for kind,i in sources if kind == 'rule'),
                                      used_premise_ids=sorted(i for kind,i in sources if kind in ('fact','premise')),
                                      used_axiom_indices=sorted(i for kind,i in sources if kind == 'axiom'),
                                      concrete_goal_proved=goal is not None, existential_goal_proved=goal is None,
                                      answer_template=wire_atom(goal) if goal is not None else None)
                        if goal is None:
                            pred = symbols.predicates[spec['answer']['predicate']]
                            terms = [a.arguments[0] for row in proof if (a := unit(row['formula']))
                                     and a.predicate == pred and len(a.arguments) == 1]
                            inverse = {encoded:name for name,encoded in symbols.constants.items()}
                            result['named_witness_candidates'] = list(dict.fromkeys(
                                [inverse[t] for t in terms if t in inverse] + list(symbols.constants)))
            except ValueError as exc:
                if not planning:
                    result.update(proof_found=False, status='UNVERIFIED_RUNTIME_PROOF')
                result.update(compile_status='unsupported_proof' if result['proof_parsed'] else 'parse_failed', error=str(exc))
        result.update(self.timing_record())
        self._write_new(prefix.with_suffix('.result.json'), result)
        suffixes = ('.in', '.symbols.json', '.manifest.json', '.out', '.err', '.result.json')
        self._write_new(prefix.with_suffix('.receipt.json'), {'manifest': manifest,
            'hashes': {suffix: self._hash(prefix.with_suffix(suffix)) for suffix in suffixes}})
        return result
