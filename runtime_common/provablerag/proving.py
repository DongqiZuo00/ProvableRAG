from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import time
from typing import Mapping


@dataclass(frozen=True)
class Atom:
    predicate: str
    arguments: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "arguments", tuple(self.arguments))


@dataclass(frozen=True)
class Rule:
    rule_id: str
    premises: tuple[Atom, ...]
    conclusion: Atom
    source_kind: str = "question_definition"

    def __post_init__(self):
        object.__setattr__(self, "premises", tuple(self.premises))


@dataclass
class Theory:
    predicates: dict[str, int]
    rules: tuple[Rule, ...]
    facts: dict[str, Atom] = field(default_factory=dict)

    @property
    def version_hash(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


@dataclass
class ProofResult:
    status: str
    proof_found: bool = False
    proof_parsed: bool = False
    compile_status: str = "not_attempted"
    used_rule_ids: list[str] = field(default_factory=list)
    used_premise_ids: list[str] = field(default_factory=list)
    depth: int | None = None
    answer_template: Atom | None = None
    raw_input_path: str | None = None
    raw_output_path: str | None = None
    duration_monotonic: float = 0.0
    exit_code: int | None = None
    solver_started: bool = False
    proof_dag: list[dict] = field(default_factory=list)
    replay: list[dict] = field(default_factory=list)
    error: str | None = None
    diagnostics: dict = field(default_factory=dict)


SAFE = re.compile(r"^[a-z][a-zA-Z0-9_]*$")


def diagnose_literals(proposals):
    positive, negative, invalid = set(), set(), []
    for i, item in enumerate(proposals):
        if not isinstance(item, dict) or not isinstance(item.get('predicate'), str) or not isinstance(item.get('arguments'), (list, tuple)):
            invalid.append(i)
            continue
        args = item['arguments']
        if not SAFE.fullmatch(item['predicate']) or any(not isinstance(t,str) or t.startswith('?') for t in args):
            invalid.append(i)
            continue
        key = (item['predicate'], tuple(args), item.get('scope','static'))
        (negative if item.get('negated', False) else positive).add(key)
    conflicts = sorted(positive & negative)
    return {'consistency_status':'inconsistent' if conflicts else 'unknown',
            'compiler_status':'unsupported' if negative or invalid else 'supported_positive_literals',
            'explicit_conflicts':[{'predicate':p,'arguments':list(a),'scope':s} for p,a,s in conflicts],
            'invalid_literal_indices':invalid, 'changes_candidate_acceptance':False}


def classify_exit(code, output='', stderr=''):
    if code == 1 and re.search(r'%%ERROR:|sread_term error|read_term error|syntax error|parse error|bad character|closing parenthesis expected', output + '\n' + stderr, re.I):
        return 'PARSE_ERROR'
    return {0:'PROVED',1:'SOLVER_ERROR',2:'SEARCH_EXHAUSTED',3:'MEMORY_LIMIT',4:'TIMEOUT',
            5:'OTHER_LIMIT',6:'OTHER_LIMIT',7:'OTHER_LIMIT'}.get(code,'SOLVER_ERROR')


def validate(theory: Theory, premises: Mapping[str, Atom], goal: Atom):
    def atom(a):
        if not isinstance(a, Atom) or not SAFE.fullmatch(a.predicate):
            raise ValueError("unsupported_predicate")
        if a.predicate not in theory.predicates or len(a.arguments) != theory.predicates[a.predicate]:
            raise ValueError("undeclared_predicate_or_arity")
        if any(not isinstance(t, str) or not t or (t.startswith('?') and not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',t[1:])) for t in a.arguments):
            raise ValueError("invalid_term")
    ids = [r.rule_id for r in theory.rules]
    if len(ids) != len(set(ids)) or set(theory.facts) & set(premises):
        raise ValueError("duplicate_source_id")
    for r in theory.rules:
        if r.source_kind not in {"question_definition", "question_explicit", "tested_library", "tested_rule_library", "deterministic_contract"}:
            raise ValueError("unadmitted_rule_source")
        if not r.premises:
            raise ValueError("empty_rule_body_use_given_fact")
        for a in (*r.premises, r.conclusion):
            atom(a)
        bound = {x for a in r.premises for x in a.arguments if x.startswith('?')}
        if any(x.startswith('?') and x not in bound for x in r.conclusion.arguments):
            raise ValueError("unsafe_rule_variable")
    for a in list(theory.facts.values()) + list(premises.values()) + [goal]:
        atom(a)
    if any(x.startswith('?') for a in theory.facts.values() for x in a.arguments):
        raise ValueError("given_facts_must_be_ground")
    allowed = {x for a in premises.values() for x in a.arguments if x.startswith('?')}
    if any(x.startswith('?') and x not in allowed for x in goal.arguments):
        raise ValueError("ungroundable_answer_template")
    if any(a.predicate == goal.predicate for a in premises.values()):
        raise ValueError("answer_predicate_in_anticipated_premises")


class Symbols:
    def __init__(self):
        self.variables = {}
        self.constants = {}

    def term(self, term):
        if term.startswith('?'):
            return self.variables.setdefault(term, 'x' + str(len(self.variables)))
        return self.constants.setdefault(term, 'c' + str(len(self.constants)))

    def atom(self, atom):
        args = ','.join(self.term(t) for t in atom.arguments)
        return atom.predicate + ('(' + args + ')' if args else '')

    def closure(self, formula, variables):
        return ''.join('all ' + self.term(v) + ' ' for v in sorted(variables)) + '(' + formula + ')'


def serialize(theory, premises, goal, *, planning, cpu_seconds, memory_mb):
    validate(theory, premises, goal)
    s = Symbols()
    labels = {}
    lines = [f'assign(max_seconds,{int(cpu_seconds)}).', f'assign(max_megs,{int(memory_mb)}).',
             'clear(auto).', 'set(hyper_resolution).', 'clear(unit_deletion).',
             'assign(max_weight,1000).', 'formulas(sos).']
    for i, rule in enumerate(theory.rules):
        formula = '(' + ' & '.join(s.atom(a) for a in rule.premises) + ') -> ' + s.atom(rule.conclusion)
        variables = {t for a in (*rule.premises, rule.conclusion) for t in a.arguments if t.startswith('?')}
        label = 'r' + str(i)
        labels[label] = ['rule', rule.rule_id]
        lines.append(s.closure(formula, variables) + ' # label(' + label + ').')
    for i, (fact_id, fact) in enumerate(theory.facts.items()):
        label = 'f' + str(i)
        labels[label] = ['fact', fact_id]
        lines.append(s.atom(fact) + ' # label(' + label + ').')
    if not planning:
        if any(t.startswith('?') for a in premises.values() for t in a.arguments) or any(t.startswith('?') for t in goal.arguments):
            raise ValueError('runtime_requires_ground_facts_and_goal')
        for i, (pid, a) in enumerate(premises.items()):
            label = 'h' + str(i)
            labels[label] = ['premise', pid]
            lines.append(s.atom(a) + ' # label(' + label + ').')
    lines.extend(['end_of_list.', 'formulas(goals).'])
    if planning:
        if len(goal.arguments) != 1:
            raise ValueError('planning_requires_unary_answer_condition')
        answer_var = 'answer_witness'
        while '?' + answer_var in s.variables:
            answer_var += '_fresh'
        quantified_answer = 'exists ' + s.term('?' + answer_var) + ' ' + s.atom(Atom(goal.predicate, ('?' + answer_var,)))
        body = '(' + ' & '.join(s.atom(a) for a in premises.values()) + ') -> (' + quantified_answer + ')' if premises else quantified_answer
        variables = {t for a in list(premises.values()) + [goal] for t in a.arguments if t.startswith('?')}
        lines.append(s.closure(body, variables) + ' # label(query).')
    else:
        lines.append(s.atom(goal) + ' # label(query).')
    lines.extend(['end_of_list.', ''])
    return '\n'.join(lines), s, labels


def parse_proof(output):
    blocks = re.findall(r'=+ PROOF =+(.*?)=+ end of proof =+', output, re.S | re.I)
    if not blocks:
        raise ValueError('missing_proof_block')
    records = []
    for m in re.finditer(r'^\s*(\d+)\s+(.+?)\.\s*\[([^\]]*)\]\.', blocks[0], re.M | re.S):
        number, formula, justification = m.groups()
        formula = ' '.join(formula.split())
        labels = re.findall(r'#\s*label\(([^)]+)\)', formula)
        parents = [int(v) for v in re.findall(r'(?<=[(,])\s*(\d+)(?=\s*[,)]|\s*$)', justification)]
        records.append({'line': int(number), 'formula': formula.split(' #')[0],
                        'labels': labels, 'justification': justification, 'parents': list(dict.fromkeys(parents))})
    by_line = {r['line']: r for r in records}
    roots = [r['line'] for r in records if r['formula'].strip() == '$F']
    if len(roots) != 1:
        raise ValueError('missing_or_ambiguous_refutation')
    used = set()
    def visit(n):
        if n in used:
            return
        if n not in by_line:
            raise ValueError('missing_proof_parent')
        used.add(n)
        for p in by_line[n]['parents']:
            visit(p)
    visit(roots[0])
    return [r for r in records if r['line'] in used]


def match(pattern: Atom, fact: Atom, substitution):
    if pattern.predicate != fact.predicate or len(pattern.arguments) != len(fact.arguments):
        return None
    result = dict(substitution)
    for p, f in zip(pattern.arguments, fact.arguments):
        if p.startswith('?'):
            if p in result and result[p] != f:
                return None
            result[p] = f
        elif p != f:
            return None
    return result


def instantiate(atom, sub):
    return Atom(atom.predicate, tuple(sub.get(t, t) for t in atom.arguments))


def unit(formula):
    m = re.fullmatch(r"([a-z][a-zA-Z0-9_]*)(?:\(([^()]*)\))?", formula.strip())
    if not m:
        return None
    return Atom(m[1], tuple(m[2].split(',')) if m[2] else ())


def compile_core(theory, premises, goal, proof_dag, labels, symbols):
    by_line = {r['line']: r for r in proof_dag}
    rules = {r.rule_id: r for r in theory.rules}
    leaves = {**theory.facts, **premises}
    inverse = {v: k for k, v in symbols.constants.items()}
    query_leaves = [r for r in proof_dag if 'query' in r['labels'] and r['justification'].startswith('deny(') and unit(r['formula'])]
    choices = [(dict(inverse), {})]
    for row in query_leaves:
        raw = unit(row['formula'])
        next_choices = []
        for mapping, leaf_ids in choices:
            for pid, source in premises.items():
                if source.predicate != raw.predicate or len(source.arguments) != len(raw.arguments):
                    continue
                mapped = dict(mapping)
                valid = True
                for external, original in zip(raw.arguments, source.arguments):
                    if external in mapped and mapped[external] != original:
                        valid = False
                    if external not in mapped and not original.startswith('?'):
                        valid = False
                    if original.startswith('?') and any(k != external and v == original for k, v in mapped.items()):
                        valid = False
                    mapped[external] = original
                if valid:
                    next_choices.append((mapped, {**leaf_ids, row['line']: pid}))
        choices = next_choices
        if not choices:
            raise ValueError('cannot_map_goal_skolem_premise')
        if len(choices) > 256:
            raise ValueError('ambiguous_skolem_mapping_limit')
    def attempt(mapping, leaf_ids):
        derived = {}
        def decode(raw):
            if any(t not in mapping for t in raw.arguments):
                raise ValueError('unmapped_proof_term')
            return Atom(raw.predicate, tuple(mapping[t] for t in raw.arguments))
        for row in proof_dag:
            raw = unit(row['formula'])
            if raw is None or row['justification'] == 'goal':
                continue
            if row['line'] in leaf_ids:
                pid = leaf_ids[row['line']]
                a = decode(raw)
                if a != premises[pid]:
                    raise ValueError('premise_decode_mismatch')
                derived[row['line']] = {'atom': a, 'depth': 0, 'premises': {pid}, 'rules': set(), 'steps': []}
                continue
            source_labels = [labels[l] for l in row['labels'] if l in labels and labels[l][0] in ('fact', 'premise')]
            if row['justification'] == 'assumption' and len(source_labels) == 1:
                pid = source_labels[0][1]
                a = decode(raw)
                if a != leaves[pid]:
                    raise ValueError('assumption_decode_mismatch')
                derived[row['line']] = {'atom': a, 'depth': 0, 'premises': {pid}, 'rules': set(), 'steps': []}
                continue
            if not row['justification'].startswith('hyper('):
                if row['justification'].startswith(('clausify(', 'deny(')):
                    continue
                raise ValueError('unsupported_positive_inference:' + row['justification'])
            parent_rows = [by_line[n] for n in row['parents']]
            rule_parents = [(r, labels[l][1]) for r in parent_rows for l in r['labels'] if l in labels and labels[l][0] == 'rule']
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
                        m = match(pattern, parent['atom'], sub)
                        if m is not None:
                            following.append((m, selected | {i}))
                matches = following
            valid = [sub for sub, selected in matches if selected == set(range(len(parents))) and instantiate(rule.conclusion, sub) == actual]
            if not valid:
                raise ValueError('actual_hyper_instance_replay_failed')
            sub = valid[0]
            depth = 1 + max(p['depth'] for p in parents)
            steps = []
            for p in parents:
                for step in p['steps']:
                    if step not in steps:
                        steps.append(step)
            steps.append({'proof_line':row['line'], 'rule_proof_line':rule_row['line'], 'parent_proof_lines':parent_ids,
                          'rule_id':rid,'substitution':sub,'premises':[asdict(p['atom']) for p in parents],
                          'conclusion':asdict(actual),'depth':depth})
            derived[row['line']] = {'atom':actual,'depth':depth,'premises':set().union(*(p['premises'] for p in parents)),
                                    'rules':{rid}.union(*(p['rules'] for p in parents)), 'steps':steps}
        root = next(r for r in proof_dag if r['formula'] == '$F')
        candidates = [derived[n] for n in root['parents'] if n in derived and derived[n]['atom'] == goal]
        if len(candidates) != 1:
            raise ValueError('refutation_does_not_supply_requested_answer_witness')
        return candidates[0]
    errors = []
    for mapping, leaf_ids in choices:
        try:
            return attempt(mapping, leaf_ids)
        except ValueError as exc:
            errors.append(str(exc))
    raise ValueError(errors[0] if errors else 'proof_replay_failed')


class Prover9:
    def __init__(self, binary, output_dir, wall_seconds=1.0, cpu_seconds=2, memory_mb=256, os_memory_mb=512):
        self.binary = str(Path(binary).resolve())
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.wall_seconds = float(wall_seconds)
        self.cpu_seconds = int(cpu_seconds)
        self.memory_mb = int(memory_mb)
        self.os_memory_mb = None if os_memory_mb is None else int(os_memory_mb)
        if self.os_memory_mb is not None and self.os_memory_mb < 1:
            raise ValueError('OS_address_space_limit_must_be_positive_or_null')
        self.total_wall_seconds = 0.0

    def planning(self, theory, premises, answer_template, call_id):
        return self._prove(theory, premises, answer_template, call_id, True)

    def runtime(self, theory, facts, goal, call_id):
        return self._prove(theory, facts, goal, call_id, False)

    def diagnose_empty(self, theory, goal, call_id):
        result = self.runtime(theory, {}, goal, call_id)
        return {'empty_premises_prove_goal':result.proof_found,
                'diagnostic_status':'proved' if result.proof_found else 'unknown',
                'proof_result':asdict(result),'changes_candidate_acceptance':False}

    def _prove(self, theory, premises, goal, call_id, planning):
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', call_id):
            raise ValueError('unsafe_call_id')
        prefix = self.output_dir / call_id
        if prefix.with_suffix('.in').exists():
            raise ValueError('solver_call_id_already_used')
        result = ProofResult(status='PARSE_ERROR', answer_template=goal)
        try:
            script, symbols, labels = serialize(theory, premises, goal, planning=planning,
                                                cpu_seconds=self.cpu_seconds, memory_mb=self.memory_mb)
        except ValueError as exc:
            result.error = str(exc)
            return result
        inp, out, err = (prefix.with_suffix(s) for s in ('.in', '.out', '.err'))
        inp.write_text(script, encoding='utf-8')
        result.raw_input_path, result.raw_output_path = str(inp), str(out)
        prefix.with_suffix('.symbols.json').write_text(json.dumps({'constants': symbols.constants, 'variables': symbols.variables, 'labels': labels}, indent=2))
        start = time.monotonic()
        try:
            with out.open('w', encoding='utf-8') as stdout, err.open('w', encoding='utf-8') as stderr:
                memory_guard = None
                if self.os_memory_mb is not None and os.name == 'posix':
                    import resource
                    limit = self.os_memory_mb * 1024 * 1024
                    def memory_guard():
                        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
                child = subprocess.Popen([self.binary, '-f', str(inp)], stdout=stdout, stderr=stderr,
                                         cwd=prefix.parent, preexec_fn=memory_guard)
                result.solver_started = True
                try:
                    code = child.wait(timeout=max(0.000001, self.wall_seconds))
                    result.exit_code = code
                    result.status = classify_exit(code)
                except subprocess.TimeoutExpired:
                    child.kill()
                    result.exit_code = child.wait()
                    result.status = 'TIMEOUT'
                    result.error = 'external_wall_watchdog'
        except OSError as exc:
            result.status = 'SOLVER_ERROR'
            result.error = str(exc)
        result.duration_monotonic = time.monotonic() - start
        self.total_wall_seconds += result.duration_monotonic
        output = out.read_text(encoding='utf-8', errors='replace') if out.exists() else ''
        if result.exit_code == 1:
            result.status = classify_exit(result.exit_code, output, err.read_text(errors='replace'))
        stderr_text = err.read_text(errors='replace') if err.exists() else ''
        if self.os_memory_mb is not None and result.status == 'SOLVER_ERROR' and re.search(r'out of memory|cannot allocate memory|malloc.*fail|memory allocation.*fail|failed to map segment', output+'\n'+stderr_text,re.I):
            result.status = 'MEMORY_LIMIT'
            result.error = 'os_address_space_limit_allocation_failure'
        cpu = re.search(r'User_CPU=([0-9.]+), System_CPU=([0-9.]+)',output)
        megs = re.search(r'Megabytes=([0-9]+(?:\.[0-9]+)?)',output)
        result.diagnostics.update(resource_limits={'internal_cpu_seconds':self.cpu_seconds,
            'external_wall_seconds':self.wall_seconds,'solver_max_megs':self.memory_mb,
            'os_rss_limit_mb':None,'os_address_space_limit_mb':self.os_memory_mb if os.name=='posix' else None,
            'per_question_wall_enforcement':'executor'},
            solver_user_cpu_seconds=float(cpu[1]) if cpu else None,
            solver_system_cpu_seconds=float(cpu[2]) if cpu else None,
            solver_reported_megabytes=float(megs[1]) if megs else None)
        result.proof_found = result.status == 'PROVED' and 'THEOREM PROVED' in output
        if result.status == 'PROVED' and not result.proof_found:
            result.status = 'SOLVER_ERROR'
            result.error = 'exit_zero_without_proof'
        if result.proof_found:
            try:
                result.proof_dag = parse_proof(output)
                result.proof_parsed = True
                replay = compile_core(theory, premises, goal, result.proof_dag, labels, symbols)
                result.used_rule_ids = sorted(replay['rules'])
                result.used_premise_ids = sorted(replay['premises'])
                result.depth = replay['depth']
                result.replay = replay['steps']
                result.compile_status = 'compiled'
            except ValueError as exc:
                result.compile_status = 'unsupported_proof' if result.proof_parsed else 'parse_failed'
                result.error = str(exc)
        prefix.with_suffix('.result.json').write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding='utf-8')
        return result
