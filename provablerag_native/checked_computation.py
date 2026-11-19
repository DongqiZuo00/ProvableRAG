from copy import deepcopy
from datetime import date
from decimal import Decimal, InvalidOperation
import hashlib
import re


CONTRACT_VERSION = 'paper_checked_computation_v3'
DEFAULT_INPUTS = {'compare_date': [0, 1], 'earlier': [0, 1], 'compare_year': [0, 1], 'less_year': [0, 1],
                  'compare_number': [0, 1], 'less_number': [0, 1], 'year': [0], 'equal': [0, 1], 'not_equal': [0, 1]}
MONTHS = {}
for i, name in enumerate(('January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
                           'September', 'October', 'November', 'December'), 1):
    MONTHS[name.lower()] = MONTHS[name[:3].lower()] = i
MONTHS['sept'] = 9


class InputIssue(ValueError):
    def __init__(self, reason, status='unknown'):
        self.reason, self.status = reason, status
        super().__init__(reason)


def parse_full_date(value):
    if not isinstance(value, str):
        raise InputIssue('invalid_date')
    text = ' '.join(value.strip().split())
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}', text):
        try:
            return date.fromisoformat(text)
        except ValueError:
            raise InputIssue('invalid_date') from None
    if re.fullmatch(r'\d{1,4}|\d{4}-\d{1,2}', text) or re.fullmatch(r'[A-Za-z]+\.?\s+\d{4}', text):
        raise InputIssue('partial_date')
    if re.search(r'\d\s*/\s*\d|\bor\b|[–—]', text, re.I):
        raise InputIssue('ambiguous_date')
    month_first = re.fullmatch(r'([A-Za-z]+)\.?\s+(\d{1,2})(st|nd|rd|th)?(?:,\s*|\s+)(\d{4})', text, re.I)
    day_first = re.fullmatch(r'(\d{1,2})(st|nd|rd|th)?\s+([A-Za-z]+)\.?(?:,\s*|\s+)(\d{4})', text, re.I)
    if month_first:
        month, day, ordinal, year = month_first.groups()
    elif day_first:
        day, ordinal, month, year = day_first.groups()
    else:
        raise InputIssue('unsupported_date_format')
    if month.lower() not in MONTHS:
        raise InputIssue('invalid_date_month')
    day = int(day)
    correct = 'th' if 10 < day % 100 < 14 else {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
    if ordinal and ordinal.lower() != correct:
        raise InputIssue('invalid_date_ordinal')
    try:
        return date(int(year), MONTHS[month.lower()], day)
    except ValueError:
        raise InputIssue('invalid_date') from None


def input_terms(operation):
    computation = operation.get('computation', {})
    kind = computation.get('kind')
    if kind not in DEFAULT_INPUTS:
        raise InputIssue('unsupported_checked_computation:' + str(kind))
    declaration = computation.get('inputs', DEFAULT_INPUTS[kind])
    count = 1 if kind == 'year' else 2
    if not isinstance(declaration, list) or len(declaration) != count:
        raise InputIssue('invalid_computation_input_count')
    args = operation['formula']['arguments']
    expanded = []
    for position,value in enumerate(declaration):
        if type(value) is int:
            if not 0 <= value < len(args):
                raise InputIssue('invalid_input_position')
            expanded.append(args[value])
        elif isinstance(value, str) and value and value != '?':
            expanded.append(value)
        elif isinstance(value,dict) and set(value)=={'node','position'}:
            planned=operation.get('input_terms',[])
            sources=operation.get('operand_sources',[])
            if (not isinstance(value['node'],str) or type(value['position']) is not int or value['position']<0
                    or len(planned)!=count or len(sources)!=count or not sources[position]):
                raise InputIssue('uncompiled_source_selector')
            for row in sources[position]:
                if (row.get('selected_node',row.get('node_id'))!=value['node']
                        or row.get('argument_index')!=value['position']):
                    raise InputIssue('source_selector_mismatch','rejected')
            expanded.append(planned[position])
        else:
            raise InputIssue('invalid_computation_input_term')
    terms = operation.get('input_terms', expanded)
    if not isinstance(terms, list) or len(terms) != count or any(not isinstance(t, str) or not t or t == '?' for t in terms):
        raise InputIssue('invalid_operation_input_terms')
    return list(terms)


def selected_sources(operation, facts):
    allowed, pending = set(), list(operation.get('requires', []))
    while pending:
        sid = pending.pop()
        if sid in allowed:
            continue
        allowed.add(sid)
        record = facts.get(sid, {})
        pending.extend(record.get('input_fact_ids', []))
        pending.extend(record.get('premise_ids', []))
    for item in operation.get('derived_inputs', []):
        suppliers = set(item.get('supplier_ids', []))
        if suppliers <= allowed and facts.get(item['node_id'],{}).get('rule_execution'):
            allowed.add(item['node_id'])
    return allowed


def source_descriptors(operation, term, position):
    per_operand=operation.get('operand_sources')
    result = per_operand[position] if per_operand is not None else operation.get('input_sources', {}).get(term, [])
    if isinstance(result, list) and len(result) == 2 and isinstance(result[0], str) and type(result[1]) is int:
        result = [{'node_id': result[0], 'argument_index': result[1]}]
    if not isinstance(result, list) or any(not isinstance(r, dict) for r in result):
        raise InputIssue('invalid_input_sources')
    return result


def exact_question_span(text, question):
    if not question or not isinstance(question.get('question'), str):
        return None
    content = question['question']
    match = re.search(r'(?<!\w)' + re.escape(text) + r'(?!\w)', content)
    if match is None:
        return None
    return {'source_kind': 'question_literal', 'question_id': question.get('id'),
            'question_sha256': hashlib.sha256(content.encode('utf-8')).hexdigest(),
            'span': [match.start(), match.end()], 'text': text}


def resolve_input(term, operation, bindings, facts, question, allowed, position):
    variable = term.startswith('?')
    bound = bindings.get(term) if variable else term
    if bound is not None and (not isinstance(bound, str) or not bound or bound.startswith('?')):
        raise InputIssue('ungrounded_input')
    if facts is None:
        if bound is None:
            raise InputIssue('ungrounded_input')
        return bound, []
    refs = source_descriptors(operation, term, position)
    values, receipts = [], []
    scope = operation.get('scope', 'static')
    for ref in refs:
        sid = ref.get('node_id', ref.get('given_id'))
        index = ref.get('argument_index')
        if not isinstance(sid, str) or type(index) is not int:
            raise InputIssue('invalid_input_source_slot')
        record = facts.get(sid)
        if record is None:
            raise InputIssue('missing_input_source')
        given = record.get('given') is True
        if not given and sid not in allowed:
            raise InputIssue('unselected_input_source', 'rejected')
        if given and ref.get('source_kind') != 'given':
            raise InputIssue('input_not_declared_as_fixed_theory_given')
        if record.get('status') not in (None, 'accepted', 'resolved', 'proved'):
            raise InputIssue('input_source_not_established')
        if record.get('scope', 'static') != scope or ref.get('scope', scope) != scope:
            raise InputIssue('input_scope_mismatch', 'rejected')
        args = record.get('formula', {}).get('arguments', [])
        if not 0 <= index < len(args):
            raise InputIssue('invalid_input_source_slot')
        value = args[index]
        if not isinstance(value, str) or not value or value.startswith('?'):
            raise InputIssue('ungrounded_input_source')
        values.append(value)
        receipts.append({'node_id': sid, 'argument_index': index, 'scope': scope,
                         'source_kind': 'given' if given else 'supplier'})
    if values:
        if len(set(values)) != 1 or bound is not None and bound != values[0]:
            raise InputIssue('input_value_mismatch', 'rejected')
        return values[0], receipts
    if variable:
        raise InputIssue('missing_input_source' if bound is not None else 'ungrounded_input')
    observed = exact_question_span(term, question)
    if observed is None or scope != 'static':
        raise InputIssue('unobserved_literal_input')
    return term, [observed]


def normalized(kind, raw):
    if kind in ('compare_date', 'earlier', 'year'):
        value = parse_full_date(raw)
        return value, value.isoformat(), 'day'
    if kind in ('compare_year', 'less_year'):
        if not re.fullmatch(r'\d{1,4}', raw.strip()) or not 1 <= int(raw) <= 9999:
            raise InputIssue('invalid_year')
        value = int(raw)
        return value, str(value), 'year'
    if kind in ('compare_number', 'less_number'):
        if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', raw.strip()):
            raise InputIssue('invalid_number')
        try:
            value = Decimal(raw.strip())
        except InvalidOperation:
            raise InputIssue('invalid_number') from None
        if not value.is_finite():
            raise InputIssue('nonfinite_number')
        if abs(value.adjusted()) > 4096:
            return value, str(value), 'number'
        text = format(value, 'f')
        if '.' in text:
            text = text.rstrip('0').rstrip('.')
        return value, '0' if value == 0 else text, 'number'
    return raw, raw, 'text'


def equality_operands(raw):
    if raw[0]==raw[1]:
        return [(v,v,'identical_literal') for v in raw]
    for kind in ('compare_number','compare_date'):
        try:
            return [normalized(kind,v) for v in raw]
        except InputIssue:
            continue
    raise InputIssue('no_certified_identity_comparison')


def inputs_available(operation, bindings, supplied_facts, question=None):
    if not isinstance(operation, dict) or not isinstance(bindings, dict) or not isinstance(supplied_facts, dict):
        return False
    try:
        terms = input_terms(operation)
        allowed = selected_sources(operation, supplied_facts)
        for position,term in enumerate(terms):
            resolve_input(term, operation, bindings, supplied_facts, question, allowed, position)
        return True
    except (InputIssue, KeyError, TypeError, ValueError, IndexError):
        return False


def run(operation, bindings, supplied_facts=None, question=None):
    computation = operation.get('computation', {})
    kind = computation.get('kind')
    receipt = {'contract_version': CONTRACT_VERSION, 'kind': kind, 'input_terms': [], 'original_inputs': [],
               'normalized_inputs': [], 'input_precision': [], 'input_sources': [], 'input_source_ids': [],
               'given_input_source_ids': [], 'outcome': None, 'arithmetic_only': supplied_facts is None}
    atom = None
    def result(status, reason):
        return {'status': status, 'reason': reason, 'formula': deepcopy(atom),
                'value': ' '.join(atom['arguments']) if atom else '', 'evidence_ids': [],
                'premise_ids': list(operation.get('requires', [])), 'computation': deepcopy(receipt),
                'verification': 'deterministic_checked_computation',
                'admission_eligible': status == 'accepted' and supplied_facts is not None}
    try:
        terms = input_terms(operation)
        receipt['input_terms'] = terms
        allowed = selected_sources(operation, supplied_facts or {})
        local = dict(bindings)
        values = []
        for position,term in enumerate(terms):
            raw, sources = resolve_input(term, operation, bindings, supplied_facts, question, allowed, position)
            receipt['original_inputs'].append(raw)
            receipt['input_sources'].append(sources)
            receipt['input_source_ids'] = sorted({r['node_id'] for group in receipt['input_sources'] for r in group if 'node_id' in r})
            receipt['given_input_source_ids'] = sorted({r['node_id'] for group in receipt['input_sources'] for r in group if r.get('source_kind') == 'given'})
            if term.startswith('?'):
                local[term] = raw
            value, text, precision = normalized(kind, raw)
            values.append(value)
            receipt['normalized_inputs'].append(text)
            receipt['input_precision'].append(precision)
        if kind in ('equal','not_equal'):
            compared=equality_operands(receipt['original_inputs'])
            values=[row[0] for row in compared]
            receipt['normalized_inputs']=[row[1] for row in compared]
            receipt['input_precision']=[row[2] for row in compared]
        atom = {'predicate': operation['formula']['predicate'],
                'arguments': [local.get(t, t) for t in operation['formula']['arguments']]}
        if kind == 'year':
            if len(atom['arguments']) < 2:
                raise InputIssue('year_requires_output_argument_1')
            output = str(values[0].year)
            output_term = operation['formula']['arguments'][1]
            existing = bindings.get(output_term) if output_term.startswith('?') else output_term
            if existing is not None and existing != output:
                raise InputIssue('output_value_mismatch', 'rejected')
            atom['arguments'][1] = output
            receipt.update(output_argument_index=1, output_value=output)
            outcome = True
        elif kind == 'equal':
            outcome = values[0] == values[1]
        elif kind == 'not_equal':
            outcome = values[0] != values[1]
        else:
            outcome = values[0] < values[1]
        if any(not isinstance(t, str) or not t or t.startswith('?') for t in atom['arguments']):
            raise InputIssue('ungrounded_computation_head')
        receipt['outcome'] = outcome
        return result('accepted' if outcome else 'rejected', 'checked_true' if outcome else 'checked_false')
    except InputIssue as error:
        atom = None
        return result(error.status, error.reason)
    except (KeyError, TypeError, ValueError, IndexError):
        atom = None
        return result('unknown', 'invalid_checked_contract')
