from .util import canonical, ground, is_ground, unify
from .planning import _unify, _atom, fixed_equalities


def apply_rule(spec, operation, bindings, supplied):
    rule = next(r for r in spec["theory"]["admitted_rules"] if r["id"] == operation["rule_id"])
    equalities=fixed_equalities(spec['theory'].get('axioms',[]))
    def instances(index, substitution, parents):
        if index == len(rule["premises"]):
            yield substitution, parents
            return
        pattern = rule["premises"][index]
        for fid in operation["rule_source_ids"]:
            record = supplied.get(fid)
            if record is None or record.get("scope", "static") != operation["scope"]:
                continue
            found = _unify(pattern, record["formula"], substitution,equalities=equalities)
            if found is not None:
                yield from instances(index+1,found,[*parents,fid])
    conclusions = {}
    for substitution, parents in instances(0, {}, []):
        formula = ground(rule["conclusion"], substitution)
        found=_unify(operation["formula"],formula,bindings,equalities=equalities)
        if is_ground(formula) and found is not None:
            formula=_atom(operation["formula"],found)
            conclusions.setdefault(canonical(_atom(formula,equalities)), (formula, parents))
            if len(conclusions)>1:
                break
    if len(conclusions) != 1:
        return {"status":"unknown", "reason":"rule_needs_unique_supported_ground_instance"}
    formula, parents = next(iter(conclusions.values()))
    return {"status":"accepted", "formula":formula, "value":canonical(formula),
            "rule_id":rule["id"], "premise_ids":list(dict.fromkeys(parents)), "rule_execution":True,
            "evidence_ids":sorted({e for fid in parents for e in supplied[fid].get("evidence_ids", [])}),
            "verification":"ground_instance_of_admitted_rule"}
