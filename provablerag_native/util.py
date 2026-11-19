from dataclasses import asdict, is_dataclass
from pathlib import Path
import hashlib
import json
import os


def plain(value):
    return asdict(value) if is_dataclass(value) else value


def canonical(value):
    return json.dumps(plain(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as out:
        out.write(json.dumps(plain(value), ensure_ascii=False, indent=2) + "\n")
        out.flush()
        os.fsync(out.fileno())
    temp.replace(path)


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as out:
        out.write(canonical(value) + "\n")
        out.flush()
        os.fsync(out.fileno())


def ground(atom, bindings):
    return {"predicate": atom["predicate"], "arguments": [bindings.get(t, t) for t in atom["arguments"]]}


def is_ground(atom):
    return all(not t.startswith("?") for t in atom["arguments"])


def unify(pattern, fact, bindings=None):
    if pattern["predicate"] != fact["predicate"] or len(pattern["arguments"]) != len(fact["arguments"]):
        return None
    result = dict(bindings or {})
    for term, value in zip(pattern["arguments"], fact["arguments"]):
        if term.startswith("?"):
            if term in result and result[term] != value:
                return None
            result[term] = value
        elif term != value:
            return None
    return result


def closure(spec, facts, limit=4096):
    known = {}
    for fid, record in facts.items():
        known[canonical(record["formula"])] = {**record, "fact_id": fid}
    for item in spec["theory"].get("given_facts", []):
        known[canonical(item["formula"])] = {"fact_id": item["id"], "formula": item["formula"], "premise_ids": [], "evidence_ids": [], "given": True}
    changed = True
    while changed:
        changed = False
        for rule in spec["theory"]["admitted_rules"]:
            choices = [({}, [])]
            for pattern in rule["premises"]:
                following = []
                for sub, parents in choices:
                    for record in list(known.values()):
                        found = unify(pattern, record["formula"], sub)
                        if found is not None:
                            following.append((found, parents + [record]))
                choices = following
            for sub, parents in choices:
                atom = ground(rule["conclusion"], sub)
                key = canonical(atom)
                if is_ground(atom) and key not in known:
                    known[key] = {"fact_id": "derived:" + digest([rule["id"], atom])[:16], "formula": atom,
                        "rule_id": rule["id"], "premise_ids": [p["fact_id"] for p in parents],
                        "evidence_ids": sorted({e for p in parents for e in p.get("evidence_ids", [])}),
                        "rule_substitution": sub}
                    changed = True
                    if len(known) >= limit:
                        return list(known.values())
    return list(known.values())
