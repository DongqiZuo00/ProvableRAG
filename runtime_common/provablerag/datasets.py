from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import hashlib
import json


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    metadata: dict

    @classmethod
    def from_public(cls, record):
        if not isinstance(record, dict) or set(record) - {"id", "question", "metadata"}:
            raise ValueError("public_question_has_unapproved_fields")
        metadata = record.get("metadata", {})
        if not isinstance(metadata, dict) or set(metadata) - {"dataset", "split", "type", "level"}:
            raise ValueError("public_metadata_has_unapproved_fields")
        question = record.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("empty_question")
        return cls(str(record.get("id", canonical_hash(question)[:20])), question, metadata)


def load_source(path):
    path = Path(path)
    if path.suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line]
    return json.loads(path.read_text(encoding="utf-8-sig"))


def normalize_row(row, index, dataset):
    question = row.get("question", row.get("Question"))
    answer = row.get("answer", row.get("Answer"))
    if not question or answer is None:
        raise ValueError("source_missing_question_or_answer")
    qid = str(row.get("_id", row.get("id", dataset + ":" + str(index))))
    metadata = {"dataset": dataset}
    for k in ("type", "level"):
        if k in row:
            metadata[k] = row[k]
    labels = {"question_id": qid, "answers": [answer] if isinstance(answer, str) else answer}
    if row.get("Answer 2"):
        labels["answers"].append(row["Answer 2"])
    for key in ("answer_aliases", "supporting_facts", "answer_id", "evidences", "evidences_id"):
        if key in row:
            labels[key] = row[key]
    if "answer_aliases" in labels:
        labels["answers"] = list(dict.fromkeys(labels["answers"] + labels["answer_aliases"]))
    passages = []
    if "context" in row:
        for title, sentences in row["context"]:
            if isinstance(sentences, str):
                sentences = [sentences]
            passages.append({"title": title, "sentences": [{"sentence_id": i, "text": s} for i, s in enumerate(sentences)]})
    for paragraph in row.get("paragraphs", []):
        passages.append({"title": paragraph["title"], "sentences": [{"sentence_id": 0, "text": paragraph["paragraph_text"]}]})
    for passage in passages:
        passage["id"] = "p_" + canonical_hash(passage)[:24]
    return asdict(Question(qid, question, metadata)), labels, passages


def prepare_dataset(source_path, out_dir, dataset, source_url, seed=20261113, test_n=1000, dev_n=100, license_name=None, aliases_path=None):
    if test_n < 1 or dev_n < 0:
        raise ValueError("invalid_split_size")
    rows = load_source(source_path)
    normalized = [normalize_row(r, i, dataset) for i, r in enumerate(rows)]
    aliases = {}
    if aliases_path:
        with Path(aliases_path).open(encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                if entry["Q_id"] in aliases:
                    raise ValueError("duplicate_alias_id")
                aliases[entry["Q_id"]] = entry
    if len({q[0]["id"] for q in normalized}) != len(rows):
        raise ValueError("duplicate_question_id")
    normalized.sort(key=lambda x: hashlib.sha256(f'{seed}:{x[0]["id"]}'.encode()).hexdigest())
    n_test = min(test_n, len(rows))
    n_dev = min(dev_n, len(rows) - n_test)
    selected = normalized[:n_test + n_dev]
    questions, labels, corpus = [], [], {}
    split_ids = {"test": [], "dev": []}
    for i, (q, label, passages) in enumerate(selected):
        split = "test" if i < n_test else "dev"
        q["metadata"]["split"] = split
        Question.from_public(q)
        questions.append(q)
        label["split"] = split
        if label.get("answer_id") in aliases:
            entry = aliases[label["answer_id"]]
            label["answers"] = list(dict.fromkeys(label["answers"] + sorted(set(entry.get("aliases", []) + entry.get("demonyms", [])))))
        labels.append(label)
        split_ids[split].append(q["id"])
        for p in passages:
            corpus[p["id"]] = p
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "public_questions.jsonl", sorted(questions, key=lambda q: q["id"]))
    write_jsonl(out / "evaluation" / "labels.jsonl", sorted(labels, key=lambda q: q["question_id"]))
    write_jsonl(out / "corpus.jsonl", [corpus[k] for k in sorted(corpus)])
    manifest = {"protocol": "rewrite_v1", "dataset": dataset, "source_url": source_url,
                "source_sha256": sha256(source_path), "source_rows": len(rows), "license": license_name,
                "seed": seed, "selection": "SHA256(seed:id) ascending; first test_n then dev_n",
                "test_n": n_test, "dev_n": n_dev, "requested_test_n": test_n, "requested_dev_n": dev_n,
                "ids": {k: sorted(v) for k, v in split_ids.items()}, "corpus_passages": len(corpus),
                "corpus_protocol": "union_of_selected_dev_and_test_candidate_passages; content_hash_dedup",
                "corpus_sentence_policy": "preserve supplied sentences or store an unsegmented paragraph at index 0",
                "historical_protocol_recovered": False, "prepared_not_evaluated": True,
                "files": {f: sha256(out / f) for f in ("public_questions.jsonl", "evaluation/labels.jsonl", "corpus.jsonl")}}
    if aliases_path:
        wanted = {label.get("answer_id") for label in labels}
        alias_file = "evaluation/id_aliases.jsonl"
        write_jsonl(out / alias_file, [aliases[key] for key in sorted(wanted - {None}) if key in aliases])
        manifest["alias_source_sha256"] = sha256(aliases_path)
        manifest["files"][alias_file] = sha256(out / alias_file)
        manifest["alias_policy"] = "official answer_id aliases + demonyms, only evaluation process"
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
