# ProvableRAG

ProvableRAG takes a public question and a corpus. It constructs a graph of information needs, fixes a background theory and an answer condition, and uses Prover9 to check which plans could establish an answer. During execution, the system retrieves and verifies evidence or applies rules and checked computations. It returns an answer with a concrete proof, or an unresolved result.

## Inputs and requirements

ProvableRAG uses Python 3.10 or later on Linux or WSL. Install Prover9 on the executable search path or set `PROVER9` to its location.

The question JSON requires a `question` string. Each corpus JSONL record contains a passage `id`, a `title`, and a `sentences` array of objects with `sentence_id` and `text`. Resource limits can be supplied with `--config`. A frozen graph and theory can be supplied with `--fixed-spec`.

[English tutorial](TUTORIAL.md) · [Algorithm and implementation](METHOD.md) · [Source file descriptions](INSTRUCTIONS.md)
