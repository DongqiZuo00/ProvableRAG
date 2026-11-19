# ProvableRAG source files

| File | Purpose |
|---|---|
| [run.py](run.py) | Locates the runtime and starts the single-question command-line interface. |
| [provablerag_native/__main__.py](provablerag_native/__main__.py) | Reads the supplied question and corpus, selects the model backend, and writes the run result. |
| [provablerag_native/__init__.py](provablerag_native/__init__.py) | Defines the method name and implementation version. |
| [provablerag_native/pipeline.py](provablerag_native/pipeline.py) | Coordinates formalization, planning, and execution for one question and records the answer and resource usage. |
| [provablerag_native/published_prompts.py](provablerag_native/published_prompts.py) | Defines the prompts for graph construction, retrieval queries, candidate facts, and answer proposals. |
| [provablerag_native/formalization.py](provablerag_native/formalization.py) | Builds the graph in two stages and validates the fixed background theory Γ and answer condition Q. |
| [provablerag_native/planning.py](provablerag_native/planning.py) | Enumerates candidate plans by choosing a retrieval, computation, given fact, or rule for each required fact. |
| [provablerag_native/proving.py](provablerag_native/proving.py) | Calls Prover9 for conditional and concrete proofs, recovers the rule instances used, and checks fact compatibility. |
| [provablerag_native/compiler.py](provablerag_native/compiler.py) | Traces a conditional proof back to its premises and compiles their required operations and dependencies. |
| [provablerag_native/execution.py](provablerag_native/execution.py) | Tries verified plans in proof-depth order, tracks facts and bindings for each branch, and checks for a concrete answer proof. |
| [provablerag_native/acquisition.py](provablerag_native/acquisition.py) | Retrieves passages, proposes candidate facts, and checks each candidate against cited sentences in a separate model call. |
| [provablerag_native/citations.py](provablerag_native/citations.py) | Shortens passage and sentence IDs for a model request and maps them back to the original citations. |
| [provablerag_native/rule_execution.py](provablerag_native/rule_execution.py) | Applies an admitted rule to established premises under consistent variable bindings. |
| [provablerag_native/checked_computation.py](provablerag_native/checked_computation.py) | Resolves operands from their recorded sources and performs date, number, and equality checks. |
| [provablerag_native/config.py](provablerag_native/config.py) | Defines per-question resource limits and locates the Prover9 executable. |
| [provablerag_native/budget.py](provablerag_native/budget.py) | Defines the exception raised when the remaining allowance cannot cover a request or operation. |
| [provablerag_native/token_counting.py](provablerag_native/token_counting.py) | Counts DeepSeek prompt tokens with the pinned tokenizer and estimates the input allowance a request needs. |
| [provablerag_native/funding_session.py](provablerag_native/funding_session.py) | Verifies billing records for the optional paid backend and waits for pending requests before closing the session. |
| [provablerag_native/vllm.py](provablerag_native/vllm.py) | Runs structured model requests and records token usage. |
| [provablerag_native/util.py](provablerag_native/util.py) | Provides JSON storage, hashing, variable binding, and forward application of rules to ground facts. |
| [runtime_common/provablerag/__init__.py](runtime_common/provablerag/__init__.py) | Exposes the shared runtime version. |
| [runtime_common/provablerag/datasets.py](runtime_common/provablerag/datasets.py) | Validates public question records and prepares questions, corpus passages, and separate evaluation labels. |
| [runtime_common/provablerag/models.py](runtime_common/provablerag/models.py) | Sends chat-completion requests and provides shared types for model settings, responses, and usage. |
| [runtime_common/provablerag/proving.py](runtime_common/provablerag/proving.py) | Defines atoms, rules, and theories and provides low-level Prover9 serialization and proof parsing. |
| [runtime_common/provablerag/retrieval.py](runtime_common/provablerag/retrieval.py) | Ranks local corpus passages with BM25 and returns their text, sentence IDs, and scores. |
| [validation/__init__.py](validation/__init__.py) | Marks the package that stores single-question run records. |
| [validation/records.py](validation/records.py) | Hashes source files, preserves run records, and locks the output directory against concurrent writers. |
