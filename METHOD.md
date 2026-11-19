# ProvableRAG: conditional planning and grounded execution

**Input** A question *q*, a document corpus *D*, and planning and execution budgets.  
**Output** A concrete answer with its proof and supporting records, or an unresolved result.

## Algorithm 1

Transcribed from the paper, page 19. Underscores mark subscripts. For a short overview of how the code implements the algorithm, see the [English tutorial](TUTORIAL.md).

The fixed theory Γ contains facts given in the question and admitted inference rules. Q(a) means that *a* answers the question. For a candidate plan, H contains the facts that retrieval or checked computation would have to establish. F contains the ground facts actually accepted on the current branch.

```text
Algorithm 1 ProvableRAG: conditional planning and grounded execution
Require: Question q, corpus 𝒟, planning and execution budgets
 1: Formalize q to obtain Γ_q, Q_q(a); construct G_q
 2: Initialize an empty list of verified plans
 3: for each structurally admissible candidate ℋ within the planning budget do
 4:     Use Prover9 to search for π satisfying Equation (3)
 5:     if a proof is returned and compiles to an executable plan P_π then
 6:         Store P_π with its proof and used premises
 7:     end if
 8: end for
 9: Sort stored plans by increasing D(π), breaking ties by candidate order
10: F_0 ← ∅; t ← 0; initialize plan bindings and attempt records
11: loop
12:     if Prover9 derives Q_q(a) from Γ_q ∪ F_t for a concrete a then
13:         T ← t; return answer a and supporting records
14:     end if
15:     Select a ready operation with consistent bindings from the first viable plan
        whose operation fits the remaining execution budget
16:     if no such operation exists then
17:         T ← t; return unresolved
18:     end if
19:     Execute retrieval or reasoning to obtain a candidate fact
20:     Check its support, grounding, and compatibility with Γ_q ∪ F_t
21:     Update F_{t+1} by Equation (1)
22:     Pass accepted facts and bindings; update plan and attempt records
23:     t ← t + 1
24: end loop
```

The conditional proof establishes that H would suffice for an answer under every assignment of its variables. The concrete proof uses Γ and the accepted facts Fₜ. A failed acquisition counts as an attempt and leaves the branch's fact set unchanged.

## How the code implements the algorithm

The single-question entry point is `run_question`. On the default path it builds the specification, enumerates candidates, constructs the prover and evidence acquirer, and calls `Executor.run`. [pipeline.py](provablerag_native/pipeline.py)

| Algorithm 1 lines | Actual implementation |
|---|---|
| 1 · Formalize the question | `build_spec` first asks for information needs and assigns stable node IDs. A second call uses those IDs to propose relations, prerequisite groups and goal groups. A separate translation call writes the graph as node formulas, Γ and Q. `validate_spec` checks the structured result before planning. An existing specification can be supplied through `--fixed-spec`. [formalization.py](provablerag_native/formalization.py) |
| 2–4 · Enumerate and prove candidates | `iter_candidates` starts from each goal group and recursively chooses producers and input suppliers. All REQUIRES predecessors are mandatory. Members of a prerequisite group are conjunctive, and groups are alternatives. `PaperProver.planning` submits the universally closed conditional goal to Prover9 before retrieval. [planning.py](provablerag_native/planning.py) · [proving.py](provablerag_native/proving.py) |
| 5–9 · Compile and rank | `replay_core` recovers the answer expression, used premises, rule instances and depth from the returned proof. `compile_operations` traces dependencies and adds the inputs required to execute those steps. `Executor.prepare` records eligibility; `Executor.rank` orders compatible eligible plans by depth and original candidate order. Depth counts the longest chain of rule applications. [proving.py](provablerag_native/proving.py) · [compiler.py](provablerag_native/compiler.py) · [execution.py](provablerag_native/execution.py) |
| 10–14 · Check for a concrete answer | `Executor.run` starts with an empty fact set. `Executor.completion` tries concrete answer terms through `PaperProver.runtime` before the first acquisition and after each accepted fact. A successful result contains the proved answer, the proof file, and the identifiers of its supporting facts, rules, axioms and evidence. [execution.py](provablerag_native/execution.py) · [proving.py](provablerag_native/proving.py) |
| 15–20 · Select and execute an operation | `Executor.run` checks dependencies, binding consistency, previous attempts and resource limits. In retrieval mode, `EvidenceAcquirer.acquire` builds a query, retrieves passages, proposes a fact and independently checks it against the cited sentences. In reasoning mode, `apply_rule` applies an admitted rule, or `checked_computation.run` evaluates a registered operation on established inputs. [execution.py](provablerag_native/execution.py) · [acquisition.py](provablerag_native/acquisition.py) · [rule_execution.py](provablerag_native/rule_execution.py) · [checked_computation.py](provablerag_native/checked_computation.py) |
| 20–24 · Update facts or try alternatives | The executor checks the candidate against the node formula, shared bindings, mutual exclusions and the current theory. Once admitted, the fact carries its source IDs and input records. Alternative assignments have separate branch state. When every viable choice is exhausted, the executor returns an unresolved status and a null answer. [execution.py](provablerag_native/execution.py) |

The paper specifies the two proof obligations and their scheduling. This implementation uses a model to translate free text into formulas, local BM25 for corpus search, and a separate model call to assess cited evidence. The proof certifies derivability under the fixed translation and accepted evidence. A conditional proof that the compiler cannot map to executable operations remains ineligible.
