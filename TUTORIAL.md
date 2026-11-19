# How ProvableRAG works

For one question and a corpus, `run_question` coordinates four steps. [Entry point](run.py) / [Pipeline](provablerag_native/pipeline.py)

1. **Formalize the question.** The model first identifies the facts the question needs, then connects those nodes. It translates the graph into logical formulas, using shared variables where two facts must refer to the same entity or value. The background theory and the condition that defines an answer stay fixed for the rest of the run. [formalization.py](provablerag_native/formalization.py)
2. **Verify and compile plans.** The planner works backward from a proposed goal and selects a way to obtain each required fact. Before retrieval, Prover9 checks whether those facts, if established, would imply an answer. The compiler follows the returned proof to build executable operations and their dependencies. The executor tries verified plans in order of increasing proof depth, breaking ties by candidate order. [planning.py](provablerag_native/planning.py) / [proving.py](provablerag_native/proving.py) / [compiler.py](provablerag_native/compiler.py)
3. **Establish facts.** The executor selects an operation whose inputs are ready. It retrieves passages and checks a proposed fact against the cited sentences, or derives a fact using an admitted rule or checked computation. It passes accepted facts and their bindings to dependent operations. Alternative bindings stay on separate branches. [acquisition.py](provablerag_native/acquisition.py) / [execution.py](provablerag_native/execution.py)
4. **Prove the answer.** Before the first operation and after each fact update, the executor asks Prover9 whether the fixed theory and the accepted facts prove a concrete answer. A successful run returns that value with its proof and supporting records. If no viable operation fits the remaining budget and no answer is proved, the run returns unresolved. [execution.py](provablerag_native/execution.py) / [proving.py](provablerag_native/proving.py)

## Algorithm 1

Transcribed from page 19 of the paper. Underscores mark subscripts.

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
