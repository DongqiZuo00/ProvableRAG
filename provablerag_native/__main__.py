import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import time

from . import METHOD, VERSION
from .config import Config, prover9_path
from .util import digest


@contextmanager
def model_session(args, funded_config, role, root):
    if args.backend == "funded":
        from provablerag.models import ModelConfig
        from .funding_session import attested_funded_pool
        config = ModelConfig(**json.loads(funded_config.read_bytes()))
        report = root / "sessions" / f"{role}-{time.time_ns()}.json"
        with attested_funded_pool(config, role, report_path=report) as model:
            yield model
    else:
        from .vllm import VLLMConfig, VLLMModel
        config = VLLMConfig(base_url=args.base_url, exact_model_id=args.model, format_mode=args.format,
                            deployment_revision=args.deployment_revision)
        yield VLLMModel(config, root / "vllm.sqlite3")


def main(workspace):
    from provablerag.datasets import Question
    from provablerag.retrieval import LocalBM25
    from .pipeline import run_question
    from validation.records import freeze, locked_writer, read, runtime_sources, sha

    parser = argparse.ArgumentParser(description="Run ProvableRAG on a question and its candidate corpus.")
    sub = parser.add_subparsers(dest="command", required=True)
    single = sub.add_parser("run", help="one public question, one complete proof-producing invocation")
    single.add_argument("--public-question", required=True, type=Path)
    single.add_argument("--corpus", required=True, type=Path)
    single.add_argument("--fixed-spec", type=Path, help="enumerate and prove plans from a supplied question graph and fixed theory")
    single.add_argument("--backend", choices=("funded", "vllm"))
    single.add_argument("--base-url")
    single.add_argument("--model")
    single.add_argument("--format", choices=("json_object", "json_schema"), default="json_object")
    single.add_argument("--deployment-revision", default="")
    single.add_argument("--proposal-only", action="store_true",
                        help="request an answer proposal and independent review before returning a proved answer")
    single.add_argument("--config", type=Path, help="optional JSON resource settings")
    single.add_argument("--out", type=Path, required=True)
    single.add_argument("--solver", type=Path, default=prover9_path(workspace))
    args = parser.parse_args()
    funded_config = workspace / "outputs/provablerag_reproduction/configs/deepseek_benchmark.json"
    if args.backend is None:
        args.backend = "vllm" if args.model or not funded_config.is_file() else "funded"
    if args.backend == "funded" and not funded_config.is_file():
        parser.error("funded model configuration is missing")
    if args.backend == "vllm" and not (args.base_url and args.model):
        parser.error("the vLLM backend requires --base-url and --model")
    if not args.solver.is_file():
        parser.error("supply a Prover9 executable via --solver, PROVER9, or PATH")
    root = args.out.resolve()
    question = Question.from_public(read(args.public_question))
    fixed_spec = read(args.fixed_spec) if args.fixed_spec is not None else None
    profile = Config(**(read(args.config) if args.config else {}))
    corpus = LocalBM25(args.corpus)
    with locked_writer(root):
        inputs = dict(method=METHOD, version=VERSION, question=asdict(question), corpus=corpus.snapshot_hash,
                      config=asdict(profile), solver=sha(args.solver), backend=args.backend, proposal_only=args.proposal_only,
                      fixed_spec_sha256=digest(fixed_spec) if fixed_spec is not None else None,
                      transport=[args.base_url,args.model,args.format,args.deployment_revision] if args.backend=="vllm" else
                      sha(funded_config))
        inputs["sources"] = runtime_sources()
        freeze(root/"invocation.json", inputs)
        if (root/"completion.json").exists():
            freeze(root/"completion.json", {"result_sha256":sha(root/"result.json")})
            result = read(root/"result.json")
        else:
            with model_session(args, funded_config, "inference", root) as model:
                result = run_question(question, model, corpus, args.solver, root, profile,
                    namespace=VERSION+":"+digest([str(root),inputs])[:24], fixed_spec=fixed_spec,
                    proposal_only=args.proposal_only)
            freeze(root/"completion.json", {"result_sha256":sha(root/"result.json")})
    print(json.dumps({k:result.get(k) for k in ("method","answer_status","final_answer","proof_verified","search_calls")},ensure_ascii=False))


if __name__ == "__main__":
    from run import WORKSPACE
    main(WORKSPACE)
