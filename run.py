from pathlib import Path
import os
import sys

PACKAGE = Path(__file__).resolve().parent
WORKSPACE = Path(os.environ.get("PROVABLERAG_WORKSPACE", ".")).expanduser().resolve()
REFERENCE = WORKSPACE / "outputs/provablerag_reproduction"
sys.path[:0] = [str(PACKAGE), str(PACKAGE / "runtime_common")]
import provablerag
if (REFERENCE / "provablerag").is_dir():
    provablerag.__path__.append(str(REFERENCE / "provablerag"))
from provablerag_native import token_counting
from provablerag_native.config import prover9_path
token_counting.RELEASE = WORKSPACE / "outputs/provablerag_paper_reference"


if __name__ == "__main__":
    from provablerag_native.__main__ import main
    main(WORKSPACE)
