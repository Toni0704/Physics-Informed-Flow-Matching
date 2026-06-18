# Vendored third-party dependency: PCFM

`PCFM-main/` in this folder is a **verbatim copy** of the PCFM (Physics-Constrained
Flow Matching) repository, included here so this experiment directory is fully
self-contained and reproducible offline (PCFM is not a pip-installable package).

- **Upstream:** https://github.com/cpfpengfei/PCFM
- **Copyright:** © 2025 Pengfei Cai (Learning Matter @ MIT) and Utkarsh (Julia Lab @ MIT)
- **License:** MIT for the authors' own code; portions adapted from Amazon's
  "ECI-Sampling" project are under the Apache License 2.0. The original
  `LICENSE`, `LICENSE-APACHE-2.0`, and `NOTICE` files are preserved unchanged
  inside `PCFM-main/`.

**No source files were modified.** This is an unaltered redistribution; all of the
work in *this* study lives in `../src/` and `../experiments/`, which import PCFM as
an external library (see `../src/__init__.py::add_pcfm_to_path`). The only addition
is an empty `PCFM-main/datasets/data/` directory used as the data output location;
generated datasets are not committed.

If you would rather not vendor it, delete this `external/` folder and point the
`PCFM_REPO_PATH` environment variable at your own clone instead — the code resolves
either location automatically.
