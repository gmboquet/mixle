# Corpus execution against the final 0.8.2 candidate wheel

- **Date:** 2026-09-15.
- **Artifact under test:** `mixle-0.8.2-py3-none-any.whl` built with the pinned build closure from
  `release/0.8.2` at `f72a933c` (sha256 `a105b1c0…`). Its embedded source content digest is
  `84c1cd26…`, computed over `pyproject.toml`, `setup.py` and every `mixle/**/*.{json,py,pyx}`;
  the commits that follow it change release records, and the rehearsal pre-release `v0.8.2rc1` also
  changes the version string (D-0216); the final candidate's wheel carries this same digest. Installed into a venv with the optional extras the corpus uses
  (numba, torch, pyspark on Java 17, and the sibling `mixle-pde`, `mixle-sim` and `mixle-physics`
  packages the physics notebooks import). Every run was launched from `/tmp` with `PYTHONPATH` unset,
  and the interpreter's `mixle.__path__`, distribution version and embedded source commit were
  recorded before the first run, so what executed is the installed wheel and not a source checkout.
- **Corpus:** `mixle-notebooks` `release/0.8.2` at `eb85a99` (131 notebooks), which carries the six
  notebook repairs of the ten-pass review, and `examples/` from the candidate commit (57 scripts).

## Examples — 57 of 57 exit 0

Every `examples/*.py` ran to completion against the installed wheel, one at a time, in 5 minutes.

## Notebooks — 131 of 131 execute to completion

Each notebook was executed in place from its own directory with `jupyter nbconvert --execute`, two
at a time, with a per-cell limit of 3600 seconds (9000 for `malware_certificate_embedding`) and
`OMP/OPENBLAS/MKL_NUM_THREADS=2`, `PYTHONHASHSEED=0`. The run took 0.9 hours of wall clock; the
longest notebooks were `malware_certificate_embedding` (1915 s), `model_based_embeddings` (1353 s)
and `seismic_full_waveform_inversion` (368 s).

No notebook failed, timed out or was skipped.

The executed copies are committed to `mixle-notebooks` `release/0.8.2` (`7e37784`) as the corpus's
stored outputs, so the numbers a reader sees in a notebook are this wheel's. That closes Q08-F14,
whose stored outputs still showed three repaired defects' pre-repair numbers.

## Superseded

The record this page replaces measured the corpus at `d99d296` against the 0.8.2 wheel of
2026-09-08/09 (`ac8f0aab…`): 57 of 57 examples, 130 of 131 notebooks, the Spark tutorial failing for
want of a JVM on that host. The ten-pass review then ran the corpus again on its own candidate
(`866078be`, 131 of 131). Both measured trees that the repairs made since have changed.
