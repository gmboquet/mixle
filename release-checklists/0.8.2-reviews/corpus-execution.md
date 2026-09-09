# Corpus execution against the 0.8.2 wheel

- **Date:** 2026-09-08/09.
- **Artifact under test:** `mixle-0.8.2-py3-none-any.whl` built from this tree
  (sha256 `ac8f0aab…`), installed into a venv with the optional extras. Every run was launched from
  `/tmp` with `PYTHONPATH` unset, so `import mixle` resolves to the installed distribution and not
  to the source checkout.

## Examples — 57 of 57 exit 0

Every `examples/*.py` was executed against the installed wheel. No failures.

## Notebooks — 130 of 131 execute; 1 needs a JVM this host does not have

The `mixle-notebooks` corpus at `d99d296` (131 notebooks) was executed against the same wheel.

| Notebook | Outcome |
|---|---|
| `notebooks/tutorials/estimation_using_spark.ipynb` | **cannot run on this host.** `PySparkRuntimeError: [JAVA_GATEWAY_EXITED]`. `/usr/bin/java` here is the macOS stub ("Unable to locate a Java Runtime"), and `/usr/libexec/java_home` finds none, which is the same absence that makes the test suite skip its Spark cases ("pyspark or a usable JVM is not available"). Not a library defect and not evidence either way. |
| `notebooks/applications/malware_certificate_embedding.ipynb` | timed out at the runner's 900-second per-cell limit on the first pass, which ran while a full test suite was competing for the machine; re-run on a quiet host (see the receipt for the outcome). |

The remaining 129 executed cleanly. The first pass took about 20 hours of wall clock because the
machine was running test suites at the same time; the per-notebook seconds recorded in
`notebooks_run.jsonl` are inflated by that contention and are not performance evidence.
