# Deploying a pysparkplug model

A minimal, production-shaped deployment of a **pysparkplug** model: a FastAPI server over
`pysp.inference.ModelService`, a container image, and Kubernetes manifests. This is for the
**classical / probabilistic models pysp builds** (CPU, one-shot scoring, retrain-on-drift) — it is *not*
an LLM serving stack (those use GPU-sharded engines like vLLM/TGI/TensorRT-LLM; the orchestration concepts
here transfer, the inference engine does not).

## Pieces

| file | role |
|---|---|
| `app.py` | FastAPI app: `POST /score`, `GET /health` (k8s probe), `GET /info` (provenance), `POST /drift`, `POST /reload` |
| `seed_registry.py` | trains + registers a model and promotes it to `production` (the init step) |
| `drift_retrain.py` | drift check → retrain → register → promote (run by the CronJob) |
| `Dockerfile` | builds the server image from source |
| `k8s/` | `pvc` (shared registry), `seed-job`, `deployment` + `service`, `drift-retrain-cronjob` |

## How it fits together

```
seed_registry.py ─▶ ModelRegistry (shared volume) ──┐
                                                     ▼
   Deployment (N replicas) ─ ModelService.from_registry(alias="production")
        │  GET /health  → k8s liveness/readiness
        │  POST /score  → log-density        ── interact via the Service
        │  POST /drift  → drift report
        ▼
   CronJob drift_retrain.py ─ detect_drift ─▶ retrain ─▶ register ─▶ promote(alias)
                                                                      │
                                          rolling restart / POST /reload picks up the new model
```

## Run locally

```sh
pip install -e ".[ ]" fastapi "uvicorn[standard]"   # base pysp + serving extras
export PYSP_REGISTRY_ROOT=./models PYSP_MODEL_NAME=model
python deploy/seed_registry.py                       # populate the registry
export PYSP_REFERENCE_PATH=./models/model/reference.json
uvicorn deploy.app:app --port 8000
```

Interact:

```sh
curl localhost:8000/health
curl -X POST localhost:8000/score -H 'content-type: application/json' \
  -d '{"records": [1.0, 2.5, 3.1, 9.9]}'
curl -X POST localhost:8000/drift -H 'content-type: application/json' \
  -d '{"records": [6.0, 6.2, 5.8, 6.5]}'        # shifted batch -> drift: true
curl localhost:8000/info                          # provenance header
```

## On Kubernetes

```sh
docker build -f deploy/Dockerfile -t <registry>/pysp-model:latest .   # build from repo root
docker push <registry>/pysp-model:latest
# set the image in k8s/*.yaml, then:
kubectl apply -f deploy/k8s/pvc.yaml
kubectl apply -f deploy/k8s/seed-job.yaml          # one-shot: populate the registry
kubectl apply -f deploy/k8s/deployment.yaml -f deploy/k8s/service.yaml
kubectl apply -f deploy/k8s/drift-retrain-cronjob.yaml
```

Swap a model: `registry.promote(name, version)` (done by `drift_retrain.py`), then
`kubectl rollout restart deployment/pysp-model` (or `POST /reload` on each pod).

## Caveats (what to harden for real use)

- **Registry storage.** `ModelRegistry` is filesystem-backed; the manifests use a `ReadWriteMany` PVC. If
  your cluster has no RWX class, back it with object storage (S3/GCS via a CSI mount, or adapt the registry).
- **Logging.** Set `PYSP_ACTIVITY_LOG=/dev/stdout` so the per-request activity log (count, latency, mean
  log-lik, unscorable count) lands in container logs for aggregation; don't rely on the in-memory list
  across replicas.
- **Record shape.** `/score` JSON arrays map to model records (inner arrays → tuples for composite/record
  models). Match your model's expected fields.
- **Auth / rate limiting / TLS.** Add at the Ingress; none is included here.
- **Estimator in the CronJob.** `seed_registry.py` / `drift_retrain.py` use a Gaussian example — swap in
  your real model + estimator and wire `_recent_batch()` to your production data store.
