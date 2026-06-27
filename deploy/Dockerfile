# pysparkplug model server image.
#
# Build from the REPO ROOT so the package is installed from source (works whether or not pysp-learn is
# published to PyPI yet):
#     docker build -f deploy/Dockerfile -t pysp-model:latest .
#
# The image bundles the base install (numpy/scipy/pandas/mpmath) + the FastAPI serving stack. It does NOT
# bundle a model -- the model is loaded at runtime from the registry volume (seed it with seed_registry.py).

FROM python:3.12-slim

WORKDIR /app

# Install the package from source, then the serving extras.
COPY pyproject.toml README.md ./
COPY pysp ./pysp
RUN pip install --no-cache-dir . \
    && pip install --no-cache-dir "fastapi>=0.110" "uvicorn[standard]>=0.29" "pydantic>=2"

# The serving app + the seed/retrain helpers.
COPY deploy/app.py deploy/seed_registry.py deploy/drift_retrain.py ./

ENV PYSP_REGISTRY_ROOT=/models \
    PYSP_MODEL_NAME=model \
    PYSP_MODEL_ALIAS=production \
    PYSP_ACTIVITY_LOG=/dev/stdout

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
