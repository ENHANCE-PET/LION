# Pin the multi-platform Python image so rebuilding a release uses the same base.
FROM python:3.10.20-slim-trixie@sha256:63669fd2563fa90b0442fa7b568e66e3667755636cda086d7bcaaa895f66fe39

ARG LIONZ_VERSION=1.0.5
ARG LIONZ_REVISION=unknown

ENV UV_PROJECT_ENVIRONMENT=/opt/lionz \
    UV_LINK_MODE=copy \
    LIONZ_MODELS_DIRECTORY=/usr/local/models/nnunet_trained_models \
    PATH="/opt/lionz/bin:${PATH}"

# Install LION's system runtime plus a pinned lockfile installer.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && python -m pip install --no-cache-dir "uv==0.9.22" \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the locked dependency graph in its own cacheable layer.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN test "$(uv version --short)" = "${LIONZ_VERSION}" \
    && uv sync --locked --no-dev --no-install-project --no-cache

# Install the exact LION source from the release tag without re-resolving.
COPY lionz ./lionz
RUN uv sync --locked --no-dev --no-editable --no-cache

# Fail the image build if dependency resolution drifts from the supported GPU ABI.
COPY docker/verify_runtime.py /tmp/verify_runtime.py
RUN python /tmp/verify_runtime.py && rm /tmp/verify_runtime.py

LABEL org.opencontainers.image.version="${LIONZ_VERSION}" \
    org.opencontainers.image.source="https://github.com/ENHANCE-PET/LION" \
    org.opencontainers.image.revision="${LIONZ_REVISION}"

ENTRYPOINT ["lionz"]
CMD ["-h"]
