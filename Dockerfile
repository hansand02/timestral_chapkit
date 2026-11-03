# chapkit_timestral_tsfm Dockerfile
FROM ghcr.io/dhis2-chap/chapkit-py:latest

# Build steps run as root; the service itself runs as the unprivileged `chapkit`
# user (the switch happens right before CMD). Same posture as chap-core's own
# images. Newer chapkit base images ship the user already (uid/gid 1000); older
# ones do not, so create it only when it is missing.
USER root
RUN id -u chapkit >/dev/null 2>&1 \
    || (groupadd --gid 1000 chapkit && useradd --uid 1000 --gid 1000 --no-create-home --shell /usr/sbin/nologin chapkit)

# timestral is installed from GitHub, so uv needs git.
RUN command -v git >/dev/null || (apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*)

WORKDIR /work
# Copy lockfile + manifest first so the dep-install layer caches independently of code changes.
# Run `uv sync` once locally before the first `docker build` to generate uv.lock.
COPY pyproject.toml uv.lock ./

# Sync user deps into the venv at /app/.venv. --frozen pins to uv.lock for reproducible
# builds, --no-dev skips the dev-only dependency group, --no-install-project because this
# project isn't a package, just a service. The uvicorn used by the CMD below comes from
# this sync, not from the base image: chapkit depends on servicekit, which depends on
# fastapi[standard], which pulls in uvicorn.
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_PROJECT_ENVIRONMENT=/app/.venv uv sync --frozen --no-dev --no-install-project

# Git commit the image was built from, exposed as git_revision on /api/v1/info. Same
# convention as chap-core: pass --build-arg GIT_REVISION=$(git rev-parse HEAD) (the
# scaffolded publish workflow and compose.yml already do). Declared after the uv sync
# layer so a new commit does not invalidate the dependency cache.
# Bake the Timestral weights (~840 MB) into the image: at runtime HOME is a tmpfs and the
# root filesystem is read-only, so a runtime download would land in RAM on every start.
ENV HF_HOME=/opt/hf
RUN /app/.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('mistralai/Timestral-209M')" \
    && chmod -R a+rX /opt/hf
ENV HF_HUB_OFFLINE=1

ARG GIT_REVISION=""
ENV GIT_REVISION=${GIT_REVISION}

COPY main.py ./

# The service writes in exactly two places: /work/data (SQLite database, a named
# volume in compose.yml) and /tmp (ML workspaces, a tmpfs in compose.yml). Everything
# else stays read-only, which is what compose.yml's `read_only: true` enforces. HOME,
# MPLCONFIGDIR and XDG_CACHE_HOME point libraries that want a scratch directory
# (R session files, matplotlib, pip/uv caches) at /tmp as well.
RUN mkdir -p /work/data && chown -R chapkit:chapkit /work/data
ENV HOME=/tmp \
    MPLCONFIGDIR=/tmp \
    XDG_CACHE_HOME=/tmp/.cache
USER chapkit

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

# Run uvicorn directly so server flags (--workers, --log-level, --timeout-keep-alive,
# etc.) are visible here in the Dockerfile. main.py's `if __name__ == "__main__":`
# block stays as the local-dev shortcut for `python main.py`.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]