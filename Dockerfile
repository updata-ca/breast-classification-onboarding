FROM python:3.10
WORKDIR /usr/src/curated-breast-cancer-model
COPY pyproject.toml uv.lock model/train.py /usr/src/curated-breast-cancer-model/
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
RUN uv sync
RUN uv pip install "tensorflow<2.15.0"
ENV PATH="/usr/src/curated-breast-cancer-model/.venv/bin:$PATH"