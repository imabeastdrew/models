# Music Agent

An unfaithful reimplementation of ReaLchords paper.

## Setup

```bash
# Install deps
uv sync
```

## Training Example

```bash
musicagent-train-offline --epochs 20

musicagent-train-online --epochs 20
```

## CLI

```bash
musicagent-train-offline --help
musicagent-train-online --help
```

Without package:

```bash
python -m musicagent.training.offline --help
python -m musicagent.training.online --help
```

# TODO

- Refactor to a unified vocab across preprocessing, training, evaluation, and testing.

