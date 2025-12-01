# Music Agent

An unfaithful reimplementation of ReaLchords paper.

## Setup

```bash
# Install deps
uv sync
```

## Training Example

```bash
musicagent-train --mode offline --epochs 100

musicagent-train --mode online --epochs 100
```

## CLI

```bash
musicagent-preprocess --help
musicagent-train --help
musicagent-eval --help
```

Without package:

```bash
python -m musicagent.scripts.preprocess --help
python -m musicagent.training --help
python -m musicagent.eval --help
```

## End-to-end workflow

```bash
musicagent-preprocess --input sheetsage-data/hooktheory/Hooktheory.json --output realchords_data
musicagent-train --mode offline --epochs 100
musicagent-eval --mode offline --checkpoint checkpoints/offline/best_model.pt
```

