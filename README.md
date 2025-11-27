# Music Agent

An unfaithful reimplementation of ReaLchords paper.

## Setup

```bash
# Install deps
uv sync
```

## Training EX

```bash
python -m musicagent.train --epochs 20

# Custom wandb project/run name
python -m musicagent.train --wandb-project my-project --run-name exp-1
```

### CLI

The training module exposes a CLI; run:

```bash
python -m musicagent.train --help
```

to see flags.