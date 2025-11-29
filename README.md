# Music Agent

An unfaithful reimplementation of ReaLchords paper.

## Setup

```bash
# Install deps
uv sync
```

## Training Example

```bash
# Offline (encoder–decoder, full‑melody context)
musicagent-train-offline --epochs 20

# Online (decoder‑only, interleaved melody/chord frames)
musicagent-train-online --epochs 20
```

## CLI

After installation, the following CLIs are available:

```bash
musicagent-train-offline --help
musicagent-train-online --help
```

For running modules directly without installing as a package:

```bash
python -m musicagent.training.offline --help
python -m musicagent.training.online --help
```

