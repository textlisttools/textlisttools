#!/bin/bash
# Installs the dependencies the clipify skill needs (Whisper + numpy) in cloud sessions.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

if ! python3 -c "import whisper, numpy" >/dev/null 2>&1; then
  # CPU-only torch wheels: much smaller than the default CUDA build, and cloud sessions have no GPU.
  pip install -q --root-user-action=ignore torch --index-url https://download.pytorch.org/whl/cpu
  pip install -q --root-user-action=ignore openai-whisper numpy
fi

# Pre-download the model clipify uses so the first transcription doesn't stall.
if [ ! -f "$HOME/.cache/whisper/tiny.en.pt" ]; then
  python3 -c "import whisper; whisper.load_model('tiny.en')" >/dev/null 2>&1
fi
