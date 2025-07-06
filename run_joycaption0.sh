#!/usr/bin/env bash
# ─ JoyCaption one-click launcher (POSIX) ───────────────────────────
# Usage: ./run_joycaption.sh           # normal start
#        ./run_joycaption.sh --update  # refresh env from YAML first

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0    # set to 0 in the other script
ENV_NAME="joycaption"
UPDATE_ENV=0
PORT=7863

# parse optional --update flag
for arg in "$@"; do
  [[ "$arg" == "--update" ]] && UPDATE_ENV=1
done

# Detect Conda (falls back to ~/.miniconda3 if conda not on PATH yet)
if ! command -v conda &>/dev/null; then
  CONDA_ROOT="$HOME/miniconda3"
  if [[ ! -x "$CONDA_ROOT/bin/conda" ]]; then
    echo "⏬  Installing Miniconda into $CONDA_ROOT …"
    curl -L -o /tmp/miniconda.sh \
      https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_ROOT"
    rm /tmp/miniconda.sh
  fi
  export PATH="$CONDA_ROOT/bin:$PATH"
fi


# Create / update env only when needed
if [[ -x "$(conda info --base)/envs/$ENV_NAME/bin/python" ]]; then
  echo "✓ Conda env $ENV_NAME already exists"
  if [[ $UPDATE_ENV -eq 1 ]]; then
    echo "⟳ Updating env from environment.yml …"
    conda env update -n "$ENV_NAME" -f "$(dirname "$0")/environment.yml"
  fi
else
  echo "🆕 Creating env $ENV_NAME …"
  conda env create -n "$ENV_NAME" -f "$(dirname "$0")/environment.yml"
fi

# shellcheck disable=SC1090
#source "$(conda info --base)/etc/profile.d/conda.sh"
# Activate Conda (assumes Miniconda in $HOME/miniconda3; edit if different)
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# suppress Windows-specific HF symlink warning on Unix just in case
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

# start Gradio app in background and open browser
#!/usr/bin/env bash

# everything below is the same as run_joycaption.sh  ───────────────▼
for arg in "$@"; do [[ "$arg" == "--update" ]] && UPDATE_ENV=1; done

# (…Conda detection, env create/update, activation…)

python "$(dirname "$0")/app.py" --port "$PORT" &
PID=$!
sleep 3
xdg-open "http://localhost:$PORT" >/dev/null 2>&1 || true

echo
echo "JoyCaption running on GPU $CUDA_VISIBLE_DEVICES  →  http://localhost:$PORT"
echo "Ctrl-C to stop."
trap 'kill $PID 2>/dev/null' INT
wait $PID