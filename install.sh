#!/usr/bin/env bash
# MFlow one-step install into the current Python environment (conda env or venv).
#
#   bash install.sh                 # mflow + mace-torch 0.3.16 + phonopy, skills for pi, then `mflow doctor`
#   bash install.sh --all           # + cuequivariance-torch, nvidia-ml-py, pytest
#   bash install.sh --no-skills     # do not link the agent skills into ~/.agents/skills
#   bash install.sh --completion    # also add tab completion to ~/.bashrc
#
# torch is NOT installed here: it has to match your CUDA driver. Install it first, e.g.
#   pip install torch --index-url https://download.pytorch.org/whl/cu124
set -euo pipefail
cd "$(dirname "$0")"

extras="mace,validate"
skills=1
completion=0
for arg in "$@"; do
  case "$arg" in
    --all) extras="all" ;;
    --no-skills) skills=0 ;;
    --completion) completion=1 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

py=$(command -v python || command -v python3)
echo ">> python: $py ($("$py" --version 2>&1))"
if ! "$py" -c "import torch" 2>/dev/null; then
  echo ">> WARNING: torch is not installed in this environment."
  echo "   Install the build matching your CUDA first (https://pytorch.org), otherwise pip pulls a default one."
fi

echo ">> pip install -e \".[$extras]\""
"$py" -m pip install -e ".[$extras]"

if [ "$skills" = 1 ]; then
  echo ">> linking the agent skills"
  mflow skills install
fi
if [ "$completion" = 1 ] && ! grep -q "mflow completion" ~/.bashrc 2>/dev/null; then
  echo 'eval "$(mflow completion bash)"' >> ~/.bashrc
  echo ">> tab completion added to ~/.bashrc (open a new shell)"
fi

echo ">> mflow doctor"
mflow doctor || true
cat <<'MSG'

next:
  mflow config set model /path/to/your.model     # default model for calc / relax / check / validate
  mflow config set lmp /path/to/lmp              # your KOKKOS + ML-IAP LAMMPS build
  mflow                                          # list of commands
MSG
