#!/usr/bin/env bash
# Keep Isaac Gym training (Python 3.8) separate from MuJoCo (Python 3.11).
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-all}"

usage() {
    cat <<'EOF'
Usage: bash scripts/setup_envs.sh [all|train|mujoco]

  all      Set up both environments (default).
  train    Python 3.8 in .venv for Isaac Gym training, play and export.
  mujoco   Python 3.11 in .venv-mujoco for MuJoCo simulation.

Install uv first. Existing environments are reused only when their Python
version matches; this script never deletes an environment.

ISAAC_GYM_PATH may point to the Isaac Gym SDK's python directory.
It defaults to $HOME/isaacgym/python. The SDK must already be extracted.
uv uses its normal cache; UV_CACHE_DIR can override the cache location.
EOF
}

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

if (( $# > 1 )); then
    usage >&2
    exit 2
fi
case "$mode" in
    -h|--help) usage; exit 0 ;;
    all|train|mujoco) ;;
    *) usage >&2; exit 2 ;;
esac

command -v uv >/dev/null 2>&1 || fail "uv is not on PATH."
[[ "$(uname -s)" == Linux ]] || fail "These environments target Linux."

# Check all requested inputs before creating or changing either environment.
if [[ "$mode" == all || "$mode" == train ]]; then
    isaac_gym_path="${ISAAC_GYM_PATH:-$HOME/isaacgym/python}"
    [[ -f "$isaac_gym_path/setup.py" ]] ||
        fail "Missing Isaac Gym setup.py in $isaac_gym_path; set ISAAC_GYM_PATH."
    [[ -f "$isaac_gym_path/isaacgym/_bindings/linux-x86_64/gym_38.so" ]] ||
        fail "Missing Isaac Gym Python 3.8 Linux bindings in $isaac_gym_path."
    [[ -f "$repo_root/requirements/train.txt" ]] || fail "Missing requirements/train.txt."
fi
if [[ "$mode" == all || "$mode" == mujoco ]]; then
    [[ -f "$repo_root/requirements/mujoco.txt" ]] || fail "Missing requirements/mujoco.txt."
fi

ensure_venv() {
    local env_path="$1" python_version="$2" actual_version
    if [[ -e "$env_path" || -L "$env_path" ]]; then
        [[ -f "$env_path/pyvenv.cfg" && -x "$env_path/bin/python" ]] ||
            fail "$env_path exists but is not a usable virtual environment."
        actual_version="$("$env_path/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
        [[ "$actual_version" == "$python_version" ]] ||
            fail "$env_path uses Python $actual_version; expected $python_version. Move it aside before retrying."
    else
        uv venv --python "$python_version" "$env_path"
    fi
}

if [[ "$mode" == all || "$mode" == train ]]; then
    ensure_venv "$repo_root/.venv" 3.8
    train_python="$repo_root/.venv/bin/python"
    uv pip install --python "$train_python" -r "$repo_root/requirements/train.txt"
    # Build tools are pinned in train.txt. Local editable packages must not
    # resolve a different PyTorch/NumPy stack or download an unrelated SDK.
    uv pip install --python "$train_python" --no-deps --no-build-isolation \
        -e "$isaac_gym_path" -e "$repo_root/rsl_rl" -e "$repo_root"
    uv pip check --python "$train_python"
    printf 'Training/play environment ready: %s\n' "$train_python"
fi

if [[ "$mode" == all || "$mode" == mujoco ]]; then
    ensure_venv "$repo_root/.venv-mujoco" 3.11
    mujoco_python="$repo_root/.venv-mujoco/bin/python"
    uv pip install --python "$mujoco_python" --torch-backend cpu \
        -r "$repo_root/requirements/mujoco.txt"
    # Simulation entry points already add repository sources to sys.path.
    # Installing legged_gym here would also pull in the Python 3.8-only SDK.
    uv pip check --python "$mujoco_python"
    printf 'MuJoCo environment ready: %s\n' "$mujoco_python"
fi
