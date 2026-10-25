#!/usr/bin/env bash
set -euo pipefail
project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project"
export XDG_CACHE_HOME="$project/.cache/rosbridge/xdg"
export UV_CACHE_DIR="$project/.cache/rosbridge/uv"
export UV_PYTHON_INSTALL_DIR="$project/.runtime/python"
export TMPDIR="$project/.cache/rosbridge/tmp"
mkdir -p "$TMPDIR"
if ! command -v uv >/dev/null 2>&1; then
  echo 'uv is required to bootstrap the rosbridge CLI.' >&2
  exit 1
fi
if [[ $# -gt 1 ]]; then
  echo 'Usage: bash tools/bootstrap.sh [VENV_DIRECTORY]' >&2
  exit 2
fi
environment_dir=${1:-.venv}
if [[ -e "$environment_dir" && ! -f "$environment_dir/pyvenv.cfg" ]]; then
  echo 'Refusing an existing target that is not a Python virtual environment.' >&2
  exit 1
fi
if [[ ! -f "$environment_dir/pyvenv.cfg" ]]; then
  uv venv --python 3.12 --python-preference only-managed "$environment_dir"
fi
"$environment_dir/bin/python" -I -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 required"'
uv pip install --python "$environment_dir/bin/python" -e .
echo "Activate: source $environment_dir/bin/activate"
echo 'Prepare a simulator: rosbridge prepare abc'
