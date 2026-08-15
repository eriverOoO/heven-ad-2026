#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/.." && pwd)"
cd "$repo_dir"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

heven_python="${HEVEN_PYTHON:-/usr/bin/python3}"
if [[ ! -x "$heven_python" ]]; then
  echo "Python interpreter is not executable: $heven_python" >&2
  exit 1
fi

python_version="$("$heven_python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$python_version" != "3.10" ]]; then
  echo "ROS 2 Humble development requires Python 3.10; got $python_version." >&2
  exit 1
fi

if [[ -e .venv && ! -x .venv/bin/python ]]; then
  echo ".venv exists but is not a usable Python environment." >&2
  exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
  uv venv \
    --python "$heven_python" \
    --no-python-downloads \
    .venv
fi

if ! grep -Fxq "include-system-site-packages = false" .venv/pyvenv.cfg; then
  echo ".venv is not isolated from user-site packages." >&2
  echo "Recreate it with: uv venv --clear --python $heven_python .venv" >&2
  exit 1
fi

uv sync --locked --python "$heven_python" --no-python-downloads

venv_site_dir="$(
  .venv/bin/python -c 'import site; print(site.getsitepackages()[0])'
)"
printf '%s\n' "/usr/lib/python3/dist-packages" \
  > "$venv_site_dir/heven_ubuntu_dist_packages.pth"

# Ubuntu's Matplotlib 3.5 uses a regular mpl_toolkits package. Without this
# local marker it wins over the locked Matplotlib namespace package.
touch "$venv_site_dir/mpl_toolkits/__init__.py"

.venv/bin/python - <<'PY'
from pathlib import Path
import site
import sys

import matplotlib
import numpy
import optuna
import pytest

environment = Path(sys.prefix).resolve()
if site.ENABLE_USER_SITE:
    raise SystemExit("user-site packages are enabled inside .venv")
for module in (pytest, numpy, matplotlib, optuna):
    module_path = Path(module.__file__).resolve()
    if not module_path.is_relative_to(environment):
        raise SystemExit(
            f"{module.__name__} was loaded outside .venv: {module_path}"
        )
PY
