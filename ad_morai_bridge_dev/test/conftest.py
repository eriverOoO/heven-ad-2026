import os
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPOSITORY_ROOT.parents[1]

sys.path.insert(0, str(REPOSITORY_ROOT / "ad_morai_bridge_dev"))
sys.path.insert(0, str(REPOSITORY_ROOT / "ad_morai_bridge"))

generated_paths = []
ament_prefixes = [
    Path(value)
    for value in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep)
    if value
]
for package in ("ad_morai_interfaces_dev", "ad_morai_interfaces"):
    roots = [prefix / "local" / "lib" for prefix in ament_prefixes]
    roots.append(WORKSPACE_ROOT / "install" / package / "local" / "lib")
    for generated_root in roots:
        generated_paths.extend(generated_root.glob("python*/dist-packages"))

sys.path[:0] = [str(path) for path in generated_paths]
