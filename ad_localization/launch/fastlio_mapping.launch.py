"""Start the adapter and the FastLIO mapping mode only."""

import importlib.util
from pathlib import Path

from ament_index_python.packages import get_package_share_directory


# launch/ is not an installed Python package, so load the shared module by path.
_spec = importlib.util.spec_from_file_location(
    "ad_localization_fastlio_launch",
    Path(__file__).with_name("fastlio_launch.py"),
)
_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_common)


def generate_launch_description():
    return _common.generate_fastlio_launch_description(
        "mapping", get_package_share_directory
    )
