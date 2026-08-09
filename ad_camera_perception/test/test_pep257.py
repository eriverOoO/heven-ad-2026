"""Run pydocstyle through the ROS 2 ament test adapter."""

from ament_pep257.main import main
import pytest


@pytest.mark.linter
@pytest.mark.pep257
def test_pep257():
    """Check Python docstrings."""
    assert main(argv=["."]) == 0
