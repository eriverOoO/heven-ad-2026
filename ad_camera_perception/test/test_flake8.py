"""Run flake8 through the ROS 2 ament test adapter."""

from ament_flake8.main import main_with_errors
import pytest


@pytest.mark.flake8
@pytest.mark.linter
def test_flake8():
    """Check Python source formatting."""
    return_code, errors = main_with_errors(argv=[])
    assert return_code == 0, "\n".join(errors)
