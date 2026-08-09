from glob import glob
import os

from setuptools import find_packages, setup

package_name = "ad_tuning"

setup(
    name=package_name,
    version="0.3.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml", "README.md"]),
        (
            os.path.join("share", package_name, "launch"),
            glob("launch/*.launch.py"),
        ),
        (
            os.path.join("share", package_name, "config"),
            glob("config/*.yaml"),
        ),
    ],
    install_requires=[
        "setuptools",
        "optuna==4.9.*",
        "psycopg[binary]>=3.1,<4",
    ],
    zip_safe=True,
    maintainer="Taeyeong",
    maintainer_email="taeyeong@users.noreply.github.com",
    description="ROS 2 MORAI controller and local-planner tuning for HEVEN AD",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "global_path_tuner = ad_tuning.tuner_node:main",
            "dwa_tuner = ad_tuning.tuner_node:dwa_main",
        ],
    },
)
