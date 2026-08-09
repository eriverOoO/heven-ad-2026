from glob import glob
import os

from setuptools import find_packages, setup


package_name = "ad_vehicle_profiling"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob("launch/*.launch.py"),
        ),
        (
            os.path.join("share", package_name, "config"),
            glob("config/*.yaml"),
        ),
        (
            os.path.join("share", package_name, "schema"),
            glob("schema/*.json"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Taeyeong",
    maintainer_email="taeyeong@users.noreply.github.com",
    description="Resumable MORAI longitudinal vehicle response profiler",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "ad_vehicle_profiler = ad_vehicle_profiling.profiler_node:main",
            "ad_vehicle_profile_report = ad_vehicle_profiling.report:main",
            "ad_vehicle_profile_loop_guard = ad_vehicle_profiling.loop_guard_node:main",
        ],
    },
)
