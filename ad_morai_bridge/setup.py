from glob import glob
import os

from setuptools import find_packages, setup


package_name = "ad_morai_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob("launch/*.launch.py"),
        ),
        (
            os.path.join("share", package_name, "config"),
            glob("config/*.yaml"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Taeyeong",
    maintainer_email="taeyeong@users.noreply.github.com",
    description="Competition-safe MORAI SIM 24.R2 UDP to ROS 2 bridge",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "ad_morai_bridge_node = ad_morai_bridge.morai_bridge_node:main",
            "ad_velodyne_adapter = ad_morai_bridge.velodyne_adapter_node:main",
            "ad_point_time_zero_boundary = "
            "ad_morai_bridge.point_time_zero_node:main",
            "ad_measurement_compatibility = "
            "ad_morai_bridge.measurement_compatibility:main",
        ],
    },
)
