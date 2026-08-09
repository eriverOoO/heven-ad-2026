from glob import glob
from setuptools import find_packages, setup


package_name = "heven_slam"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="HEVEN",
    maintainer_email="heven@example.com",
    description="Experimental global 3D LiDAR SLAM for the MORAI course.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "analyze_rtabmap_db = heven_slam.rtabmap_db_quality:main",
            "save_cloud_pcd = heven_slam.save_cloud_pcd_node:main",
        ],
    },
)
