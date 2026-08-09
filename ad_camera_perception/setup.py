from glob import glob

from setuptools import find_packages, setup


PACKAGE_NAME = "ad_camera_perception"


setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE_NAME}"]),
        (f"share/{PACKAGE_NAME}", ["package.xml"]),
        (f"share/{PACKAGE_NAME}/config", glob("config/*.yaml")),
        (f"share/{PACKAGE_NAME}/launch", glob("launch/*.launch.py")),
        (f"share/{PACKAGE_NAME}/models", glob("models/*")),
        (f"share/{PACKAGE_NAME}/docs", glob("docs/*.md")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Taeyeong",
    maintainer_email="taeyeong@users.noreply.github.com",
    description="Camera traffic-signal and object perception for the HEVEN AD stack",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "ad_traffic_signal_node = "
            "ad_camera_perception.nodes.traffic_signal_node:main",
            "ad_traffic_light_detector_node = "
            "ad_camera_perception.nodes.traffic_light_detector_node:main",
            "ad_traffic_light_evaluator_node = "
            "ad_camera_perception.nodes.traffic_light_evaluator_node:main",
            "ad_traffic_light_visualizer_node = "
            "ad_camera_perception.nodes.traffic_light_visualizer_node:main",
            "ad_dynamic_obstacle_detector_node = "
            "ad_camera_perception.nodes.dynamic_obstacle_detector_node:main",
            "ad_vision_visualizer_node = "
            "ad_camera_perception.nodes.vision_visualizer_node:main",
        ],
    },
)
