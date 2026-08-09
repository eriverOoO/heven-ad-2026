from setuptools import find_packages, setup


package_name = "heven_control"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="HEVEN",
    maintainer_email="heven@example.com",
    description="Legacy low-speed route follower retained during migration.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "route_follower = heven_control.route_follower_node:main",
        ],
    },
)
