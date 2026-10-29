from setuptools import find_packages, setup

package = "rosbridge_behavior_g1"

setup(
    name=package,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package}"]),
        (f"share/{package}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "rosbridge.backends": ["behavior-g1 = rosbridge_behavior_g1:BehaviorG1Worker"],
        "console_scripts": ["control_demo = rosbridge_behavior_g1.demos.control:main"],
    },
)
