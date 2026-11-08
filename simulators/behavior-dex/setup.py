from setuptools import find_packages, setup

package = "rosbridge_behavior_dex"

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
        "rosbridge.backends": ["behavior-dex = rosbridge_behavior_dex:BehaviorDexWorker"],
        "console_scripts": ["control_demo = rosbridge_behavior_dex.demos.control:main"],
    },
)
