from glob import glob

from setuptools import find_packages, setup

setup(
    name="rosbridge_runtime", version="0.2.0", packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/rosbridge_runtime"]),
        ("share/rosbridge_runtime", ["package.xml"]),
        ("share/rosbridge_runtime/launch", glob("launch/*.py")),
    ],
    install_requires=["setuptools", "rosbridge"],
    entry_points={"console_scripts": ["sim_node = rosbridge_runtime.sim_node:main"]},
)
