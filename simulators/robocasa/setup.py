from setuptools import find_packages, setup

package = "rosbridge_robocasa"

setup(
    name=package,
    version="0.2.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package}"]),
        (f"share/{package}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "rosbridge.backends": ["robocasa = rosbridge_robocasa.backend:RoboCasaSim"],
        "console_scripts": ["control_demo = rosbridge_robocasa.demos.control:main"],
    },
)
