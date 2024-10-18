from setuptools import find_packages, setup

setup(
    name="gpuselect",
    version="0.1",
    packages=find_packages(exclude=("tests",)),
    install_requires=[
        "nvidia-ml-py"
    ],
    entry_points={
        "console_scripts": [
            "gpuselect=gpuselect.nvmlgpuselect:main"
        ],
    },
)
