from setuptools import setup, find_packages

setup(
    name="sbc",
    version="30.0.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)