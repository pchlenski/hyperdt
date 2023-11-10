from setuptools import setup, find_packages

setup(
    name="hyperdt",
    version="0.1",
    packages=find_packages(where="src/hyperdt"),
    package_dir={"": "src"},
)
