from setuptools import setup, find_packages

setup(
    name="hyperdt",
    version="0.0.1",
    packages=find_packages(where="hyperdt"),
    package_dir={"": "hyperdt"},
    requires=["numpy", "geomstats", "matplotlib", "scikit-learn", "scipy", "tqdm", "joblib"],
)
