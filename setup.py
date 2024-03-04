from setuptools import setup, find_packages

setup(
    name="hyperdt",
    version="0.0.3",
    description="Fast Hyperboloid Decision Tree Algorithms",
    author="Philippe Chlenski",
    author_email="pac@cs.columbia.edu",
    url="http://www.github.com/pchlenski/hyperdt",
    packages=["hyperdt"],
    package_dir={"hyperdt": "hyperdt"},
    install_requires=["numpy", "geomstats", "matplotlib", "scikit-learn", "scipy", "tqdm", "joblib"],
)
