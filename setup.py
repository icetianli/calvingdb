from setuptools import setup, find_packages

setup(
    name="calvingdb",
    version="0.1.0",
    packages=find_packages(),
    py_modules=["losses", "metrics", "train", "evaluate", "utils"],
    python_requires=">=3.9",
)
