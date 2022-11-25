from setuptools import setup, find_packages

def _requires_from_file(filename):
    return open(filename).read().splitlines()


setup(
    name="isc-feature-extractor",
    version="1.0.0",
    url="https://github.com/lyakaap/ISC21-Descriptor-Track-1st",
    author="lyakaap",
    packages=find_packages(exclude=["exp", "scripts"]),
    python_requires=">=3.7",
    install_requires=_requires_from_file("requirements.txt"),
)
