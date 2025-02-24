from setuptools import setup, find_packages

setup(
    name="safecontext",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "spacy>=3.8.0",
        "pytest>=8.0.0",
    ],
    python_requires=">=3.8",
)
