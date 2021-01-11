from setuptools import setup, find_packages

setup(
    name="structshot",
    packages=find_packages(),
    install_requires=[
        "pytorch-lightning~=0.8.5",
        "transformers~=3.3.1",
        "seqeval",
    ],
)
