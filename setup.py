from setuptools import setup, find_packages

setup(
    name="mini_pole",
    version="0.3",
    description="The Python code provided implements the matrix-valued version of the Minimal Pole Method (MPM) as described in Phys. Rev. B 110, 235131 (2024).",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Lei Zhang",
    author_email="lzphy@umich.edu",
    url="https://github.com/Green-Phys/MiniPole",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
