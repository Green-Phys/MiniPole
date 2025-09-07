from setuptools import setup, find_packages

setup(
    name="mini_pole",
    version="0.5",
    description="This Python code implements the Minimal Pole Method (MPM) for both Matsubara and real-frequency data.",
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
