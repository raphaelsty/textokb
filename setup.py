import setuptools

from textokb.__version__ import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()

setuptools.setup(
    name="textokb",
    version=f"{__version__}",
    author="Raphael Sourty",
    author_email="raphael.sourty@gmail.com",
    description="Knowledge extraction from raw text.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/raphaelsty/textokb",
    packages=setuptools.find_packages(),
    install_requires=required,
    package_data={},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
