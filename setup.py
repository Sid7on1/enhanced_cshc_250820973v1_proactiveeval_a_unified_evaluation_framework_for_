import os
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from typing import List, Dict

# Constants
PROJECT_NAME = "ProactiveEval"
VERSION = "1.0.0"
AUTHOR = "Tianjian Liu, Fanqi Wan, Jiajian Guo, Xiaojun Quan"
EMAIL = "liutj9@mail2.sysu.edu.cn, wanfq@mail2.sysu.edu.cn, guojj59@mail2.sysu.edu.cn, quanxj3@mail.sysu.edu.cn"
DESCRIPTION = "A unified framework for evaluating proactive dialogue capabilities of LLMs"
LONG_DESCRIPTION = """
ProactiveEval is a unified framework designed for evaluating proactive dialogue capabilities of LLMs.
This framework decomposes proactive dialogue into target planning and dialogue guidance, establishing evaluation metrics across various scenarios.
"""
URL = "https://github.com/ProactiveEval/ProactiveEval"
LICENSE = "MIT"
CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
]
KEYWORDS = "proactive dialogue, LLMs, evaluation framework"
PYTHON_VERSION = ">=3.8"
DEPENDENCIES = [
    "torch",
    "numpy",
    "pandas",
]

# Setup configuration
def read_file(filename: str) -> str:
    """Read the contents of a file."""
    with open(filename, "r", encoding="utf-8") as file:
        return file.read()

def get_version() -> str:
    """Get the version of the package."""
    return VERSION

class CustomInstallCommand(install):
    """Custom install command to handle additional installation tasks."""
    def run(self) -> None:
        install.run(self)
        print("Installation complete.")

class CustomDevelopCommand(develop):
    """Custom develop command to handle additional development tasks."""
    def run(self) -> None:
        develop.run(self)
        print("Development environment set up.")

class CustomEggInfoCommand(egg_info):
    """Custom egg info command to handle additional egg info tasks."""
    def run(self) -> None:
        egg_info.run(self)
        print("Egg info generated.")

def main() -> None:
    """Main function to set up the package."""
    setup(
        name=PROJECT_NAME,
        version=get_version(),
        author=AUTHOR,
        author_email=EMAIL,
        description=DESCRIPTION,
        long_description=read_file("README.md"),
        long_description_content_type="text/markdown",
        url=URL,
        license=LICENSE,
        classifiers=CLASSIFIERS,
        keywords=KEYWORDS,
        python_requires=PYTHON_VERSION,
        packages=find_packages(),
        install_requires=DEPENDENCIES,
        cmdclass={
            "install": CustomInstallCommand,
            "develop": CustomDevelopCommand,
            "egg_info": CustomEggInfoCommand,
        },
    )

if __name__ == "__main__":
    main()