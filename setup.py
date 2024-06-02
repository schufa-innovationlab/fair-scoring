import os
import sys

from setuptools import setup, find_packages



here = os.path.abspath(os.path.dirname(__file__))


def run_setup():

    try:
        readme = open(os.path.join(here, "README.md")).read()
    except IOError:
        readme = ""

    setup(
        name="fairscoring",
        author="SCHUFA Holding AG",
        description="Fairness metrics for continuous scores",
        long_description=readme,
        long_description_content_type="text/markdown",
        license="Apache 2.0",
        use_scm_version=True,
        packages=find_packages(where="src"),
        package_dir={"":"src"},
        include_package_data=True,
        url="https://github.com/schufa-innovationlab/fairscoring",
        install_requires=[
            # TODO: list dependencies, here (remove numpy, if not needed)
            "numpy>=1.11.2",
        ],
        python_requires=">=3.7",
        extras_require={
            "dev": [
                "sphinx>=6.2.1",              # Basic sphinx engine
                "sphinx-book-theme>=1.1.0",   # Theme for this template
                "sphinx-autoapi>=3.0.0",      # Automatically parse docstrings
                "myst-nb>=1.0.0",             # Allows to include jupyter & markdown files
                "pytest>=7.4",                # Testing Framework
            ]
        },
        classifiers=[
            "License :: OSI Approved :: Apache Software License",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
        ],
    )


if __name__ == "__main__":
    run_setup()
