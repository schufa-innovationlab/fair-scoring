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
        name="fair-scoring",
        author="SCHUFA Holding AG",
        description="Fairness metrics for continuous risk scores",
        long_description=readme,
        long_description_content_type="text/markdown",
        license="Apache 2.0",
        use_scm_version=True,
        packages=find_packages(where="src"),
        package_dir={"":"src"},
        include_package_data=True,
        url="https://github.com/schufa-innovationlab/fair-scoring",
        project_urls={
            "Documentation": "https://fair-scoring.readthedocs.io/en/stable/",
            "Changelog": "https://fair-scoring.readthedocs.io/en/stable/changelog.html",
            "Source code": "https://github.com/schufa-innovationlab/fair-scoring",
        },
        install_requires=[
            "numpy>=1.22.0,<2.0",             # Not tested with numpy 2.0, yet
            "scikit-learn>=1.0.0",
            "matplotlib>=3.6.0",
            "seaborn>=0.12.0"
        ],
        python_requires=">=3.9",
        extras_require={
            "dev": [
                "sphinx>=6.2.1",              # Basic sphinx engine
                "sphinx-book-theme>=1.1.0",   # Theme for this template
                "sphinx-autoapi>=3.0.0",      # Automatically parse docstrings
                "sphinx-design>=0.6.0",       # Nice layouts
                "myst-nb>=1.0.0",             # Include jupyter & markdown files
                "pytest>=7.4",                # Testing Framework
                "pandas>=2.2.1",              # Pandas is required for tests
            ]
        },
        classifiers=[
            "License :: OSI Approved :: Apache Software License",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
        ],
    )


if __name__ == "__main__":
    run_setup()
