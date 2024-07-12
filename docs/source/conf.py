# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from importlib.metadata import version as get_version

sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Fair-Scoring'
copyright = '2024, SCHUFA Holding AG'
author = 'SCHUFA Holding AG'

# Suggested by https://setuptools-scm.readthedocs.io/en/latest/usage/#usage-from-sphinx
release = get_version('fair-scoring')
version = ".".join(release.split('.')[:2])     # for example take major/minor

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',             # Include documentation from docstrings
    'sphinx.ext.mathjax',             # Render Math in html outputs
    'sphinx.ext.napoleon',            # Support Numpy-Styled docstrings
    'autoapi.extension',              # Automatically creating .rst files from docstrings
    'sphinx_design',                  # Responsive Websites
    'myst_nb',                        # Support Notebooks and Markdown
]

templates_path = ['_templates']
exclude_patterns = []

napoleon_google_docstring = False   # Turn off googledoc strings
napoleon_numpy_docstring = True     # Turn on numpydoc strings
napoleon_use_ivar = True 	        # For maths symbology

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
# html_static_path = ['_static']
html_theme_options = {
    "show_toc_level": 3
}


# -- Options for autoapi -----------------------------------------------------
# see https://sphinx-autoapi.readthedocs.io/en/latest/reference/config.html
autoapi_dirs = [
    '../../src/fairscoring',
]
autoapi_root = "autoapi"
autoapi_add_toctree_entry = False  # Automatically add
autoapi_options = [
    'members',                     # Display children of an object
    'inherited-members',           # Display children of an object that have been inherited from a base class.
    # 'undoc-members',             # Display objects that have no docstring
    # 'private-members',           # Display private objects (eg. _foo in Python)
    'special-members',             # Display special objects (eg. __foo__ in Python)
    'show-inheritance',            # Display a list of base classes below the class signature.
    # 'show-inheritance-diagram',  # Display an inheritance diagram in generated class documentation.'
    'show-module-summary',         # Whether to include autosummary directives in generated module documentation.
    # 'imported-members',            # Display objects imported from the same top level package or module.
]

# Turn on for debugging only
autoapi_keep_files = False

# To support implicit namespaces
autoapi_python_use_implicit_namespaces = False

autoapi_template_dir = "_templates/autoapi"

autodoc_typehints = 'description'


# Fix for namespace packages
def top_level(pages):
    page_names = {page.name for page in pages}
    for page in pages:
        if page.name.rpartition(".")[0] not in page_names:
            yield page


def prepare_jinja_env(jinja_env) -> None:
    jinja_env.filters["top_level"] = top_level

autoapi_prepare_jinja_env = prepare_jinja_env


# -- Myst configuration ------------------------------------------------------
# Attention: you need to run your notebooks on your own
nb_execution_mode = "off"

# See https://myst-parser.readthedocs.io/en/latest/syntax/optional.html
myst_enable_extensions  = [
    "amsmath",
    # "attrs_inline",
    "colon_fence",
    # "deflist",
    "dollarmath",
    # "fieldlist",
    # "html_admonition",
    # "html_image",
    # "linkify",
    # "replacements",
    # "smartquotes",
    # "strikethrough",
    # "substitution",
    # "tasklist",
]

myst_dmath_double_inline = True
