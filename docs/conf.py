# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# Need to import as get_version since version is a reserved variable name in
# sphinx.
from importlib.metadata import version as get_version
import sphinx_rtd_theme

# -- Project information -----------------------------------------------------

project = 'psiz'
copyright = '2021, The PsiZ Authors'
author = 'Brett D. Roads'

# The full version, including alpha/beta/rc tags
release = get_version("psiz")
ver = '.'.join(release.split('.')[:3])  # TODO why ver? is it necessary?
version = ver

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.napoleon',
    'sphinx.ext.imgmath',  # Render math equations.
    'sphinx_last_updated_by_git',  # Infer last updated date via git.
    'nbsphinx',  # Jupyter notebooks.
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_favicon = 'img/favicon.ico'
html_logo = 'img/full_logo_300.png'
html_theme_options = {
    'logo_only': True,
}
