#!/usr/bin/env python3
# coding: utf-8

""" Configuration file of documentation. """

# Built-in
import os
import re
import sys
from datetime import date

needs_sphinx = '7.0'

# --------------------------------------------------------------------------- #
#                           General configuration                             #
# --------------------------------------------------------------------------- #

sys.path.append(os.path.abspath('../..'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'numpydoc',
    'matplotlib.sphinxext.plot_directive',
]

import fynance
project = 'Fynance'
copyright = '2018-{}, Arthur Bernard'.format(date.today().year)
author = 'Arthur Bernard'
version = re.sub(r'\.dev-.*$', r'.dev', fynance.__version__)
release = fynance.__version__

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
pygments_style = 'sphinx'

add_function_parentheses = False
add_module_names = True

# --------------------------------------------------------------------------- #
#                                HTML config                                  #
# --------------------------------------------------------------------------- #

html_theme = 'furo'
html_theme_options = {
    "source_repository": "https://github.com/ArthurBernard/Fynance/",
    "source_branch": "master",
    "source_directory": "doc/source/",
}
html_title = '{} v{} Reference Guide'.format(project, version)
html_static_path = ['_static']

html_context = {
    "display_github": True,
    "github_user": "ArthurBernard",
    "github_repo": "Fynance",
    "github_version": "master",
    "conf_py_path": "/source/",
}

html_domain_indices = True
html_copy_source = False
html_file_suffix = '.html'

# --------------------------------------------------------------------------- #
#                             Intersphinx config                              #
# --------------------------------------------------------------------------- #

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}

# --------------------------------------------------------------------------- #
#                             Autosummary config                              #
# --------------------------------------------------------------------------- #

autosummary_generate = True

# --------------------------------------------------------------------------- #
#                               Autodoc config                                #
# --------------------------------------------------------------------------- #

autodoc_default_options = {}
autodoc_inherit_docstrings = False

# Suppress RST formatting warnings from third-party docstrings (torch.nn.Module)
suppress_warnings = ['docutils']
