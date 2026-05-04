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
#                              Numpydoc config                                #
# --------------------------------------------------------------------------- #

# Let the class.rst template handle method tables; disable numpydoc's auto-
# generated tables to avoid stub-file warnings for inherited PyTorch members.
numpydoc_show_class_members = False

# --------------------------------------------------------------------------- #
#                               Autodoc config                                #
# --------------------------------------------------------------------------- #

autodoc_default_options = {}
autodoc_inherit_docstrings = False

# Suppress RST formatting warnings from third-party docstrings (torch.nn.Module)
suppress_warnings = ['docutils']

# --------------------------------------------------------------------------- #
#                          Autodoc skip-member hook                           #
# --------------------------------------------------------------------------- #

import torch.nn as _torch_nn
_TORCH_MODULE_ATTRS = frozenset(_torch_nn.Module.__dict__)


def _skip_torch_member(app, what, name, obj, skip, options):
    """Skip members that originate from torch.nn.Module, not from fynance."""
    if skip:
        return True
    if what == 'module':
        return skip
    if name not in _TORCH_MODULE_ATTRS:
        return skip
    module = getattr(obj, '__module__', '') or ''
    if not module.startswith('fynance'):
        return True
    return skip


def setup(app):
    app.connect('autodoc-skip-member', _skip_torch_member)
