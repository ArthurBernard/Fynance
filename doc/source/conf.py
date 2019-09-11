#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-08-21 07:05:30
# @Last modified by: ArthurBernard
# @Last modified time: 2019-09-11 10:10:30

""" Configuration file of documentation. """

# Built-in packages
import os
import sys
from unittest.mock import MagicMock
# import glob

# Third party packages
# from sphinx.ext.autosummary import _import_by_name
# from numpydoc.docscrape import NumpyDocString
# from numpydoc.docscrape_sphinx import SphinxDocString
# import numpydoc.docscrape as np_docscrape
import sphinx

# Check Sphinx version
if sphinx.__version__ < "1.6":
    raise RuntimeError("Sphinx 1.6 or newer required")

needs_sphinx = '1.6'


class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()


# --------------------------------------------------------------------------- #
#                           General configuration                             #
# --------------------------------------------------------------------------- #

# np_docscrape.ClassDoc.extra_public_methods = [  # should match class.rst
#    '__call__', '__mul__', '__getitem__', '__len__',
# ]

# autosummary_generate = glob.glob("reference/*.rst")

# sys.path.insert(0, os.path.abspath('../..'))
sys.path.append(os.path.abspath('../..'))
sys.path.append(os.path.abspath('../sphinxext'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'numpydoc',
    'matplotlib.sphinxext.plot_directive',
]

project = 'fynance'
copyright = '2018-2019, Arthur Bernard'
author = 'Arthur Bernard'

version = "1.0.5"
release = "1.0.5"

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
pygments_style = 'sphinx'  # Style of code source

add_function_parentheses = False  # Parentheses are appended to function
add_module_names = True  # Module names are prepended to all object name

themedir = os.path.join(os.pardir, 'scipy-sphinx-theme', '_theme')
html_theme = 'scipy'
html_theme_path = [themedir]

# USELESS ?
# numpydoc_show_class_members = True
# class_members_toctree = False
# nitpicky = True
# numpydoc_attributes_as_param_list = False

# html_theme = 'scipy-sphinx-theme'  # 'sphinx_rtd_theme'  # Theme of docs
# html_theme_path = ["./_theme/scipy/"]
html_theme_option = {
    'edit_links': True,
    'sidebar': 'left',
    'scipy_org_logo': False,
    'navigation_links': True,
    'rootlinks': [
        (
            'https://github.com/ArthurBernard/Fynance',
            'Fynance'
        ),
        (
            'https://fynance.readthedocs.io',
            'Docs'
        ),
    ]
}
html_sidebars = {'index': ['searchbox.html', 'indexsidebar.html']}
html_static_path = ['_static']
html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "ArthurBernard",  # Username
    "github_repo": "Fynance",    # Repo name
    "github_version": "master",  # Version
    "conf_py_path": "/source/",  # Path in the checkout to the docs root
}

intersphinx_mapping = {
    'python': (
        'https://docs.python.org/dev',
        None
    ),
    'dccd': (
        'https://download-crypto-currencies-data.readthedocs.io/en/latest/',
        None
    ),
}


autosummary_generate = True
