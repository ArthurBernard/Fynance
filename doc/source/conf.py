#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-08-21 07:05:30
# @Last modified by: ArthurBernard
# @Last modified time: 2023-07-28 12:41:36

""" Configuration file of documentation. """

# Built-in packages
import os
import sys
import re
from unittest.mock import MagicMock
from datetime import date
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

project = 'Fynance'
copyright = '2018-{}, Arthur Bernard'.format(date.today().year)
author = 'Arthur Bernard'

# The default replacements for |version| and |release|, also used in various
# other places throughout the built documents.
import fynance
version = re.sub(r'\.dev-.*$', r'.dev', fynance.__version__)
release = fynance.__version__

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
pygments_style = 'sphinx'  # Style of code source

add_function_parentheses = False  # Parentheses are appended to function
add_module_names = True  # Module names are prepended to all object name

# --------------------------------------------------------------------------- #
#                                HTML config                                  #
# --------------------------------------------------------------------------- #

themedir = os.path.join(os.pardir, 'scipy-sphinx-theme', '_theme')
html_theme = 'scipy'
html_theme_path = [themedir]

html_theme_options = {
    'edit_link': True,
    'sidebar': 'left',
    'scipy_org_logo': False,
    'navigation_links': True,
    'rootlinks': [("https://github.com/ArthurBernard/Fynance/", "Fynance"),
                  ("https://fynance.readthedocs.io/", "Docs")]
}
html_sidebars = {'index': ['searchbox.html', 'indexsidebar.html']}
html_title = '{} v{} Reference Guide'.format(project, version)
html_static_path = ['_static']

html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "ArthurBernard",  # Username
    "github_repo": "Fynance",    # Repo name
    "github_version": "master",  # Version
    "conf_py_path": "/source/",  # Path in the checkout to the docs root
}

html_domain_indices = True
html_copy_source = False
html_file_suffix = '.html'

# --------------------------------------------------------------------------- #
#                             Intersphinx config                              #
# --------------------------------------------------------------------------- #

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

# --------------------------------------------------------------------------- #
#                             Autosummary config                              #
# --------------------------------------------------------------------------- #

autosummary_generate = True
# autosummary_generate = glob.glob("reference/*.rst")

# --------------------------------------------------------------------------- #
#                               Autodoc config                                #
# --------------------------------------------------------------------------- #

autodoc_default_options = {
    'inherited-members': None,
}
