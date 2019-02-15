import os
import sys

from sphinx.ext.autosummary import _import_by_name
from numpydoc.docscrape import NumpyDocString
from numpydoc.docscrape_sphinx import SphinxDocString

sys.path.insert(0, os.path.abspath('../..'))

extensions = [
    'sphinx.ext.autodoc',
    'numpydoc',
    'sphinx.ext.intersphinx', 
    'sphinx.ext.coverage',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'matplotlib.sphinxext.plot_directive',
]

project = 'fynance'
copyright = '2018-2019, Arthur Bernard'
author = 'Arthur Bernard'

version = "1.0" #fy.__version__
release = "1.0.4" #fy.__version__

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
pygments_style = 'sphinx'

add_function_parentheses = False

html_theme = 'sphinx_rtd_theme'
html_theme_option = {
	'display_version': True,
    'prev_next_buttons_location': 'both',
    'style_external_links': True,
    'vcs_pageview_mode' : '',
    'style_nav_header_background': 'white',
    # Toc options
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
    'github_url': 'https://github.com/ArthurBernard/Fynance',
}
html_static_path = ['_static']
html_context = {
    "display_github": True, # Integrate GitHub
    "github_user": "ArthurBernard", # Username
    "github_repo": "Fynance", # Repo name
    "github_version": "master", # Version
    "conf_py_path": "/source/", # Path in the checkout to the docs root
}

autosummary_generate = True