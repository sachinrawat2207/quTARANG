# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import mock
import sys
from pathlib import Path
path = Path(__file__).parents[2]/'src'
sys.path.insert(0, str(path))
print(path)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'quTARANG'
copyright = '2023, S.S. Rawat, S.K. Jha'
author = 'S.S. Rawat, S.K. Jha'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
MOCK_MODULES = ['numpy', 'cupy', 'h5py', 'matplotlib', 'pyfftw', 'tqdm']
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()

extensions = ['sphinx_copybutton',
'sphinx.ext.duration',
'sphinx.ext.doctest',
'sphinx.ext.autodoc',
'sphinxcontrib.bibtex',
'sphinx.ext.autosummary', 
'sphinx_design']

bibtex_bibfiles = ['refs.bib']

templates_path = ['_templates']
exclude_patterns = []
numfig = True
math_numfig = True

autosummary_generate = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# def setup(app):
#     app.add_css_file('css/custom.css')
