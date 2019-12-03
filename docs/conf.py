import sys
import os

# import recommonmark
# from recommonmark.transform import AutoStructify

# to allow autodoc to discover the documented modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

project = 'pysynth'
copyright = '2019, Jan Šimbera'
author = 'Jan Šimbera'

extensions = [
    'sphinx.ext.autodoc',
    'recommonmark',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

templates_path = ['_templates']

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinxdoc'

html_static_path = ['_static']

# At the bottom of conf.py
# def setup(app):
    # app.add_config_value('recommonmark_config', {
        # # 'url_resolver': (lambda url: github_doc_root + url),
        # 'auto_toc_tree_section': 'Contents',
    # }, True)
    # app.add_transform(AutoStructify)
