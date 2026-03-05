# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Rustling"
author = "Jackson L. Lee"
copyright = f"2026, {author}"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "autoapi.extension",
    "sphinx_copybutton",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

napoleon_google_docstring = True
napoleon_numpy_docstring = False

autoapi_dirs = ["../rustling"]
autoapi_file_patterns = ["*.pyi", "*.py"]  # prioritize .pyi stubs
autoapi_ignore = ["*/_lib_name*"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "imported-members",
]
autoapi_add_toctree_entry = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_show_sourcelink = False
html_logo = "_static/logo-with-text.svg"
html_favicon = "_static/favicon.ico"

html_sidebars = {
    # Remove the primary/left sidebar for all pages.
    "**": [],
}

html_theme_options = {
    "show_toc_level": 3,
    "secondary_sidebar_items": ["page-toc", "sidebar-ethical-ads.html"],
    "footer_start": ["sphinx-version"],
    "footer_center": ["copyright"],
}
