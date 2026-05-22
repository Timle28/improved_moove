project = "Moove"
copyright = "2025, Nils Riekers, Franziska Heubach, Jacqueline Laura Goebl, Lena Veit"
author = "Nils Riekers, Franziska Heubach, Jacqueline Laura Goebl, Lena Veit"
release = "1.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/images/image1.png"
html_theme_options = {
    "logo_only": True,
    "navigation_depth": 4,
    "collapse_navigation": False,
}

numfig = True
numfig_format = {
    "figure": "Figure %s",
    "table": "Table %s",
    "code-block": "Listing %s",
}
