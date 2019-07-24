"""
setup.py - a module to allow package installation
"""

from distutils.core import setup


NAME = "qoc"
VERSION = "0.1alpha"
DESCRIPTION = "a package for performing quantum optimal control"
AUTHOR = "Thomas Propson"
AUTHOR_EMAIL = "tcpropson@uchicago.edu"

setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
)
