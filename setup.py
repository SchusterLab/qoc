"""
setup.py - a module to allow package installation
"""

from distutils.core import setup


NAME = "qoc"
VERSION = "0.1alpha"
DEPENDENCIES = [
    "autograd",
    "filelock",
    "h5py",
    "matplotlib",
    "numba",
    "numpy",
    "scipy",
    "scqubits",
    "pathos"
]
DESCRIPTION = "a package for performing quantum optimal control"
AUTHOR = "Thomas Propson"
AUTHOR_EMAIL = "tcpropson@uchicago.edu"

setup(author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      description=DESCRIPTION,
      install_requires=DEPENDENCIES,
      name=NAME,
      version=VERSION,
)
