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
    "qutip",
    "scipy",
]
DESCRIPTION = "a package for performing quantum optimal control"
AUTHOR = "Thomas Propson"
AUTHOR_EMAIL = "tcpropson@uchicago.edu"
PY_MODULE = []
setup(author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      description=DESCRIPTION,
      install_requires=DEPENDENCIES,
      name=NAME,
      version=VERSION,
        py_modules=PY_MODULE
)
