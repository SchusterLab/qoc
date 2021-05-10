# QOC
This work has been followed up [here](https://github.com/SchusterLab/rbqoc)

## About
QOC is an implementation of the GRAPE algorithm for Quantum Optimal Control.
QOC uses [autograd](https://github.com/HIPS/autograd) to perform automatic
differentiation. QOC provides an extensible framework for adding cost functions and supports problems on
the Schroedinger Equation and the Lindblad Master Equation.

Formal documentation for this package does not exist, but the code is heavily commented.
See the [tutorial](https://github.com/SchusterLab/qoc/tree/master/examples) to get started
or check out the [docstrings](https://github.com/SchusterLab/qoc/blob/master/qoc/core/schroedingerdiscrete.py#L123)
for the core functions.

## Install
```
pip install git+https://github.com/SchusterLab/qoc
```

## Contact
QOC was originally developed at [Schuster Lab](http://schusterlab.uchicago.edu) at the University of Chicago.
If you have a feature request, a question about QOC's functionality, or a bug report, please make a github issue.
If you have another inquiry, you may contact [Thomas Propson](mailto:tcpropson@pm.me)
or [David Schuster](mailto:David.Schuster@uchicago.edu)
