# QOC
QOC performs quantum optimal control. It uses automatic differentiation to do backpropagation through the Schroedinger Equation or the Lindblad Master Equation.

[Documentation](https://qoc.readthedocs.io/en/latest/) is coming soon. For now, you can read the docstrings in the source code.

[Tutorial](https://github.com/SchusterLab/qoc/tree/master/examples)

Tips for manual_gradient:
1. Set COMPLEX_CONTROLS to False, only real control amplitudes are supported.
2. The sequence of CONTROL_HAMILTONIAN should be consistent with the one in hamiltonia
3. Manual mode only supports cost_eval_step=1

### Installation ###
You can install QOC locally via pip.
```
git clone https://github.com/SchusterLab/qoc.git
cd qoc
pip install -e .
```

### Contact ###
QOC was originally developed at [Schuster Lab](http://schusterlab.uchicago.edu) at the University of Chicago.
If you have a feature request, a question about QOC's functionality, or a bug report, please make a github issue.
If you have another inquiry, you may contact [Thomas Propson](mailto:tcpropson@pm.me)
or [David Schuster](mailto:David.Schuster@uchicago.edu)
