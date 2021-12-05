# QOC
QOC performs quantum optimal control. It uses automatic differentiation to do backpropagation through the Schroedinger Equation or the Lindblad Master Equation.

[Documentation](https://qoc.readthedocs.io/en/latest/) is coming soon. For now, you can read the docstrings in the source code.

[Tutorial](https://github.com/SchusterLab/qoc/tree/master/examples)

This branch implements auto-grad for control directly related cost functions such as penalization of
variance, total power and bandwidth. Analytical gradient is implemented for state transfer, gates optimization, etc.


Tutorial for this branch:[Tutorial](https://github.com/SchusterLab/qoc/tree/manual_gradient/Example/Analytical_example)


Tips for manual_gradient:
1. Set COMPLEX_CONTROLS to False, only real control amplitudes are supported.
2. Analytical gradients only supports cost_eval_step=1
3. Parameter "tol" means expectation error of the total cost. Sometime actual error might not match "tol".
The scheme is that we can choose "tol" to be a small value like 10^(-3) first. If cost value does not converge to the target one,
we take "tol"=10^(-16)(limitation of np.float64 type) to check if divergence is from accuracy. If it is, we change "tol" to be 10^(-6) or even smaller to make sure
convergence. The point is that we just keep the accuracy in a level that we need to speed up the optimization. In master branch, the "tol" is always set to be
10^(-16) which sometimes overkills the problem.   



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
