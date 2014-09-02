clab-autodiff-examples
======================

This repository contains some small examples of how to solve standard NLP learning problems using [automatic differentiation](http://en.wikipedia.org/wiki/Automatic_differentiation) to compute the derivatives of the objective function.

The included examples are:

 * A linear-chain [conditional random field](http://www-bcf.usc.edu/~feisha/pubs/shallow03.pdf)
 * A [log bilinear language model](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2007_MnihH07.pdf) (Section 4)

For ease of reading, the code is not optimized, but automatic differentiation can be quite fast, and it can be used in large-scale learning problems.

To build the code, you need:

 * A C++11 compiler
 * The [Adept (Automatic Differentiation using Expression Templates) library](http://www.met.reading.ac.uk/clouds/adept/)
 * The [Eigen](http://eigen.tuxfamily.org/) linear algebra library (for the LBL example)

[Here is a crash course on automatic differentiation that I put together.](http://demo.clab.cs.cmu.edu/cdyer/autodiff.pdf)

#### Further reading

 * [autodiff.org](http://www.autodiff.org/) - lots of pointers
 * [Fast Reverse-Mode Automatic Differentiation using Expression Templates in C++](http://www.met.reading.ac.uk/clouds/publications/adept.pdf) - paper describing the adept library
 * [Recipes for Adjoint Code Construction](http://twister.ou.edu/OBAN2010/Giering_recipe4adjoint.pdf) - readable discussion about constructing adjoint mode (backward) automatic differentiation code
