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

