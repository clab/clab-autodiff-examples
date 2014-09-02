clab-autodiff-examples
======================

This repository contains some small examples of how to solve standard NLP learning problems using [automatic differentiation](http://en.wikipedia.org/wiki/Automatic_differentiation) to compute the derivatives of the objective function.

Examples included are:

 * A linear chain [conditional random field](http://www-bcf.usc.edu/~feisha/pubs/shallow03.pdf)
 * A [log bilinear language model](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2007_MnihH07.pdf) (Section 4)

To build the code, you need:

 * A C++11 compiler
 * The [Adept (Automatic Differentiation using Expression Templates) library](http://www.met.reading.ac.uk/clouds/adept/) installed
 * The [Eigen](http://eigen.tuxfamily.org/) linear algebra library installed (for the LBL example)

