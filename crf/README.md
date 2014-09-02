This directory contains a very simple (and, when it comes to feature computation, very inefficient) C++ program to learn the parameters of a linear chain CRF model using [automatic differentiation](http://en.wikipedia.org/wiki/Automatic_differentiation).

The classical forward algorithm is used to compute the CRF training objective, and rather than explicitly writing the backward algorithm to compute the derivatives, I rely on autodifferentiation. This makes the code much less complicated to write and debug and is only minimally less efficient than a hand-coded backward algorithm

To compile and run this code:

    make
    ./crf sample.conll

