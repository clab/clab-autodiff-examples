This example shows how to train a log bilinear language model using automatic differentiation in place of explicitly coding the back propagation algorithm. This is not very efficient (it does one update per batch), but again, this is for pedagogical purposes to show the simplicity of coding up just forward algorithms and getting derivatives from the AD library.

To compile and run this code:

    make
    ./lbl train.txt

