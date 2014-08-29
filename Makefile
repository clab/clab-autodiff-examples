all: crf

crf: crf.cc conll.cc
	g++ -g -O3 -o $@ crf.cc conll.cc -ladept -std=c++11 -Wall

clean:
	rm -f crf
