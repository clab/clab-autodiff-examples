#include <iostream>
#include <fstream>
#include <cassert>
#include <vector>

#include "Eigen/Core"
#include "adept.h"
#include "corpus.h"

using namespace std;
using adept::adouble;

#define DIMENSION 10
#define CONTEXT 3

typedef Eigen::Matrix<double, DIMENSION, 1> RVector;
typedef Eigen::Matrix<double, DIMENSION, DIMENSION> CMatrix;
typedef Eigen::Matrix<adouble, DIMENSION, 1> RAVector;
typedef Eigen::Matrix<adouble, DIMENSION, DIMENSION> CAMatrix;
Dict d;

vector<vector<unsigned>> corpus;
set<unsigned> vocab;

template <typename F>
struct Params {
  explicit Params(unsigned vocab_size) :
    clookup(vocab_size, Eigen::Matrix<F, DIMENSION, 1>::Zero()),
    plookup(vocab_size, Eigen::Matrix<F, DIMENSION, 1>::Zero()),
    cmat(vocab_size, Eigen::Matrix<F, DIMENSION, DIMENSION>::Zero()),
    bias(vocab_size, 0.0) {}
  vector<Eigen::Matrix<F, DIMENSION, 1>> clookup;
  vector<Eigen::Matrix<F, DIMENSION, 1>> plookup;
  vector<Eigen::Matrix<F, DIMENSION, DIMENSION>> cmat;
  vector<F> bias;
};

template <typename F>
F log_loss(const Params<F>& p,
           const vector<unsigned>& context,
           const unsigned w) {
  // log loss
  Eigen::Matrix<F, DIMENSION, 1> pred = Eigen::Matrix<F, DIMENSION, 1>::Zero();
  for (unsigned i = 0; i < CONTEXT-1; ++i)
    pred += p.cmat[i] * p.clookup[context[context.size() - CONTEXT + i]];
  F z = 0.0;
  F gold = 0.0;
  for (unsigned v = 1; v < vocab.size(); ++v) {
    F score = pred.dot(p.plookup[v]) + p.bias[v];
    z += exp(score);
    if (v == w) gold = score;
  }
  return gold - log(z);
}

template <class T> void Randomize(T& v) { v = T::Random() / 5.0; }

int main(int argc, char** argv) {
  if (argc != 2) {
    cerr << "Usage: " << argv[0] << " training.txt\n";
    return 1;
  }
  const unsigned START = d.Convert("<s>");
  const unsigned STOP = d.Convert("</s>");
  ReadFromFile(argv[1], &d, &corpus, &vocab);
  cerr << "|vocab| = " << vocab.size() << endl;

  // parameters
  Params<double> params(vocab.size());
  for (auto& v : params.clookup) Randomize(v);
  for (auto& v : params.plookup) Randomize(v);
  for (auto& m : params.cmat) Randomize(m);
  for (auto& b : params.bias) b = 0.0;

  // adagrad diagonal
  Params<double> h(vocab.size());

  // ad parameters
  adept::Stack s;
  Params<adouble> aparams(vocab.size());

  vector<unsigned> ctx;
  for (unsigned iter = 0; iter < 20; ++iter) {
    s.new_recording();
    double loss = 0;
    unsigned chars = 0;
    for (auto& line : corpus) {
      ctx.resize(CONTEXT - 1, START);
      for (auto& w : line) {
        loss += log_loss(params, ctx, w);
        ctx.push_back(w);
        ++chars;
      }
      loss += log_loss(params, ctx, STOP);
      ++chars;
    }
    cerr << "perplexity = " << exp(-loss / chars) << endl;
  }
  return 0;
}

