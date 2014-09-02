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
#define NGRAM_ORDER 3
#define ETA 0.1

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

  static void AStep(double& p, double& h, const double& g) {
    if (!g) return;
    h += g * g;
    p -= ETA / sqrt(h) * g;
  }
  void UpdateAdagrad(Params<double>& h, Params<adouble>& g) {
    for (unsigned i = 0; i < bias.size(); ++i) {
      AStep(bias[i], h.bias[i], g.bias[i].get_gradient());
      for (unsigned j = 0; j < DIMENSION; ++j) {
        AStep(clookup[i](j,0), h.clookup[i](j,0), g.clookup[i](j,0).get_gradient());
        AStep(plookup[i](j,0), h.plookup[i](j,0), g.plookup[i](j,0).get_gradient());
      }
    for (unsigned c = 0; c < cmat.size(); ++c)
      for (unsigned i = 0; i < DIMENSION; ++i)
        for (unsigned j = 0; j < DIMENSION; ++j)
          AStep(cmat[c](i,j), h.cmat[c](i,j), g.cmat[c](i,j).get_gradient());
    }
  }
  template <typename T>
  void CopyFrom(const Params<T>& other) {
    for (unsigned i = 0; i < bias.size(); ++i) {
      bias[i] = other.bias[i];
      for (unsigned j = 0; j < DIMENSION; ++j) {
        clookup[i](j,0) = other.clookup[i](j,0);
        plookup[i](j,0) = other.clookup[i](j,0);
      }
    }
    for (unsigned c = 0; c < cmat.size(); ++c)
      for (unsigned i = 0; i < DIMENSION; ++i)
        for (unsigned j = 0; j < DIMENSION; ++j)
          cmat[c](i,j) = other.cmat[c](i,j);
  }
};

template <typename F>
F log_loss(const Params<F>& p,
           const vector<unsigned>& context,
           const unsigned w) {
  // log loss
  Eigen::Matrix<F, DIMENSION, 1> pred = Eigen::Matrix<F, DIMENSION, 1>::Zero();
  for (unsigned i = 0; i < NGRAM_ORDER-1; ++i)
    pred += p.cmat[i] * p.clookup[context[context.size() - NGRAM_ORDER + i]];
  F z = 0.0;
  F gold = 0.0;
  for (unsigned v = 1; v < vocab.size(); ++v) {
    F score = pred.dot(p.plookup[v]) + p.bias[v];
    z += exp(score);
    if (v == w) gold = score;
  }
  return log(z) - gold;
}

template <class T> void Randomize(T& v) { v = T::Random() / 10.0; }

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
  for (unsigned iter = 0; iter < 200; ++iter) {
    aparams.CopyFrom(params);
    s.new_recording();
    adouble loss = 0;
    unsigned chars = 0;
    for (auto& line : corpus) {
      ctx.resize(NGRAM_ORDER - 1, START);
      for (auto& w : line) {
        loss += log_loss(aparams, ctx, w);
        ctx.push_back(w);
        ++chars;
      }
      loss += log_loss(aparams, ctx, STOP);
      ++chars;
    }
    loss.set_gradient(1.0);
    s.compute_adjoint();
    cerr << "perplexity = " << exp(loss.value() / chars) << endl;
    params.UpdateAdagrad(h, aparams);
  }
  return 0;
}

