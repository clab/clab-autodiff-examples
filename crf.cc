#include <set>
#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <cassert>

#include "adept.h"

using namespace std;
using adept::adouble;

const double eta = 0.5;
const double lambda = 1.0;
const string START = "BOS";
const string STOP = "EOS";
const vector<string> start_labels = {START};
const vector<string> stop_labels = {STOP};

vector<string> labels;

void ReadCoNNL(const char* fn,
               vector<vector<vector<string>>>& xs,
               vector<vector<string>>& ys);

template <typename F>
F log_likelihood(const vector<vector<string>>& x,
                 const vector<string>& y,
                 map<string, F>& w) {
  assert(x.size() == y.size());
  const int len = x.size() + 1;
  vector<map<string, F>> fwd(len + 1);
  vector<map<string, F>> fwd_true(len + 1);
  fwd[0][START] = 1.0;
  fwd_true[0][START] = 1.0;
  F logtz = 0;
  F logz = 0;
  for (int i = 1; i <= len; ++i) {
    const auto& prev_labels = (i == 1) ? start_labels : labels;
    const auto& cur_labels = (i == len) ? stop_labels : labels;
    const auto& prev_truth = (i == 1) ? START : y[i - 2];
    const auto& cur_truth = (i == len) ? STOP : y[i - 1];

    for(auto& cur : cur_labels) {
      // observation features
      F obs_potential = 1.0;
      if (i < len) { // avoid STOP label which has no word
        const auto& xi_word = x[i-1][0]; // x is 0-indexed, loop is 1-indexed
        const auto& xi_pos = x[i-1][1];
        const string emit_char = "EmC:" + cur + "_" + xi_word[0];
        const string emit_pos = "EmP:" + cur + "_" + xi_pos;
        obs_potential = exp(w[emit_char] + w[emit_pos]);
      }

      // transition features
      const string uni = "Uni:" + cur;
      for(auto& prev : prev_labels) {
        string trans = "Bi:" + prev + "_" + cur;
        F trans_potential = exp(w[trans] + w[uni]);

        // recurrence for model & gold
        fwd[i][cur] += fwd[i-1][prev] * trans_potential * obs_potential;
        if (cur == cur_truth && prev == prev_truth)
          fwd_true[i][cur] += fwd_true[i-1][prev] * 
                              trans_potential * obs_potential;
      }
    }

    // time-step normalization to prevent underflow (could do with logsum)
    F colz = 0;
    for (auto& cur : cur_labels) colz += fwd[i][cur];
    for (auto& cur : cur_labels) fwd[i][cur] /= colz;
    logz += log(colz);
    logtz += log(fwd_true[i][cur_truth]);
    fwd_true[i][cur_truth] = 1.0;
  }

  return logtz - logz;
}

template <typename F>
F l2penalty(const double lambda, map<string, F>& w) {
  F res = 0.0;
  for (auto& kv : w)
    res += lambda * kv.second * kv.second / 2.0;
  return res;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    cerr << "Usage: " << argv[0] << " train.conll\n";
    return 1;
  }

  // read training data
  vector<vector<vector<string>>> train_x;
  vector<vector<string>> train_y;
  ReadCoNNL(argv[1], train_x, train_y);
  set<string> ls; for(auto& yy : train_y) for(auto& y : yy) ls.insert(y);
  cerr << "              LABELS:";
  for(auto& y : ls) { cerr << ' ' << y; labels.push_back(y); }
  cerr << endl;
  cerr << "# TRAINING INSTANCES: " << train_x.size() << endl;

  adept::Stack stack;
  map<string, double> h;
  map<string, double> weights;
  map<string, adouble> cweights;
  for (unsigned i = 0; i < train_x.size(); ++i)
    log_likelihood(train_x[i], train_y[i], weights);
  for (unsigned iter = 0; iter < 100; ++iter) {
    for (auto& kv : weights) cweights[kv.first] = kv.second;
    // compute CRF loss and its gradient
    stack.new_recording();
    adouble cll = 0.0;
    for (unsigned i = 0; i < train_x.size(); ++i)
      cll += log_likelihood(train_x[i], train_y[i], cweights);
    cll -= l2penalty(lambda, cweights);
    cll.set_gradient(1.0);
    stack.compute_adjoint();
    cerr << "Iteration " << (iter+1) << " loss: " << cll.value() << endl;
    // update weights with adagrad
    for (auto& fv : cweights) {
      const double g = fv.second.get_gradient();
      if (!g) continue;
      const double hh = h[fv.first] += g*g;
      weights[fv.first] += eta * g / sqrt(hh);
    }
  }
}

