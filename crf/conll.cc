#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cassert>

using namespace std;

void ReadCoNLL(const char* fname, vector<vector<vector<string>>>& x, vector<vector<string>>& y) {
  cerr << "Read " << fname << endl;
  ifstream in(fname);
  assert(in);
  string line;
  x.push_back(vector<vector<string>>());
  y.push_back(vector<string>());
  while(getline(in, line)) {
    if (line.size() == 0) {
      // new sentence
      x.push_back(vector<vector<string>>());
      y.push_back(vector<string>());
    } else { // new word
      auto& cur_x = x.back();
      auto& cur_y = y.back();
      cur_x.push_back(vector<string>());
      auto& xfeats = cur_x.back();
      unsigned cur = 0;
      while(cur < line.size()) {
        unsigned start = cur;
        while(cur < line.size() && line[cur] != ' ') { cur++; }
        if (cur == line.size()) {
          cur_y.push_back(line.substr(start));
        } else {
          xfeats.push_back(line.substr(start, cur - start));
        }
        cur++;
      }
    }
  }
  x.pop_back();
  y.pop_back();
}

