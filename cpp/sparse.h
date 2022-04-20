#ifndef PROJ_SPARSE_H
#define PROJ_SPARSE_H

#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iterator>
#include <cstdlib>

class sparse_matrix {
public:
    int num_rows;
    int num_elements;
    std::vector<int> col_idx;
    std::vector<int> delim;
    std::vector<float> vals;

    sparse_matrix() {
      std::string filename("../data/ppi_adj.txt");
      std::string line;
      std::stringstream ss;
      std::ifstream input_file(filename);
      if (!input_file.is_open()) {
        std::cerr << "Could not open the file - '"
                  << filename << "'" << std::endl;
      }
      std::getline(input_file, line);
      ss << line;
      ss >> num_rows >> num_elements;
      std::getline(input_file, line);
      std::istringstream iss{line};
      auto end = std::istream_iterator<int>();
      for (std::istream_iterator<int> p(iss); p != end; ++p) {
        col_idx.push_back(*p);
      }
      std::getline(input_file, line);
      std::istringstream iss2{line};
      for (std::istream_iterator<int> p(iss2); p != end; ++p) {
        delim.push_back(*p);
      }
      vals.reserve(num_elements);
    }
};

#endif //PROJ_SPARSE_H
