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
//    int *col_idx;
//    int *delim;
//    float *vals;

    sparse_matrix() {
//      num_elements = 19;
//      num_rows = 8;
//      col_idx = (int *) calloc(sizeof(int), num_elements);
//      vals = (float *) calloc(sizeof(float), num_elements);
//      delim = (int *) calloc(sizeof(int), num_rows);
//      int col_idx_[] = {0, 4, 7, 1, 4, 6, 2, 3, 5, 2, 3, 1, 4, 2, 5, 1, 6, 0, 7};
//      int delim_[] = {0, 3, 6, 9, 11, 13, 15, 17, 19};
//      for (int i = 0; i < num_elements; i++) {
//        col_idx[i] = col_idx_[i];
//      }
//      for (int i = 0; i < num_rows; i++) {
//        delim[i] = delim_[i];
//      }

      std::string filename("data/ppi_adj.txt");
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

    ~sparse_matrix() {
//      free(col_idx);
//      free(vals);
//      free(delim);
    }
};

#endif //PROJ_SPARSE_H
