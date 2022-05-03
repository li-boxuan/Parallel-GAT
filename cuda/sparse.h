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
    int *col_idx;
    int *delim;
    float *vals;

    sparse_matrix(std::string fname) {
      std::string filename(fname);
      std::string line;
      std::stringstream ss;
      std::ifstream input_file(filename.c_str());
      if (!input_file.is_open()) {
        std::cerr << "Could not open the file - '"
                  << filename << "'" << std::endl;
      }
      std::getline(input_file, line);
      ss << line;
      ss >> num_rows >> num_elements;
      col_idx = new int[num_elements];
      delim = new int[num_rows + 1];
      vals = new float[num_elements];
      std::stringstream iss;
      std::getline(input_file, line);
      iss << line;
      for (int i = 0; i < num_elements; i++) {
        iss >> col_idx[i];
      }
      std::getline(input_file, line);
      std::stringstream jss;
      jss << line;
      for (int i = 0; i < num_rows; i++) {
        jss >> delim[i];
      }
    }

    ~sparse_matrix() {
      delete col_idx;
      delete delim;
      delete vals;
    }
};

#endif //PROJ_SPARSE_H
