#ifndef PROJ_PARAM_H
#define PROJ_PARAM_H

#include <random>
#include <cstdlib>

class param_single_head {
public:
    float **W;
    float *A1;
    float *A2;
    int in_dim;
    int out_dim;

    param_single_head(int in_dimension, int out_dimension) :
        in_dim(in_dimension), out_dim(out_dimension) {
      W = (float **) calloc(sizeof(float *), out_dim);
      for (int i = 0; i < out_dim; i++) {
        W[i] = (float *) calloc(sizeof(float), in_dim);
      }
      A1 = (float *) calloc(sizeof(float), out_dim);
      A2 = (float *) calloc(sizeof(float), out_dim);
    };

    ~param_single_head() {
      for (int i = 0; i < out_dim; i++) {
        free(W[i]);
      }
      free(W);
      free(A1);
      free(A2);
    }

    void random_init() {
      std::default_random_engine generator;
      std::normal_distribution<float> distribution(0.f, 0.1f);
      for (int i = 0; i < out_dim; i++) {
        for (int j = 0; j < in_dim; j++) {
          W[i][j] = distribution(generator);
        }
      }
      for (int i = 0; i < out_dim; i++) {
        A1[i] = distribution(generator);
      }
      for (int i = 0; i < out_dim; i++) {
        A2[i] = distribution(generator);
      }
//      for (int i = 0; i < out_dim; i++) {
//        for (int j = 0; j < in_dim; j++) {
//          W[i][j] = 0.1f;
//        }
//      }
//      for (int i = 0; i < out_dim; i++) {
//        A1[i] = 0.1f;
//      }
//      for (int i = 0; i < out_dim; i++) {
//        A2[i] = 0.1f;
//      }
    }
};

#endif //PROJ_PARAM_H
