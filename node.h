#include <vector>
#include <cstdlib>
#include <random>
#ifndef PROJ_NODE_H
#define PROJ_NODE_H

class node {
public:
    int feat_dim;
    int num_heads;
    int msg_dim;
    float** input_feats;  // (num_heads, feat_dim)
    float** msgs;  // (num_heads, msg_dim)
    float** next_input_feats;  // (num_heads, feat_dim)

    node(int n_feats, int n_heads, int message_dim) : feat_dim(n_feats), num_heads(n_heads),
                                                      msg_dim(message_dim){
      input_feats = (float**)calloc(sizeof(float*), num_heads);
      for (int i = 0; i < num_heads; i++) {
        input_feats[i] = (float*)calloc(sizeof(float), feat_dim);
      }
      msgs = (float**)calloc(sizeof(float*), num_heads);
      for (int i = 0; i < num_heads; i++) {
        msgs[i] = (float*)calloc(sizeof(float), msg_dim);
      }
      next_input_feats = (float**)calloc(sizeof(float*), num_heads);
      for (int i = 0; i < num_heads; i++) {
        next_input_feats[i] = (float*)calloc(sizeof(float), feat_dim);
      }

    }

    void random_init() {
      std::default_random_engine generator;
      std::normal_distribution<float> distribution(0.f,0.1f);
      for (int i = 0; i < num_heads; i++) {
        for (int j = 0; j < feat_dim; j++) {
          input_feats[i][j] = distribution(generator);
        }
      }

//      for (int i = 0; i < num_heads; i++) {
//        for (int j = 0; j < feat_dim; j++) {
//          input_feats[i][j] = 0.1f;
//        }
//      }
    }
    void flush() {
      float** tmp = input_feats;
      input_feats = next_input_feats;
      next_input_feats = tmp;
      zero_out();
    }

    void zero_out() {
      for (int i = 0; i < num_heads; i++) {
        for (int j = 0; j < feat_dim; j++)
          next_input_feats[i][j] = 0.f;
      }
    }
    ~node() {
      for (int i = 0; i < num_heads; i++) {
        free(input_feats[i]);
      }
      free(input_feats);
      for (int i = 0; i < num_heads; i++) {
        free(msgs[i]);
      }
      free(msgs);
      for (int i = 0; i < num_heads; i++) {
        free(next_input_feats[i]);
      }
      free(next_input_feats);
    }
};

#endif //PROJ_NODE_H
