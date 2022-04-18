#include <cstdlib>
#include <cstdio>
#include "node.h"
#include "param.h"
#include "sparse.h"
#include <math.h>
#include <random>

#ifndef PROJ_GAT_H
#define PROJ_GAT_H

float leaky_relu(float val) {
  if (val < 0.f) {
    return 0.2 * val;
  }
  return val;
}

float identity(float val) {
  return val;
}

class GAT {
public:
    int num_nodes;
    int num_heads;
    int feat_dim;
    int msg_dim;  // per head
    param_single_head **params;
    float *affinity_sum;  // (num_nodes, )
    float ***msgs;  // (num_heads, num_nodes, msg_dim)

    /**
     *
     * @param n_nodes
     * @param n_heads
     * @param feature_dim
     * @param message_dim
     */
    GAT(int n_nodes, int n_heads, int feature_dim, int message_dim) :
        num_nodes(n_nodes), num_heads(n_heads), feat_dim(feature_dim), msg_dim(message_dim) {
      params = (param_single_head **) calloc(sizeof(param_single_head *), num_heads);
      for (int i = 0; i < num_heads; i++) {
        params[i] = new param_single_head(feat_dim, msg_dim);
      }
      affinity_sum = (float *) calloc(sizeof(float), num_nodes);
      msgs = (float ***) calloc(sizeof(float **), num_heads);
      for (int i = 0; i < num_heads; i++) {
        msgs[i] = (float **) calloc(sizeof(float *), num_nodes);
        for (int j = 0; j < num_nodes; j++) {
          msgs[i][j] = (float *) calloc(sizeof(float), msg_dim);
        }
      }
    }


    void forward(Nodes *features, sparse_matrix *adj) {
      // prepare messages
      for (int i = 0; i < num_heads; i++) {
        for (int j = 0; j < num_nodes; j++) {
          for (int row_idx = 0; row_idx < msg_dim; row_idx++) {
            for (int col_idx = 0; col_idx < feat_dim; col_idx++) {
              msgs[i][j][row_idx] += params[i]->W[row_idx][col_idx] * features->input_feats[i][col_idx];
            }
          }
        }
      }
      // TODO: hard-code pre-calculated max attention for now
      float max_attention = 40.7058;
      for (int i = 0; i < num_heads; i++) {
        for (int j = 0; j < num_nodes; j++) {
          int start_idx = adj->delim[j];
          int end_idx = adj->delim[j + 1];
          affinity_sum[j] = 0.f;
          for (int k = start_idx; k < end_idx; k++) {
            int neighbor_idx = adj->col_idx[k];
            float curr_affinity = 0.f;
            for (int v = 0; v < msg_dim; v++) {
              curr_affinity += msgs[i][j][v] * params[i]->A1[v];
            }
            for (int v = 0; v < msg_dim; v++) {
              curr_affinity += msgs[i][neighbor_idx][v] * params[i]->A2[v];
            }
            curr_affinity = leaky_relu(curr_affinity);
            curr_affinity -= max_attention;
            curr_affinity = exp(curr_affinity);
            adj->vals[neighbor_idx] = curr_affinity;
            affinity_sum[j] += curr_affinity;
          }
          // out: nodes[j].output_feats[i]
          for (int k = start_idx; k < end_idx; k++) {
            int neighbor_idx = adj->col_idx[k];
            float w = adj->vals[neighbor_idx] / (affinity_sum[j] + 1e-16);
            for (int v = 0; v < msg_dim; v++) {
              features->output_feats[j][i * msg_dim + v] += w * msgs[i][neighbor_idx][v];
            }
          }
        }
      }
    }

    ~GAT() {
      for (int i = 0; i < num_heads; i++) {
        delete (params[i]);
      }
      free(params);
      free(affinity_sum);
      for (int i = 0; i < num_heads; i++) {
        for (int j = 0; j < num_nodes; j++) {
          free(msgs[i][j]);
        }
        free(msgs[i]);
      }
      free(msgs);
    }

    void random_init() {
      for (int i = 0; i < num_heads; i++) {
        params[i]->random_init();
      }
    }

};

class FC {
public:
    int num_nodes;
    int input_dim;
    int output_dim;
    float **W;
    float *b;

    float (*activation)(float);

    FC(int n_node, int in_dim, int out_dim, int activate) : num_nodes(n_node), input_dim(in_dim), output_dim(out_dim) {
      W = (float **) calloc(sizeof(float *), output_dim);
      for (int i = 0; i < output_dim; i++) {
        W[i] = (float *) calloc(sizeof(float), input_dim);
      }
      b = (float *) calloc(sizeof(float), output_dim);
      if (activate == 0) {
        activation = &identity;
      } else {
        activation = &leaky_relu;
      }
    }

    void random_init() {
      std::default_random_engine generator;
      std::normal_distribution<float> distribution(0.f, 0.1f);
      for (int i = 0; i < output_dim; i++) {
        for (int j = 0; j < input_dim; j++) {
          W[i][j] = distribution(generator);
        }
      }
      for (int i = 0; i < output_dim; i++) {
        b[i] = distribution(generator);
      }
    }

    void forward(Nodes *features) {
      for (int i = 0; i < num_nodes; i++) {
        for (int row_idx = 0; row_idx < output_dim; row_idx++) {
          for (int col_idx = 0; col_idx < input_dim; col_idx++) {
            features->output_feats[i][row_idx] += W[row_idx][col_idx] * features->input_feats[i][col_idx];
          }
          features->output_feats[i][row_idx] = activation(features->output_feats[i][row_idx] + b[row_idx]);
        }
      }
    }

    ~FC() {
      for (int i = 0; i < output_dim; i++) {
        free(W[i]);
      }
      free(W);
      free(b);
    }
};

#endif //PROJ_GAT_H
