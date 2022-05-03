#include <cstdlib>
#include <cstdio>
#include "node.h"
#include "param.h"
#include "../sparse.h"
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
    float **heats_1;  // (num_heads, num_nodes)
    float **heats_2;  // (num_heads, num_nodes)

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
      heats_1 = (float **) calloc(sizeof(float *), num_heads);
      for (int i = 0; i < num_heads; i++) {
        heats_1[i] = (float *) calloc(sizeof(float), num_nodes);
      }
      heats_2 = (float **) calloc(sizeof(float *), num_heads);
      for (int i = 0; i < num_heads; i++) {
        heats_2[i] = (float *) calloc(sizeof(float), num_nodes);
      }
    }

    void forward(Nodes *features, sparse_matrix *adj) {
      // prepare messages
      for (int i = 0; i < num_heads; i++) {
        for (int j = 0; j < num_nodes; j++) {
          for (int row_idx = 0; row_idx < msg_dim; row_idx++) {
            for (int col_idx = 0; col_idx < feat_dim; col_idx++) {
              msgs[i][j][row_idx] += params[i]->W[row_idx][col_idx] * features->input_feats[j][col_idx];
            }
          }
        }
      }

      for (int i = 0; i < num_heads; i++) {
        for (int j = 0; j < num_nodes; j++) {
          float heat_1 = 0.f;
          float heat_2 = 0.f;
          for (int v = 0; v < msg_dim; v++) {
            heat_1 += msgs[i][j][v] * params[i]->A1[v];
            heat_2 += msgs[i][j][v] * params[i]->A2[v];
          }
          heats_1[i][j] = heat_1;
          heats_2[i][j] = heat_2;
        }
      }

      for (int i = 0; i < num_heads; i++) {
        for (int j = 0; j < num_nodes; j++) {
          int start_idx = adj->delim[j];
          int end_idx = adj->delim[j + 1];
          affinity_sum[j] = 0.f;
          float max_attention = -std::numeric_limits<float>::max();
          for (int k = start_idx; k < end_idx; k++) {
            int neighbor_idx = adj->col_idx[k];
            float curr_affinity = heats_1[i][j] + heats_2[i][neighbor_idx];
            curr_affinity = leaky_relu(curr_affinity);
            // subtract max attention for numerical stability
            max_attention = std::max(max_attention, curr_affinity);
          }
          for (int k = start_idx; k < end_idx; k++) {
            int neighbor_idx = adj->col_idx[k];
            float curr_affinity = heats_1[i][j] + heats_2[i][neighbor_idx];
            curr_affinity = leaky_relu(curr_affinity) - max_attention;
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
      for (int i = 0; i < num_heads; i++) {
        free(heats_1[i]);
      }
      free(heats_1);
      for (int i = 0; i < num_heads; i++) {
        free(heats_2[i]);
      }
      free(heats_2);
    }

    void random_init() {
      for (int i = 0; i < num_heads; i++) {
        params[i]->random_init();
      }
    }

};

#endif //PROJ_GAT_H
