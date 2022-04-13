//#include <vector>
#include <cstdlib>
#include <cstdio>
#include "node.h"
#include "param.h"
#include "sparse.h"
#include <math.h>

#ifndef PROJ_GAT_H
#define PROJ_GAT_H

float leaky_relu(float val) {
  if (val < 0.f) {
    return 0.f;
  }
  return 0.2f * val;
}

class GAT {
public:
    int num_nodes;
    int num_heads;
    int msg_dim;  // per head
    int feat_dim;
    param_single_head **params;
    float *affinity_sum;

    /**
     *
     * @param n_nodes
     * @param n_heads
     * @param message_dim
     */
    GAT(int n_nodes, int n_heads, int message_dim) :
        num_nodes(n_nodes), num_heads(n_heads), msg_dim(message_dim) {
      feat_dim = n_heads * msg_dim;
      params = (param_single_head **) calloc(sizeof(param_single_head *), num_heads);
      for (int i = 0; i < num_heads; i++) {
        params[i] = new param_single_head(feat_dim, msg_dim);
      }
      affinity_sum = (float*) calloc(sizeof(float), num_nodes);
    }


    void forward(node **nodes, sparse_matrix *adj) {
      // prepare messages
      for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < num_heads; j++) {
          for (int row_idx = 0; row_idx < msg_dim; row_idx++) {
            for (int col_idx = 0; col_idx < feat_dim; col_idx++) {
              nodes[i]->msgs[j][row_idx] += params[j]->W[row_idx][col_idx] *
                                            nodes[i]->input_feats[j][col_idx];
            }
          }
        }
      }
      for (int i = 0; i < num_heads; i++) {

        for (int j = 0; j < num_nodes; j++) {
          int start_idx = adj->delim[j];
          int end_idx = adj->delim[j + 1];
          affinity_sum[j] = 0.f;
          for (int k = start_idx; k < end_idx; k++) {
            int neighbor_idx = adj->col_idx[k];
            float curr_affinity = 0.f;
            for (int v = 0; v < msg_dim; v++) {
              curr_affinity += nodes[j]->msgs[i][v] * params[i]->A1[v];
            }
            for (int v = 0; v < msg_dim; v++) {
              curr_affinity += nodes[neighbor_idx]->msgs[i][v] * params[i]->A2[v];
            }
            curr_affinity = exp(leaky_relu(curr_affinity));
            adj->vals[neighbor_idx] = curr_affinity;
            affinity_sum[j] += curr_affinity;
          }
          // out: nodes[j].output_feats[i]
          for (int k = start_idx; k < end_idx; k++) {
            int neighbor_idx = adj->col_idx[k];
            float w = adj->vals[neighbor_idx] / affinity_sum[j];
            for (int v = 0; v < msg_dim; v++) {
              nodes[j]->output_feats[i][v] += w * nodes[neighbor_idx]->msgs[i][v];
            }
          }
        }
      }
      for (int i = 0; i < num_nodes; i++) {
        nodes[i]->flush();
      }
    }

    ~GAT() {
      for (int i = 0; i < num_heads; i++) {
        delete (params[i]);
      }
      free(params);
      free(affinity_sum);
    }

    void random_init() {
      for (int i = 0; i < num_heads; i++) {
        params[i]->random_init();
      }
    }

};

#endif //PROJ_GAT_H
