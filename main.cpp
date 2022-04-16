#include "sparse.h"
#include "node.h"
#include "gat.h"
#include <cstdio>

int main() {
  sparse_matrix adj = sparse_matrix();
  int num_nodes = adj.num_rows;
  int feat_dim = 50;
  int num_heads = 8;
  int msg_dim = 121;
  node **nodes = (node **) calloc(sizeof(node *), num_nodes);;
  for (int i = 0; i < num_nodes; i++) {
    nodes[i] = new node(feat_dim, num_heads, msg_dim);
    nodes[i]->random_init();
  }

  std::ifstream in("models/gat_ppi_model.txt");
  if (!in) {
    std::cout << "Cannot open models/gat_ppi_model.txt\n";
    return 1;
  }

  GAT gat_1 = GAT(num_nodes, num_heads, feat_dim, msg_dim);
  for (int i = 0; i < num_heads; i++) {
    for (int j = 0; j < msg_dim; j++) {
      in >> gat_1.params[i]->A1[j];
    }
  }
  for (int i = 0; i < num_heads; i++) {
    for (int j = 0; j < msg_dim; j++) {
      in >> gat_1.params[i]->A2[j];
    }
  }
  for (int i = 0; i < num_heads; i++) {
    for (int k = 0; k < msg_dim; k++) {
      for (int j = 0; j < feat_dim; j++) {
        in >> gat_1.params[i]->W[k][j];
      }
    }
  }
  in.close();
  gat_1.forward(nodes, &adj);

  for (int i = 0; i < num_nodes; i++) {
    delete (nodes[i]);
  }
  free(nodes);
}

