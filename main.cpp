#include "sparse.h"
#include "node.h"
#include "gat.h"
#include <cstdio>
int main(){
  sparse_matrix adj = sparse_matrix();
  int num_nodes = adj.num_rows;
  int feat_dim = 256;
  int num_heads = 8;
  int msg_dim = 32;
  node** nodes = (node**)calloc(sizeof(node*), num_nodes);;
  for (int i = 0; i < num_nodes; i++) {
    nodes[i] = new node(feat_dim, num_heads, msg_dim);
    nodes[i]->random_init();
  }
  GAT gat_1 = GAT(num_nodes, num_heads, msg_dim);
  GAT gat_2 = GAT(num_nodes, num_heads, msg_dim);
  gat_1.random_init();
  gat_2.random_init();
  gat_1.forward(nodes, &adj);
  gat_2.forward(nodes, &adj);
  for (int i = 0; i < num_nodes; i++) {
    delete(nodes[i]);
  }
  free(nodes);
}

