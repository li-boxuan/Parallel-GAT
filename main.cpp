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

  // single layer GAT demo
  GAT gat_1 = GAT(num_nodes, num_heads, msg_dim);
  GAT gat_2 = GAT(num_nodes, num_heads, msg_dim);
  gat_1.random_init();
  gat_2.random_init();
  gat_1.forward(nodes, &adj);
  gat_2.forward(nodes, &adj);

  // multi layer GAT demo
  GAT level1 = GAT(num_nodes, 4, 256);
  level1.random_init();
  level1.forward(nodes, &adj);
//  GAT level2 = GAT(num_nodes, 4, 256);
//  level2.random_init();
//  level2.forward(nodes, &adj);
//  GAT level3 = GAT(num_nodes, 6, 121);
//  level3.random_init();
//  level3.forward(nodes, &adj);

  for (int i = 0; i < num_nodes; i++) {
    delete(nodes[i]);
  }
  free(nodes);
}

