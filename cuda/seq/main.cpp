#include "node.h"
#include "gat.h"
#include "../CycleTimer.h"
#include "../sparse.h"

void run_cora() {
  // load PPI graph adjacency matrix
  sparse_matrix adj = sparse_matrix("../../data/generated/5000.3e-1.adj.txt");
  int num_nodes = adj.num_rows;

  int in_dim = 1433;
//  int in_dim = 500;
//  int in_dim = 50;
  int num_heads = 8;
  int out_dim = 64;
  Nodes inputs = Nodes(num_nodes, in_dim, out_dim * num_heads);
  Nodes *inputs_ptr = &inputs;
  inputs_ptr->random_init();

  GAT gat = GAT(num_nodes, num_heads, in_dim, out_dim);
  gat.random_init();

  double startTime = CycleTimer::currentSeconds();
  gat.forward(inputs_ptr, &adj);
  double endTime = CycleTimer::currentSeconds();
  double overallDuration = endTime - startTime;
  printf("Overall: %.3f ms\n", 1000.f * overallDuration);
}

int main() {
  run_cora();
  return 0;
}

