#include <random>
#include <cstdlib>
#include <cstdio>

#include "sparse.h"

void gatForwardCUDA(float *W, float *A, float *input_feats, sparse_matrix *adj, int in_dim,
                    int out_dim, int num_heads, int num_nodes, float *output_feats, float min_f);

int main(int argc, char **argv) {
  sparse_matrix adj = sparse_matrix("../data/generated/5000.1e-1.adj.txt");
  int num_nodes = adj.num_rows;
  int in_dim = 1433;
//  int in_dim = 500;
  int num_heads = 8;
  int out_dim = 64;
  std::default_random_engine generator;
  std::normal_distribution<float> distribution(0.f, 0.1f);
  float *input_feats, *W, *A, *output_feats;
  input_feats = new float[num_nodes * in_dim];
  W = new float[in_dim * num_heads * out_dim];
  A = new float[num_heads * 2 * out_dim];
  output_feats = new float[num_nodes * num_heads * out_dim];
  for (int i = 0; i < num_nodes * in_dim; i++) {
    input_feats[i] = distribution(generator);
  }
  for (int i = 0; i < in_dim * num_heads * out_dim; i++) {
    W[i] = distribution(generator);
  }
  for (int i = 0; i < num_heads * 2 * out_dim; i++) {
    A[i] = distribution(generator);
  }

  gatForwardCUDA(W, A, input_feats, &adj, in_dim, out_dim, num_heads, num_nodes, output_feats,
                 -std::numeric_limits<float>::max());
  delete A;
  delete W;
  delete input_feats;
  delete output_feats;

  return 0;
}


