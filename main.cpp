#include "sparse.h"
#include "node.h"
#include "gat.h"

int runPPIGraph() {
  // load PPI graph adjacency matrix
  sparse_matrix adj = sparse_matrix();
  int num_nodes = adj.num_rows;

  int in_dim = 50;
  int num_heads = 4;
  int msg_dim = 64;
  int out_dim = num_heads * msg_dim;
  Nodes inputs = Nodes(num_nodes, in_dim, out_dim, 121);
  Nodes *inputs_ptr = &inputs;
  inputs_ptr->load_input_features("data/ppi_features.txt");
  inputs_ptr->load_labels("data/ppi_labels.txt");

  GAT gat_1 = GAT(num_nodes, num_heads, in_dim, msg_dim);
  gat_1.load_params("models/gat_ppi_model_layer1.txt");
  gat_1.forward(inputs_ptr, &adj);
  gat_1.activate(inputs_ptr);

  num_heads = 4;
  msg_dim = 64;
  out_dim = num_heads * msg_dim;
  inputs_ptr->flush(num_heads * msg_dim);
  GAT gat_2 = GAT(num_nodes, num_heads, out_dim, msg_dim);
  gat_2.load_params("models/gat_ppi_model_layer2.txt");
  gat_2.forward(inputs_ptr, &adj);
  gat_2.activate(inputs_ptr);

  num_heads = 6;
  msg_dim = 121;
  out_dim = num_heads * msg_dim;
  inputs_ptr->flush(num_heads * msg_dim);
  GAT gat_3 = GAT(num_nodes, num_heads, out_dim, msg_dim);
  gat_3.load_params("models/gat_ppi_model_layer3.txt");
  gat_3.forward(inputs_ptr, &adj);

  // Predict and calculate micro-F1 score
  // reference output in PyTorch: ~0.60
  // micro-F1 score = TP / (TP + 0.5 * (FP + FN))
  int TP = 0, FP = 0, FN = 0;
  for (int i = 0; i < num_nodes; i++) {
    // average outputs across multiple heads
    for (int j = 0; j < msg_dim; j++) {
      float output = 0.0f;
      for (int k = 0; k < num_heads; k++) {
        output += inputs_ptr->output_feats[i][j * num_heads + k];
      }
      int pred = output >= 0 ? 1 : 0;
      int label = inputs_ptr->labels[i][j];
      if (pred == 1 && label == 1) {
        TP++;
      } else if (pred != label) {
        if (pred == 1) {
          FP++;
        } else {
          FN++;
        }
      }
    }
  }
  std::cout <<"TP = " << TP << " FP = " << FP << " FN = " << FN << std::endl;
  float micro_f1_score = TP / (TP + 0.5 * (FP + FN));
  std::cout << "Micro F1 score = " << micro_f1_score << std::endl;

//  int fc_input_dim = 100;
//  int fc_output_dim = 50;
//  inputs_ptr->flush(fc_input_dim);
//  FC fc = FC(num_nodes, fc_input_dim, fc_output_dim, 1);
//  fc.forward(inputs_ptr);
  return 0;
}

int main() {
  return runPPIGraph();
}

