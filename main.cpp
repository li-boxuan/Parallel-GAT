#include "sparse.h"
#include "node.h"
#include "gat.h"

int main() {
  // load PPI graph adjacency matrix
  sparse_matrix adj = sparse_matrix();
  int num_nodes = adj.num_rows;
  int feat_dim = 50;
  int num_heads = 8;
  int msg_dim = 121;

  // load PPI graph node features and labels
  std::ifstream in_feat("ppi_features.txt");
  std::ifstream in_label("ppi_labels.txt");
  if (!in_feat || !in_label) {
    std::cout << "Cannot load graph input\n";
    return 1;
  }
  node **nodes = (node **) calloc(sizeof(node *), num_nodes);;
  for (int i = 0; i < num_nodes; i++) {
    nodes[i] = new node(feat_dim, num_heads, msg_dim);
    for (int j = 0; j < feat_dim; j++) {
      in_feat >> nodes[i]->input_feats[j];
    }
    for (int j = 0; j < msg_dim; j++) {
      in_label >> nodes[i]->label[j];
    }
  }

  // load trained GAT model
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

  // Predict and calculate micro-F1 score
  // reference output in PyTorch: ~0.60
  // micro-F1 score = TP / (TP + 0.5 * (FP + FN))
  int TP = 0, FP = 0, FN = 0;
  for (int i = 0; i < num_nodes; i++) {
    // average outputs across multiple heads
    for (int j = 0; j < msg_dim; j++) {
      float output = 0.0f;
      for (int k = 0; k < num_heads; k++) {
        output += nodes[i]->output_feats[k][j];
      }
      int pred = output >= 0 ? 1 : 0;
      int label = nodes[i]->label[j];
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
  float micro_f1_score = TP / (TP + 0.5 * (FP + FN));
  std::cout << "Micro F1 score = " << micro_f1_score << std::endl;

  for (int i = 0; i < num_nodes; i++) {
    delete (nodes[i]);
  }
  free(nodes);
}

