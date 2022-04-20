#include <vector>
#include <cstdlib>
#include <random>

#ifndef PROJ_NODE_H
#define PROJ_NODE_H

class Nodes {
public:
    int input_dim;
    int output_dim;
    int num_nodes;
    int label_dim;
    float **input_feats;
    float **output_feats;
    int **labels;

    Nodes(int n_node, int in_dim, int out_dim, int label_dim) :
    input_dim(in_dim), output_dim(out_dim), num_nodes(n_node), label_dim(label_dim) {
      input_feats = (float **) calloc(sizeof(float *), num_nodes);
      for (int i = 0; i < num_nodes; i++) {
        input_feats[i] = (float *) calloc(sizeof(float), input_dim);
      }
      output_feats = (float **) calloc(sizeof(float *), num_nodes);
      for (int i = 0; i < num_nodes; i++) {
        output_feats[i] = (float *) calloc(sizeof(float), output_dim);
      }
      labels = (int **) calloc(sizeof(int *), num_nodes);
      for (int i = 0; i < num_nodes; i++) {
        labels[i] = (int *) calloc(sizeof(int), label_dim);
      }
    }

    void load_labels(std::string label_file) {
      std::ifstream in_label("../data/ppi_labels.txt");
      if (!in_label) {
        std::cout << "Cannot load node labels\n";
        return;
      }
      for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < label_dim; j++) {
          in_label >> labels[i][j];
        }
      }
      in_label.close();
    }

    void load_input_features(std::string feature_file) {
      std::ifstream in_feat(feature_file);
      if (!in_feat) {
        std::cout << "Cannot load node features\n";
        return;
      }
      for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < input_dim; j++) {
          in_feat >> input_feats[i][j];
        }
      }
      in_feat.close();
    }

    void random_init() {
      std::default_random_engine generator;
      std::normal_distribution<float> distribution(0.f, 0.1f);
      for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < input_dim; j++) {
          input_feats[i][j] = distribution(generator);
        }
      }
      for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < output_dim; j++) {
          output_feats[i][j] = distribution(generator);
        }
      }
    }

    void flush(int new_output_dim) {
      float **tmp = input_feats;
      input_feats = output_feats;
      output_feats = tmp;
      input_dim = output_dim;
      for (int i = 0; i < num_nodes; i++) {
        free(output_feats[i]);
      }
      output_dim = new_output_dim;
      for (int i = 0; i < num_nodes; i++) {
        output_feats[i] = (float *) calloc(sizeof(float), output_dim);
      }
    }

    ~Nodes() {
      for (int i = 0; i < num_nodes; i++) {
        free(input_feats[i]);
      }
      free(input_feats);
      for (int i = 0; i < num_nodes; i++) {
        free(output_feats[i]);
      }
      free(output_feats);
    }
};

#endif //PROJ_NODE_H
