# Parallel GAT

## Authors

Xinyue Chen, Boxuan Li

## Summary

In this project, we implemented parallel Graph Attention Networks (GATs), a variant of Graph Convolutional Networks (GCNs), using methods including OpenMP, CUDA, and Graph Vertex Program. We benchmarked their performances using graphs of different sizes and connectivity. We only implemented the forward pass (inference) of the algorithm.

## Background

Graph neural networks, as a kind of modelling for structured data, have been brought to people's attention in recent years. Popular applications include node classification, and graph classification. GNNs are also integrated in other downstream tasks, such as named entity recognition, visual question answering, to help understand structured information inherent to the tasks.

The idea of GNNs is basically as follows. There is a feature tensor for each node, and each layer $k$ of GCN has a learnable matrix parameter $W_k$.  Each node calculates its outgoing message by applying linear transformation to its feature tensor $\{h_{k-1}\}$ with $W_k$ and pass the message to all neighbors. Then each node aggregates the incoming messages and uses them to update its own feature tensor. This completes the message passing of a single layer. And the updated feature tensors $\{h_k\}$ are ready for use as inputs to the next layer. 

In GCNs, the aggregation is instantiated as average pooling over the incoming messages. GATs further incorporate attention mechanism for aggregation. Specifically, we introduce a weight for each incoming message (from node $j$) specific to the receiving node $i$, by calculating the affinity between $W_{k}h_{k-1}^{i}$ and $W_{k}h_{k-1}^{j}$. The weights are then normalized as the coefficients for pooling.

We could also do a step further and introduce multi-head attention mechanism, where within each layer we use multiple $W_{k}$ and process nodes with different $W_{k}$ in parallel. Then we concatenate the resultant updated feature tensors as the canonical outputs of this layer.

## Results

Please see the [final report](./report.pdf).