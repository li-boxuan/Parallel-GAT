# Parallel GAT

## Authors

Xinyue Chen, Boxuan Li

## Summary

In this project, we plan to implement parallel Graph Attention Networks (GATs), a variant of Graph Convolutional Networks (GCNs), using methods including OpenMP and CUDA. We shall benchmark their performances using graphs of different sizes and connectivity. We will only implement the forward pass of the algorithm



## Background

Graph neural networks, as a kind of modelling for structured data, have been brought to people's attention in recent years. Popular applications include node classification, and graph classification. GNNs are also integrated in other downstream tasks, such as named entity recognition, visual question answering, to help understand structured information inherent to the tasks.

The idea of GNNs is basically as follows. There is a feature tensor for each node, and each layer $k$ of GCN has a learnable matrix parameter $W_k$.  Each node calculates its outgoing message by applying linear transformation to its feature tensor $\{h_{k-1}\}$ with $W_k$ and pass the message to all neighbors. Then each node aggregates the incoming messages and uses them to update its own feature tensor. This completes the message passing of a single layer. And the updated feature tensors $\{h_k\}$ are ready for use as inputs to the next layer. 

In GCNs, the aggregation is instantiated as average pooling over the incoming messages. GATs further incorporate attention mechanism for aggregation. Specifically, we introduce a weight for each incoming message (from node $j$) specific to the receiving node $i$, by calculating the affinity between $W_{k}h_{k-1}^{i}$ and $W_{k}h_{k-1}^{j}$. The weights are then normalized as the coefficients for pooling.

We could also do a step further and introduce multi-head attention mechanism, where within each layer we use multiple $W_{k}$ and process nodes with different $W_{k}$ in parallel. Then we concatenate the resultant updated feature tensors as the canonical outputs of this layer.



## Challenges

#### Matrix multiplication

This is the bottleneck for the computation of GNNs. There exist several tactics of optimize matrix multiplication optimization in the context of parallelism and we are interested to explore them and design an efficient approach that caters for our graphs of interest

#### The impact of graph structure upon performance

We speculate that the connectivity of the input graphs and the size of the graph will have great impact on the performance of the parallelized GATs. We shall examine these nuances and compare them in OpenMP implementation and CUDA implementation. Better still, we shall cope with potential problems using different designs.

### Parallelism over nodes and attention heads

There are a few parallelism dimensions that we could explore. We use parallelism in matrix multiplication, and we could also parallelize over all nodes within the same layer, and over multiple attention heads. 



## Resources

We use CPUs for OpenMP implementation and GPUs for CUDA implementation. 

For datasets, we intend to generate graphs of different connectivity and sizes for the initial experiments. Then, we may include popular datasets for GNNs benchmarking, such as Cora, Citeseer and PubMed.



## Goals and Deliverables

+ $75\%$ Goal
  + Implement serialized GATs in C++.
  + Implement one parallelized version using CUDA or OpenMP.
+ $100\%$ Goal
  + Implement both parallelized version.
  + Optimize the performances.
  + Compare performances with at least two datasets.
+ $125\%$ Goal
  + Use MPI for distributed computation, or
  + Implement backward pass.



## Schedule Milestones

Mar 23: Finish project proposal

Apr 11: [Checkpoint] Finish serialized GAT in C++

Apr 18:  Finish one parallelized version using OpenMP

Apr 25: Finish another parallelized version using CUDA and conduct performance comparison

Apr 29: Finish report

May 5: Presentation



## Milestone

We have completed the pipeline for serialized version of GAT in C++. The current version supports random initialization of model parameters, including weights of the model the the input features of the node. It also support variable number of attention heads and hidden sizes. 

We believe we will be able to finish at least one parallelized version using OpenMP. Most of the work will be focused on the optimization of matrix multiplication and performance comparison.  

For the poster session, we plan to present the performance comparison of different parallelization versions and the the serialized version, with input graphs with different scales and connectivities. To hit this, we first need to ensure the our model is logically correct by comparing the outputs with the outputs of an oracle model. Then, we need to implement parallelized versions and optimize the message passing mechanism to increase cache utility and improve computation efficiency catering to the mechanism of the parallelization method of interest.

The potential concerns may include: 1) how the memory size of our computing machine would limit the scale of the input graphs, 2) to what extent could we improve the efficiency using OpenMP and possibly CUDA.
