# Introduction

This folder contains a Java implementation of Graph Attention Network (inference only), leveraging [Apache TinkerPop](http://tinkerpop.apache.org/)'s vertex program framework. The trained network parameters come from a PyTorch implementation and we have verified that our vertex program achieves same accuracy compared to the inference conducted via PyTorch.

The nature of vertex program treats every vertex as an individual entity of the entire graph. The program can be executed in a distributed manner and scale well. Apache TinkerPop, as the most popular graph computing framework, supports different execution backends for vertex programs.

# Usage

Open TinkerPop's gremlin console, and run the following code:

```groovy
graph = TinkerGraph.open()
g = graph.traversal()
g.io("/path/to/ppi-v3d0.kryo").read().iterate()
graph = g.getGraph()
import org.apache.tinkerpop.gremlin.process.computer.neuralnetwork.GraphAttentionNetworkVertexProgram
result = graph.compute().program(GraphAttentionNetworkVertexProgram.build().create()).submit().get().graph()
```

The above code snippet runs the vertex program locally with multi-threading. To run it on a Spark cluster, follow the TinkerPop official guidelines, configure your Spark environment and submit the job using `SparkGraphComputer`:

```groovy
result = graph.compute(SparkGraphComputer).program(GraphAttentionNetworkVertexProgram.build().create()).submit().get().graph()
```

In both cases, the results are the same. In other words, the underlying execution engine does not impact the result but the latency.

# Performance

Running the vertex program is significantly slower than running the sequential version due to a couple of facts. First, running a vertex program has additional overheads including graph partitioniong, task allocation, message passing. Second, a vertex program uses Bulk Synchronous Parallel (BSP) model, meaning that all computations must finish before next iteration can start. Third, the dataset is too small with only 3k nodes and 100k edges.

The major benefits of vertex programs include:
1. No need to dump graph data for neural network computation. Users can run GAT inferences on any TinkerPop-enabled graph database directly, as opposed to dumping graph data to somewhere else for computation.
2. Scalability. With a huge graph, the vertex program can be run on many computation nodes.

We don't have enough time to create a gigantic graph and benchmark the GAT computation on a Spark cluster, but it is certainly possible and straight-forward if you are familiar with TinkerPop and Spark.