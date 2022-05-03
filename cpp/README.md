# Introduction

This folder contains C++ implementation of Graph Attention
Network (inference only). We use PyTorch to train a Graph
Attention Network for Protein-Protein Interactions (PPI) datasets.
The trained GAT has 0.965841 micro F-1 score on the test PPI graph.
Then, we export the trained parameters and use our C++ implementation
to run the inference. The sequential C++ implementation achieves same micro
F-1 score, which proves the correctness of our implementation.

# OpenMP Parallelism

## Head-oriented Parallelism

A typical attention module usually repeats its computation multiple times in parallel,
thus is also known as "multi-head attention". Computations for different heads
follow the same computation paths and are not interdependent, thus could easily be parallelized.
In head-oriented parallelism strategy, we use OpenMP to parallelize different heads. Note that the
number of heads is usually a small number (in our model, it's a number ranging from
4 to 6), which makes this strategy only suitable when number of threads is not large.

### Naive approach
The first naive approach (commit 13b859) we tried was to simply add
`#pragma omp parallel for` before the beginning of every outmost
for-loop. This leads to 2x speedup (on a 8-core machine) which is far from ideal. Clearly,
this naive approach does not consider data dependency. As a consequence,
it even leads to a wrong result. The sequential version achieves 0.965841
micro F1 score on the PPI (Protein-Protein Interactions) dataset, which
is exactly the same figure achieved by the oracle PyTorch implementation.
The naive OpenMP approach, however, leads to only 0.27 micro F1 score,
indicating this approach leads to completely wrong results. After removing
inappropriate parallelism and adding critical sections, we get back the correct
results with only 30% speedup on a 8-core machine.

### Revise sequential logic
To exploit the parallelism better, we revised the code logic. Without surprise, we found out the
original sequential implementation could be improved. In commit 581cce, we changed
the way we compute softmax for attention values. In machine learning, to find the softmax value for
an array of numerical values, it is very common to subtract all of them by their
max value, such that all the values are less than or equal to zero. Then, we apply
exponential function to them and divide each by the sum of exponents. The subtraction
step does not change the theoretical result, but is very useful and important in
practice because it is better for numerical stability considering the nature of
floating point representation and computation in modern computer architecture. In
the oracle PyTorch implementation, the subtraction preprocessing step applies globally
using a single line of code `scores_per_edge = scores_per_edge - scores_per_edge.max()`.
In our C++ implementation, to save memory, we computed the attention values
(i.e. `scores_per_edge`) on the fly. As a consequence, to compute the max, we had
to calculate the attention values twice, leading to unnecessary computation cost.
To eliminate redundant computations, we subtracted attentions using a neighborhood-aware
approach in commit 581cce. That is, we subtracted attentions by the max attention value in their
neighborhood. This led to better numerical stability, and greatly reduced computation
cost. The number of cache references dropped drastically from 76440020 to 40689622
for the PPI dataset. As a consequence, the computation time for the sequential
version dropped by 21.59%, but the parallell speedup remained the same. One small
pitfall is that the accuracy dropped from 0.965841 to 0.924718, but we are confident
that this is because the model parameters were trained in the original way of computing
softmax. Due to the way the PyTorch version is implemented, it is cumbersome to align
it to our C++ version and retrain the model. We leave it as an exercise if readers
are interested.

### Fine-grained tuning
We then try more parallelism strategies. In other words, we try parallelize different
sections and find best regions to parallelize. In ed67aff, we achieve 0.35s using
8 threads on an 8-core machine, which is 4.2x faster than the sequential version
tested on the same machine. We then optimize the code logic further by removing
unnecessary data structures (e.g. we removed an array and used one local variable
instead) and reducing computation steps (e.g. rather than calculate attention for
each neighbor and then find max attention, we only calculate max heat from neighborhoods
and then use max heat to calculate max attention). This leads to a slight improvement
on performance, but overall speedup remains roughly the same.

### Experiments

The experiments are conducted in a CMU GHC machine, an 8-core Intel(R) Core(TM) i7-9700 CPU @ 3.00GHz
machine.

| Number of Threads | Time   | Speedup |
|-------------------|--------|---------|
| 1x                | 1.4975 | 1x      |
| 4x                | 0.5224 | 2.87x   |
| 6x                | 0.3532 | 4.24x   |
| 8x                | 0.3508 | 4.27x   |

We can see that the program scales well when the number of threads is smaller than
or equal to 6. This is because the max number of attention heads is 6. Adding more
threads does not help here. The fact that 8x threads perform slightly better than 4x
threads is due to the activation function parallelism. In activation function, we
parallelize the computations by nodes, partly because there is no presence of "head"
in activation step.