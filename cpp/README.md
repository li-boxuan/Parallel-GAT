# Introduction

This folder contains C++ implementation of Graph Attention
Network (inference only).

# OpenMP Parallelism

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
softmax. Due to the way the oracle version is implemented, it is cumbersome to align
it to our C++ version and retrain the model. We leave it as an exercise if readers
are interested.