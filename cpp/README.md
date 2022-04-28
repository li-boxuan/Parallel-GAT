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
