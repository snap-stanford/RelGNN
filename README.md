# RelGNN: Composite Message Passing for Relational Deep Learning

[ICML 2025] [RelGNN: Composite Message Passing for Relational Deep Learning](https://arxiv.org/abs/2502.06784)

> Tianlang Chen, Charilaos Kanatsoulis, Jure Leskovec

## Overview

Predictive tasks on relational databases are critical in real-world applications spanning e-commerce, healthcare, and social media. To address these tasks effectively, [Relational Deep Learning (RDL)](https://arxiv.org/abs/2312.04615) encodes relational data as graphs, enabling Graph Neural Networks (GNNs) to exploit relational structures for improved predictions. However, existing RDL methods often overlook the intrinsic structural properties of the graphs built from relational databases, leading to modeling inefficiencies, particularly in handling many-to-many relationships. Here we introduce RelGNN, a novel GNN framework specifically designed to leverage the unique structural characteristics of the graphs built from relational databases. At the core of our approach is the introduction of atomic routes, which are simple paths that enable direct single-hop interactions between the source and destination nodes. Building upon these atomic routes, RelGNN designs new composite message passing and graph attention mechanisms that reduce redundancy, highlight key signals, and enhance predictive accuracy. RelGNN is evaluated on 30 diverse real-world tasks from [Relbench](https://arxiv.org/abs/2407.20060), and achieves state-of-the-art performance on the vast majority of tasks, with improvements of up to 25%.

