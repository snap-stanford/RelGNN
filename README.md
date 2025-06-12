# RelGNN: Composite Message Passing for Relational Deep Learning

[ICML 2025] [RelGNN: Composite Message Passing for Relational Deep Learning](https://arxiv.org/abs/2502.06784)

Tianlang Chen, Charilaos Kanatsoulis, Jure Leskovec

> Contact: tlchen@cs.stanford.edu

## Overview

Predictive tasks on relational databases are critical in real-world applications. To address these tasks effectively, [Relational Deep Learning (RDL)](https://arxiv.org/abs/2312.04615) encodes relational data as graphs, enabling GNNs to exploit relational structures for improved predictions. However, existing RDL methods often overlook the intrinsic structural properties of the graphs built from relational databases, leading to modeling inefficiencies. 

Here we introduce RelGNN, a novel GNN framework specifically designed to leverage the unique structural characteristics of the graphs built from relational databases. We first introduce *atomic routes*, which are simple paths that enable direct single-hop interactions between the source and destination nodes. Building upon these atomic routes, RelGNN designs new composite message passing and graph attention mechanisms to enhance predictive accuracy. RelGNN is evaluated on 30 diverse real-world tasks across 7 datasets from [Relbench](https://arxiv.org/abs/2407.20060), achieving state-of-the-art performance on the vast majority of tasks.

## Installation

To install the required dependencies:

```bash
conda env create -n relgnn python=3.10
conda activate relgnn

pip install relbench[full]
pip install pyg-lib -f https://data.pyg.org/whl/torch-<torch_version>+<cuda_version>.html # replace <torch_version> and <cuda_version> with your environment config
pip install pytorch-frame==0.2.3
pip install sentence-transformers==3.3.1
pip install scikit-learn==1.5.2
```

## Usage

To run the main script and reproduce the main results:

```bash
cd examples/

# For entity level tasks (entity classification and entity regression)
python relgnn_task_node.py --dataset rel-amazon --task user-churn

# For recommendation tasks that use two-tower head (tasks in rel-amazon dataset)
python relgnn_task_link_2tower.py --dataset rel-amazon --task user-item-purchase

# For recommendation tasks that use IDGNN head (tasks in other dataset)
python relgnn_task_link_idgnn.py --dataset rel-hm --task user-item-purchase
```

## Acknowledgements

This codebase builds upon implementations from [Relbench](https://github.com/snap-stanford/relbench). We thank the contributors to open-source libraries such as [PyTorch](https://pytorch.org/), [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric), [PyTorch Frame](https://github.com/pyg-team/pytorch-frame), and others used in this work.

## Citation

If you find this work useful, we would greatly appreciate it if you could cite our work:

```bibtex
@misc{chen2025relgnncompositemessagepassing,
      title={RelGNN: Composite Message Passing for Relational Deep Learning}, 
      author={Tianlang Chen and Charilaos Kanatsoulis and Jure Leskovec},
      year={2025},
      eprint={2502.06784},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.06784}, 
}
```

Please also cite the position and benchmark papers for RDL and Relbench:

```bibtex
@inproceedings{rdl,
  title={Position: Relational Deep Learning - Graph Representation Learning on Relational Databases},
  author={Fey, Matthias and Hu, Weihua and Huang, Kexin and Lenssen, Jan Eric and Ranjan, Rishabh and Robinson, Joshua and Ying, Rex and You, Jiaxuan and Leskovec, Jure},
  booktitle={Forty-first International Conference on Machine Learning}
}
```

```bibtex
@misc{relbench,
      title={RelBench: A Benchmark for Deep Learning on Relational Databases},
      author={Joshua Robinson and Rishabh Ranjan and Weihua Hu and Kexin Huang and Jiaqi Han and Alejandro Dobles and Matthias Fey and Jan E. Lenssen and Yiwen Yuan and Zecheng Zhang and Xinwei He and Jure Leskovec},
      year={2024},
      eprint={2407.20060},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.20060},
}
```