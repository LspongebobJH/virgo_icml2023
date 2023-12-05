# virgo
The codes for the ICML 2023 paper [On the Initialization of Graph Neural Networks](https://proceedings.mlr.press/v202/li23y.html)

## Requirements
DGL>=0.8.0, PyTorch>=1.13.1, PyG>=2.0.0, OGB>=1.3.6

## Usage
Simply running with `python [file_name].py`. Please refer to the appendix of the paper for details of hyperparameter tunning.

## List of files in the main directory
* utils.py and utils_pyg.py: provide **implementations of virgo initializations**. Specifically, `init_layers` function initializes each layer's learnable weights
with the virgo. Refer to files below for how to use this function.

> Below `nc`, `lp` and `gc` means `node classification`, `link prediction` and `graph classification` tasks, respectively. 
> comp* : `dataset, ..` + `model, ..` + `task, ..`. Meaning that experimenting the `models, ..` on datasets `dataset, ..` with tasks `task, ..`
* comp1: cora, pubmed, citeseer, reddit + gcn + nc
* comp2: arxiv + gcn + nc
* comp3: proteins + gcn + nc
* comp4: products + sage + nc, mini batch training
* comp5: collab + gcn + lp
* comp6: ppa + gcn + lp
* comp7: ddi + gcn + lp
* comp8: citation + cluster gcn + lp
* comp9: molhiv/mopcba + gcn/gin + gc
* comp10: ppa + gcn/gin + gc
* comp11: TU Dataset(imdb, collab) + gcn/gin + gc

## List of files in tests/
Running these files to obtain the empirical results in the paper's appendix. These results are evidences for theoretically analysis.

* p_sigma_l.py: to estimate the probability of $\sigma_l$, which should be approximately equal to $\sigma^l$, where the latter one is approximately equal to 0.5, thus $\sigma_l$ is $0.5^l$.
* corr_sigma_prod_w: to estimate the correlation between $\sigma$ and $\prod w$, $\sigma^2$ and $\prod w^2$, which should be approximately 0.
* corr_paths: to estimate the correlation between different message propagation paths.
* corr_prod_w: to estimate the correlation between different weight propagation paths.
