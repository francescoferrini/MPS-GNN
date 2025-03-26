# MPS-GNN

Code for the model Meta-path Statistics Graph Neural Network
# Databases

For this project 3 relational databases have been employed:
- [EICU](https://eicu-crd.mit.edu)
- [MONDIAL](https://relational-data.org/dataset/Mondial)
- [ErgastF1](https://relational-data.org/dataset/ErgastF)

To download the preprocessed data, please retrieve the files from [google drive](https://drive.google.com/drive/folders/1S2-uhaa04gICvQodTp0tVu7Ns-lAYUb7?usp=share_link) and place them in the `../GNN-3DDE/data/` folder.

## Run

For running the code install the requirements in `requirements.txt`, then chose the dataset inside the file `run.sh` and run

```sh
bash run.sh
```
where in the file you can specify:
- **dataset**
- **hidden_dim:** hidden embedding dimension for the model
- **pre_trained:** = True if you want to try the pre_trained models

## 📖 Cite

```
@article{
ferrini2025a,
title={A Self-Explainable Heterogeneous {GNN} for Relational Deep Learning},
author={Francesco Ferrini and Antonio Longa and Andrea Passerini and Manfred Jaeger},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=8Q4qxe9a9Z},
note={}
}
```

