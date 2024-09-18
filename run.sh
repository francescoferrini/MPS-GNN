#!/bin/bash

dataset="eicu" 
folder="/../MPS-GNN/"
pre_trained="True"

if [ "${dataset}" == "mondial" ]; then
    hidden_dim=16
elif [ "${dataset}" == "eicu" ]; then
    hidden_dim=32
elif [ "${dataset}" == "ergastf1" ]; then
    hidden_dim=64
fi
python main.py --hidden_dim "${hidden_dim}" --dataset "${dataset}" --folder "${folder}" --pre_trained "${pre_trained}"