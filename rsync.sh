#!/bin/zsh

DES_PATH_1="/data/zmining/jupyter-notebook/hieucnm/graph/v5__with_edge_features/scripts/GNN-ResSys"
MACHINE_1=zdeploy@10.50.9.13

DES_PATH_2=DES_PATH_1
MACHINE_2=zdeploy@10.50.9.17


echo "---> rsync  ..."
rsync -zavh --exclude '__pycache__' --exclude 'base' --rsync-path="mkdir -p DES_PATH_1 && rsync" ./* $MACHINE_1:$DES_PATH_1
# rsync -zavh --exclude '__pycache__' --exclude 'examples' --rsync-path="mkdir -p $DES_PATH_2 && rsync" ./* $MACHINE_2:$DES_PATH_2
echo "---> rsync done!"
