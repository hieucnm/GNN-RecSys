#!/bin/zsh

DES_PATH="/data/zmining/jupyter-notebook/hieucnm/graph/v3__with_user_group/scripts/GNN-ResSys"
MACHINE=zdeploy@10.50.9.13

echo "---> rsync  ..."
rsync -zavh --exclude '__pycache__' --exclude 'examples' --rsync-path="mkdir -p $DES_PATH && rsync" ./* $MACHINE:$DES_PATH
echo "---> rsync done!"
