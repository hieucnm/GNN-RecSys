#!/bin/zsh

DES_PATH="/data/zmining/jupyter-notebook/hieucnm/graph/v1__no_user_network/scripts"
MACHINE=zdeploy@10.50.9.13

echo "---> rsync  ..."
rsync -zavh --exclude '__pycache__' --rsync-path="mkdir -p $DES_PATH && rsync" ./* $MACHINE:$DES_PATH
echo "---> rsync done!"
