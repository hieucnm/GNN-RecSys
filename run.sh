# train
PYTHONPATH=/data/zmining/jupyter-notebook/hieucnm/graph/v1__no_user_network/scripts/GNN-ResSys  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="3,4" python example_main_train.py -ip ../../data/training_data/interactions_d90/2021/07/04 -up ../../data/training_data/preprocessed/2021/07/04/user_features.csv -rp ../../outputs --num-epochs 15

PYTHONPATH=/data/zmining/jupyter-notebook/hieucnm/graph/v1__no_user_network/scripts/GNN-ResSys  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="3,4" python example_main_train.py -ip ../../data/training_data/interactions_d90/2021/07/04 -up ../../data/training_data/preprocessed/2021/07/04/user_features.csv -rp ../../outputs --num-epochs 100 --num-neighbors 512
