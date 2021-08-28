# train
PYTHONPATH=/data/zmining/jupyter-notebook/hieucnm/graph/v1__no_user_network/scripts/GNN-ResSys  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="3,4" python example_main_train.py -ip ../../data/training_data/interactions_d90/2021/07/04 -up ../../data/training_data/preprocessed/2021/07/04/user_features.csv -rp ../../outputs --num-epochs 15

PYTHONPATH=/data/zmining/jupyter-notebook/hieucnm/graph/v1__no_user_network/scripts/GNN-ResSys  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="3,4" python example_main_train.py -ip ../../data/training_data/interactions_d90/2021/07/04 -up ../../data/training_data/preprocessed/2021/07/04/user_features.csv -rp ../../outputs --num-epochs 100 --num-neighbors 512

PYTHONPATH=/data/zmining/jupyter-notebook/hieucnm/graph/v3__with_user_group/scripts/GNN-ResSys  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="3,4" python custom/main.py --train-dirs ../../data/training_data/preprocessed/2021/06/26/ --valid-dirs ../../data/training_data/preprocessed/2021/07/03/  --result-dir ../../outputs --pred sigmoid --loss bce

PYTHONPATH=/data/zmining/jupyter-notebook/hieucnm/graph/v3__with_user_group/scripts/GNN-ResSys  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="3,4" python custom/main.py --train-dirs ../../data/training_data/preprocessed/2021/06/26,../../data/training_data/preprocessed/2021/06/19,../../data/training_data/preprocessed/2021/06/12,../../data/training_data/preprocessed/2021/06/05 --valid-dirs ../../data/training_data/preprocessed/2021/07/03/  --result-dir ../../outputs --pred sigmoid --loss bce

