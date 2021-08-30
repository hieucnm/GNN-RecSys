# train
PYTHONPATH=/data/zmining/jupyter-notebook/hieucnm/graph/v1__no_user_network/scripts/GNN-ResSys  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="3,4" python example_main_train.py -ip ../../data/training_data/interactions_d90/2021/07/04 -up ../../data/training_data/preprocessed/2021/07/04/user_features.csv -rp ../../outputs

PYTHONPATH=/data/zmining/jupyter-notebook/hieucnm/graph/v1__no_user_network/scripts/GNN-ResSys  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="3,4" python example_main_train.py -ip ../../data/training_data/interactions_d90/2021/07/04 -up ../../data/training_data/preprocessed/2021/07/04/user_features.csv -rp ../../outputs

PYTHONPATH=/data/zmining/jupyter-notebook/hieucnm/graph/v3__with_user_group/scripts/GNN-ResSys  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="3,4" python main_train.py --train-dirs ../../data/training_data/preprocessed/2021/06/26/ --valid-dirs ../../data/training_data/preprocessed/2021/07/03/  --result-dir ../../outputs/train

PYTHONPATH=/data/zmining/jupyter-notebook/hieucnm/graph/v3__with_user_group/scripts/GNN-ResSys  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="3,4" python main_train.py --train-dirs ../../data/training_data/preprocessed/2021/06/26,../../data/training_data/preprocessed/2021/06/19,../../data/training_data/preprocessed/2021/06/12,../../data/training_data/preprocessed/2021/06/05 --valid-dirs ../../data/training_data/preprocessed/2021/07/03/  --result-dir ../../outputs/train

PYTHONPATH=/data/zmining/jupyter-notebook/hieucnm/graph/v3__with_user_group/scripts/GNN-ResSys  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="3" python main_infer.py --param-path ../../outputs/train/20210830_160535/metadata/arguments.json --iid-map-path ../../outputs/train/20210830_160535/metadata/train_iid_map_df.csv --model-path ../../outputs/train/20210830_160535/model_ep_12.pth --item-embed-path ../../outputs/train/20210830_160535/metadata/item_embeddings.npy  --data-dir ../../data/inference_data/preprocessed/2021/07/10  --result-dir ../../outputs/infer
