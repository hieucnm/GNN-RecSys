# train
PYTHONPATH=/data/zmining/jupyter-notebook/hieucnm/graph/v1__no_user_network/scripts/GNN-ResSys  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="3,4" python example_main_train.py -ip ../../data/training_data/interactions_d90/2021/07/04/part-00000-52f979d1-b879-465f-b00c-a85a5437f3ef-c000.snappy.parquet -up ../../data/training_data/preprocessed/2021/07/04/user_features.csv -rp ../../outputs/