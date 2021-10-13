import argparse
import datetime as dt
import sys
import warnings

import pandas as pd

warnings.filterwarnings('ignore')
sys.path.append('/data/zmining/jupyter-hub/hieucnm')
from common.spark_processing import init_spark

sys.path.append('/data/zmining/jupyter-hub/hieucnm')
HDFS_PROJECT_DIR = '/data/jobs/rnd/development/hieucnm/graph/v5__with_edge_features'
DATA_NAMES = ['user_embeddings', 'item_embeddings', 'scores']

parser = argparse.ArgumentParser("Graph Deployment")
parser.add_argument('--hdfs-dir', type=str, help='', default=HDFS_PROJECT_DIR + '/outputs/predict/{}/%Y/%m/%d')
parser.add_argument('--local-dir', type=str, help='',
                    default='/data/zmining/hieucnm/graph/deploy_local/outputs/predict/{}/%Y/%m/%d')
parser.add_argument('--duration', type=int, help='', default=7)
parser.add_argument('--weekday', type=int, help='', default=5)
parser.add_argument('--date', type=str, help='', default='2021-09-11')


# ==============
# Main stage ===

def hdfs_path_exists(spark, path):
    try:
        _ = spark.read.parquet(path)
        return True
    except Exception as e:
        print(e)
        return False


def put_data(spark):
    for data_name in DATA_NAMES:
        df = spark.read.parquet(local_dir.format(data_name))
        if "__index_level_0__" in df.columns:
            df = df.drop("__index_level_0__")
        df.write.parquet(hdfs_dir.format(data_name))
        print(f'--> saved {hdfs_dir.format(data_name)}')


def main():
    spark = None
    try:
        spark = init_spark(n_ram=64)
        if hdfs_path_exists(spark, path=hdfs_dir.format(DATA_NAMES[0])):
            print('Output existed! Finish!')
            spark.stop()
            return

        start_time = dt.datetime.now().replace(microsecond=0)
        print('Putting data from 9.17 to hdfs ...')
        put_data(spark)
        print(f'Finish! Elapsed time: {dt.datetime.now().replace(microsecond=0) - start_time}')
        spark.stop()
    except Exception as e:
        if spark is not None:
            spark.stop()
        print(e)


if __name__ == '__main__':
    args = parser.parse_args()
    date = dt.datetime.today() if args.date == '' else dt.datetime.strptime(args.date, "%Y-%m-%d")
    date = [d for d in pd.date_range(end=date, periods=args.duration) if d.weekday() == args.weekday][0]
    print(f'On date: {date}')

    local_dir = 'file://' + date.strftime(args.local_dir)
    hdfs_dir = date.strftime(args.hdfs_dir)
    main()
