import random
import numpy as np
import json
import pandas as pd
from glob import glob
import os
from os.path import join
from dotenv import load_dotenv
load_dotenv(override=True)
from tqdm import tqdm
import argparse

sport_domains = ["baseball", "basketball", "football", "hockey", "soccer"]

NUMERIC_COLUMNS_THRH = 0.5

'''
'''
def infer_numeric_threshold(column, threshold):
    numeric_count = pd.to_numeric(column, errors='coerce').notna().sum()
    total_count = column.size
    if (numeric_count / total_count) >= threshold:
        return pd.to_numeric(column, errors='coerce')
    else:
        return column
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--data_corpus",
        default="SportsTables",
        choices=["SportsTables", "GitTables"]
    )
    
    parser.add_argument(
        "--sem_type_sim_score",
        default=0.7
    )
    
    
    args = parser.parse_args()
    print("args={}".format(json.dumps(vars(args))))
    
    # TODO: Improve code and eliminate redundacies in calculating the numeric features
    if args.data_corpus == "GitTables":
        result_data = []
        ## load all valid tables from GitTables corpus that we consider in our experiments
        with open(join(os.environ[args.data_corpus], "data", f"train_valid_test_split_1_{args.sem_type_sim_score}.json")) as f:
            train_valid_test_split = json.load(f)
            all_valid_tables = train_valid_test_split["train"] + train_valid_test_split["valid"] + train_valid_test_split["test"] 

        for idx_table_path, table_path in tqdm(enumerate(all_valid_tables), total=len(all_valid_tables)):
            # read the table in a DF
            df_table = pd.read_parquet(join(os.environ[args.data_corpus], table_path))
            
            # try to infer all columns of the dataframe to a numeric column with a threshold of numeric values 
            # that must be parsable as numeric object
            df_table = df_table.apply(lambda col: infer_numeric_threshold(col, NUMERIC_COLUMNS_THRH))
            
            # Calculate numerical statistics of the columns
            df_stats = df_table.describe(
                include="all", percentiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9]).T
            data = []
            for column in df_stats.index:
                try:
                    data.append(df_table[column].kurtosis())
                except:
                    data.append(np.NaN)
            df_stats["kurtosis"] = data
            data = []
            for column in df_stats.index:
                try:
                    data.append(df_table[column].skew())
                except Exception as err:
                    data.append(np.NaN)
            df_stats["skewness"] = data

            for id_, row in df_stats.iterrows():
                try:
                    result_data.append([table_path.split("/")[-2], table_path.split("/")[-1], id_, row[["mean", "std", "min", "10%", "20%",
                                    "30%", "40%", "50%", "60%", "70%", "80%", "90%", "max", "kurtosis", "skewness"]].values.tolist()])
                except:
                    continue
                
        df_result = pd.DataFrame(result_data, columns=["domain", "table_name", "column_name", "features"])
        df_result.to_csv(join(os.environ["MAIN_DIR"],"data_loader","features",f"GitTables_numerical_statistic_features_{args.sem_type_sim_score}.csv"), index=False)

    if args.data_corpus == "SportsTables":
        result_data = []
        for idx_sport_domain, sport_domain in enumerate(sport_domains):
            # if idx_sport_domain > 0:
            #     continue
            for idx_table_path, table_path in tqdm(enumerate(glob(join(os.environ["SportsTables"], sport_domain, "*.csv"))),total=len(glob(join(os.environ["SportsTables"], sport_domain, "*.csv")))):
                # if idx_table_path > 2:
                #     continue
                df_table = pd.read_csv(table_path)

                # try to infer all columns of the dataframe to a numeric column with a threshold of numeric values 
                # that must be parsable as numeric object
                df_table = df_table.apply(lambda col: infer_numeric_threshold(col, NUMERIC_COLUMNS_THRH))

                # Calculate numerical statistics of the columns
                df_stats = df_table.describe(
                    include="all", percentiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9]).T
                data = []
                for column in df_stats.index:
                    try:
                        data.append(df_table[column].kurtosis())
                    except:
                        data.append(np.NaN)
                df_stats["kurtosis"] = data
                data = []
                for column in df_stats.index:
                    try:
                        data.append(df_table[column].skew())
                    except Exception as err:
                        data.append(np.NaN)
                df_stats["skewness"] = data

                for id_, row in df_stats.iterrows():
                    result_data.append([sport_domain, table_path.split("/")[-1], id_, row[["mean", "std", "min", "10%", "20%",
                                    "30%", "40%", "50%", "60%", "70%", "80%", "90%", "max", "kurtosis", "skewness"]].values.tolist()])
                    
        df_result = pd.DataFrame(result_data, columns=["domain", "table_name", "column_name", "features"])
        df_result.to_csv(join(os.environ["MAIN_DIR"],"data_loader","features","SportsTables_numerical_statistic_features.csv"), index=False)