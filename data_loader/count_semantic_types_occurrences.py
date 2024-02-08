import json
from os.path import join
from dotenv import load_dotenv
load_dotenv(override=True)
import pandas as pd
from collections import Counter

sport_domains = ["baseball", "basketball", "football", "hockey", "soccer"]
random_state = 1

semantic_types = []

for idx_sport_domain, sport_domain in enumerate(sport_domains):
    # if idx_sport_domain > 0:
    #     break
    # load metadata.json containing semantic types of the columns
    with open(join(os.environ["SportsTables"], sport_domain, "metadata.json")) as f:
        metadata = json.load(f)
    with open(join(os.environ["SportsTables"], sport_domain, f"train_valid_test_split_{random_state}.json")) as f:
        train_valid_test_split = json.load(f)


    for idx_table_path, table_name_full in enumerate(train_valid_test_split["train"]+train_valid_test_split["valid"]+train_valid_test_split["test"]):
        # if idx_table_path != 1:
        #     continue
        table_name = table_name_full.split("/")[-1].split(".csv")[0]
        ## search for correct in key in metadata
        table_metadata_key = None
        for key in metadata.keys():
            if key in table_name:
                table_metadata_key = key
        if table_metadata_key == None:
            print(f"CSV {table_name_full} not in metadata.json defined!")
            continue

        df = pd.read_csv(join(os.environ["SportsTables"], sport_domain, table_name_full))
        column_list = list(range(len(df.columns)))
            
        for i in column_list:
            column_name = df.columns[i]
            # search for defined columns data type and semantic label in metadata
            if column_name in metadata[table_metadata_key]["textual_cols"].keys():
                column_data_type = "textual"
                column_label = metadata[table_metadata_key]["textual_cols"][column_name]
            elif column_name in metadata[table_metadata_key]["numerical_cols"].keys():
                column_data_type = "numerical"
                column_label = metadata[table_metadata_key]["numerical_cols"][column_name]
            else:
                print(f"Column {df.columns[i]} in {table_name} not labeled in metadata.json!")
                continue
            semantic_types.append(column_label)

df_sem_types = pd.DataFrame(Counter(semantic_types), index=["count"]).T.reset_index().sort_values(by="count", ascending=False).reset_index(drop=True).rename(columns={"index":"semantic_type"})
df_sem_types.to_csv("SportsTables_semantic_type_occurrences.csv", index=False)