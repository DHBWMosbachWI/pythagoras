import os
from tqdm import tqdm
from os.path import join
import pyarrow.parquet as pq
import random
import pandas as pd
import numpy as np
import json
import sys
sys.path.append("../..")
from column_annotation_gnn.data_loader.GitTables_data_loader import get_LabelEncoder

label_enc = get_LabelEncoder()
 
# Generating training data for gpt 
# Overall sport datasets
# Now in the form of one row belongs to one column => transposed table
# Now in the form that one prompt/completion pair belongs to one table-column

shuffle_cols = True
random_state = 1
split = "test"
number_of_rows_per_table = 10
end_of_prompt = "\n\n###\n\n"

for random_state in range(1,2):
    training_data_prompts = []
    all_class_labels = []
    with open(join(os.environ["GitTables"], "data", f"train_valid_test_split_{random_state}_0.7.json")) as f:
        train_valid_test_split = json.load(f)
                
    for idx_table_path, table_path in tqdm(enumerate(train_valid_test_split[split]), total=len(train_valid_test_split[split])):
        # if idx_table_path > 0:
        #     continue
        
        # read metadata
        table_metadata = json.loads(pq.read_schema(join(os.environ["GitTables"], table_path)).metadata[b"gittables"])
        dbpedia_types = table_metadata["dbpedia_embedding_column_types"]
        dbpedia_similarities = table_metadata["dbpedia_embedding_similarities"]

        # read the table in a DF
        df = pd.read_parquet(join(os.environ["GitTables"], table_path))
            
        valid_cols = []
        valid_labels = []
        if shuffle_cols:
            column_list = list(range(len(df.columns)))
            #random.seed(self.random_state)
            random.shuffle(column_list)
        else:
            column_list = list(range(len(df.columns)))
            
        for i in column_list:
            column_name = df.columns[i]
            
            try:
                # check if the semantic type of the columns is in the valid semantic types that we consider in our experiments 
                if len(df[column_name].dropna()) > 0:
                    if dbpedia_similarities[column_name] >= 0.7:
                        if table_metadata["dtypes"][column_name] == "object" or table_metadata["dtypes"][column_name] == "string":
                            if (dbpedia_types[column_name]["cleaned_label"]+"_tt" in label_enc.classes_):
                                column_data_type = 0 # => "textual"
                                column_label = dbpedia_types[column_name]["cleaned_label"]+"_tt"
                            else:
                                 continue
                        else:
                            if (dbpedia_types[column_name]["cleaned_label"]+"_nt" in label_enc.classes_):
                                column_data_type = 1 # => "numerical"
                                column_label = dbpedia_types[column_name]["cleaned_label"]+"_nt"
                            else:
                                continue
                    else:
                        continue
                else:
                    continue
                
            except Exception as e:
                continue
                #print(e)
                #print(f"Not considering column: {column_name} from table: {table_path}")
            
            valid_cols.append(column_name)
            valid_labels.append(column_label)
        
        if len(valid_cols) == 0:
            continue
        if len(df) >= number_of_rows_per_table:
            df_result = df[valid_cols][:number_of_rows_per_table]
        else:
            df_result = df[valid_cols]
        if len(df_result) == 0:
            #print(f"Table {table_name} has no columns with assigned semantic types!")
            continue
        try:
            for idx, column in enumerate(df_result.columns):
                training_data = {}
                training_data["prompt"] = df_result[column].to_csv(header=False, index=False)+end_of_prompt
                training_data["completion"] = " "+str(label_enc.transform([valid_labels[idx]])[0])
                training_data_prompts.append(training_data)
                all_class_labels.append(valid_labels[idx])
        except Exception as e:
            print(e)
            print(valid_labels)
                
    with open(f"./training_data/GitTables_{split}_{shuffle_cols}_{random_state}_0.7_columnwise.jsonl", "w") as f:
        for entry in training_data_prompts:
            json.dump(entry, f)
            f.write("\n")
    print(f"GitTables_{split}_{shuffle_cols}_{random_state}_0.7.jsonl Number of unique classes: {len(np.unique(all_class_labels))}")