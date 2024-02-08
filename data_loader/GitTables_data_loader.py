from sklearn.preprocessing import LabelEncoder
import dgl
from dgl.data import DGLDataset
import json
import pandas as pd
from glob import glob
import transformers
from torch.utils.data import Dataset
import torch
import os
from os.path import join
from dotenv import load_dotenv
load_dotenv(override=True)
from functools import reduce
import operator
import numpy as np
import random
from ast import literal_eval
import pyarrow.parquet as pq
from tqdm import tqdm
from dgl import save_graphs, load_graphs


def get_all_textual_semantic_types():
    with open(join(os.environ["GitTables"], "data", f"valid_semantic_types_10_0.7.json")) as f:
    #with open(join(os.environ["GitTables"], "data", f"valid_semantic_types_10_0.7.json")) as f:
        textual_semantic_types = json.load(f)["textual_types"]
    
    return textual_semantic_types

def get_all_numerical_semantic_types():
    with open(join(os.environ["GitTables"], "data", f"valid_semantic_types_10_0.7.json")) as f:
    #with open(join(os.environ["GitTables"], "data", f"valid_semantic_types_10_0.7.json")) as f:
        numerical_semantic_types = json.load(f)["numerical_types"]
    
    return numerical_semantic_types

def get_LabelEncoder():
    all_semantic_types = get_all_textual_semantic_types() + \
        get_all_numerical_semantic_types()
    label_enc = LabelEncoder()
    label_enc.fit(all_semantic_types)
    return label_enc

def convert_Table_to_dgl_graph_enriched(df: pd.DataFrame, tokenized_table_name, num_features):
    """ This function is to convert the table to a graph representation where each 
     numeric column node is connected to each textual column node. Additionally a numeric node
     is also connected to another node, which represents the special numeric value features of the 
     numerical column (sherlock features). There is also a node which represents the table name. This node is
     connected to all column nodes
    """
    # directed edges from each textual column to each numerical columns
    textual_node_ids = df[df["columns_data_type"] == 0].index.tolist()
    numerical_node_ids = df[df["columns_data_type"] == 1].index.tolist()
    graph_data = {}
    
    # table_name to each column
    table_to_column_source = torch.tensor([0] * len(df))
    table_to_column_target = torch.tensor(df.index.tolist())
    graph_data[('table', 'table_column', 'column')] = (table_to_column_source, table_to_column_target)
    
    # text to num columns
    textual_to_numerical_source = torch.tensor(textual_node_ids).repeat(len(numerical_node_ids)).to(torch.int64)
    textual_to_numerical_target = torch.tensor(numerical_node_ids * len(textual_node_ids)).to(torch.int64)
    
    # add self-loop for column nodes
    textual_to_numerical_source = torch.cat((textual_to_numerical_source, torch.tensor(textual_node_ids + numerical_node_ids))).to(torch.int64)
    textual_to_numerical_target = torch.cat((textual_to_numerical_target, torch.tensor(textual_node_ids + numerical_node_ids))).to(torch.int64)
    
    graph_data[('column', 'column_column', 'column')] = (textual_to_numerical_source, textual_to_numerical_target)
    
    # numerical feature node to numerical column
    # build numerical feature node ids from 0 to len(numerical_node_ids) with the usage of idx
    numerical_feature_node_source = torch.arange(len(numerical_node_ids))
    numerical_feature_node_target = torch.tensor(numerical_node_ids)
    graph_data[('num_feature_node', 'num_feature_column', 'column')] = (numerical_feature_node_source, numerical_feature_node_target)
    

    g = dgl.heterograph(graph_data)
    
    # set features to the nodes
    g.nodes["table"].data["data_tensor"] = torch.tensor([tokenized_table_name])
    g.nodes["column"].data["data_tensor"] = torch.LongTensor(df["data_tensor"].tolist())
    g.nodes["column"].data["label_tensor"] = torch.LongTensor(df["label_tensor"].tolist())
    g.nodes["column"].data["col_type_tensor"] = torch.LongTensor(df["columns_data_type"].tolist())
    g.nodes["num_feature_node"].data["data_tensor"] = torch.tensor(num_features, dtype=torch.float64)
    
    return g


class GitTablesDGLDataset_enriched(DGLDataset):
    """ Dataloader that loads the graph representation with the additional node for numeric
    column that contains a vector of special numeric features (sherlock numeric feature)
    """

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        save_dir:str = join(os.environ["MAIN_DIR"], "data_loader", "tmp"),
        force_reload:bool = False,
        max_length: int = 128,
        device: torch.device = torch.device("cpu"),
        random_state:int = 1,
        split:str = "train",
        shuffle_cols:bool = False,
        semantic_sim_score:float = 0.7
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.random_state = random_state
        self.split = split
        self.shuffle_cols = shuffle_cols
        self.semantic_sim_score = semantic_sim_score
        super(GitTablesDGLDataset_enriched, self).__init__(name="GitTables", save_dir=save_dir, force_reload=force_reload)
        
                 
    def process(self):
        print("Executing dataset generating process...")
        # get the Label Encoder to encode the semantic types 
        label_enc = get_LabelEncoder()
        
        # load sherlock features
        with open(join(os.environ["MAIN_DIR"], "data_loader","valid_sherlock_features.json"), "r") as f:
            selected_sherlock_feature_set = json.load(f)
        print("Loading sherlock feature set...")
        df_sherlock_features = pd.read_csv("/ext/daten-wi/slangenecker/sato/extract/out/features/type_gittables/GitTables_type_gittables_sherlock_features.csv", usecols=["locator", "dataset_id", "field_id", "header", "header_c"]+selected_sherlock_feature_set)
        
        # load additional numerical stats features
        df_num_stats_features = pd.read_csv(join(os.environ["MAIN_DIR"],"data_loader", "features",f"GitTables_numerical_statistic_features_{self.semantic_sim_score}.csv"))
        
        # load all tables from GitTables, tokenize every table column as described in Doduo and build the graph representation
        self.data_list = []
        
        # if idx_sport_domain > 0:
        #     break
        ## load train_valid_test split definitions
        with open(join(os.environ["GitTables"], "data", f"train_valid_test_split_{self.random_state}_{self.semantic_sim_score}.json")) as f:
            train_valid_test_split = json.load(f)

        for idx_table_path, table_path in tqdm(enumerate(train_valid_test_split[self.split]), total=len(train_valid_test_split[self.split])):
            # if idx_table_path > 2:
            #     continue
            #print(table_path)
            
            # read metadata
            table_metadata = json.loads(pq.read_schema(join(os.environ["GitTables"], table_path)).metadata[b"gittables"])
            dbpedia_types = table_metadata["dbpedia_embedding_column_types"]
            dbpedia_similarities = table_metadata["dbpedia_embedding_similarities"]

            # read the table in a DF
            df_table = pd.read_parquet(join(os.environ["GitTables"], table_path))
            
            data_list = []
            
            if self.shuffle_cols:
                column_list = list(range(len(df_table.columns)))
                #random.seed(self.random_state)
                random.shuffle(column_list)
            else:
                column_list = list(range(len(df_table.columns)))
                
            for i in column_list:
                column_name = df_table.columns[i]
                
                try:
                    # check if the semantic type of the columns is in the valid semantic types that we consider in our experiments 
                    # in case we want to filter assigned types regarding the similarity score later on
                    if len(df_table[column_name].dropna()) > 0: # filter out columns that contain only nan values
                        if dbpedia_similarities[column_name] >= self.semantic_sim_score: # filter semantic type similarity scores under a threshold
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
                    
                data_list.append([
                    table_path,
                    column_name,
                    column_data_type,
                    column_label,
                    " ".join([str(x) for x in df_table.iloc[:, i].dropna().tolist()])
                ])
                
            self.df = pd.DataFrame(data_list, columns=[
                                    "table_name", "column_name", "columns_data_type", "column_label", "data"])
            if len(self.df) == 0:
                #print(f"Table {table_path} has no columns with assigned semantic types!")
                continue
            if len(self.df[self.df["columns_data_type"] == 1].index.tolist()) == 0:
                #print(f"Table {table_path} has no numerical columns. Continue...")
                continue
             
            # self.df["data_tensor"] = self.df["data"].apply(
            #     lambda x: torch.LongTensor(
            #         tokenizer.encode(
            #             x, return_tensors="pt", add_special_tokens=True, max_length=max_length + 2, truncation=True)).to(
            #                 device)).tolist()
            
            self.df = self.df.dropna().reset_index(drop=True)
                       
            self.df["data_tensor"] = self.tokenizer([seq for seq in self.df["data"].tolist()], padding="max_length", max_length=self.max_length + 2, truncation=True)["input_ids"]

            self.df["label_tensor"] = self.df["column_label"].apply(
                lambda x: torch.LongTensor(label_enc.transform([x])).to(self.device))

            # tokenize table name
            tokenized_table_name = self.tokenizer(table_path.split("/")[-1].split(".parquet")[0], padding="max_length", max_length=20, truncation=True)["input_ids"]
            # build numerical features for each numerical table column
            num_features = []
            for idx, row in self.df.iterrows():
                if row['columns_data_type'] == 1:
                    try:
                        sherlock_feature = df_sherlock_features.query(f"locator == '{row['table_name'].split('/')[-2]}' & dataset_id == '{row['table_name'].split('/')[-1]}' & header == '{row['column_name']}'")[selected_sherlock_feature_set].iloc[0].tolist()
                    except:
                        #print(f"No sherlock features for {row['column_name']} in table {row['table_name']}")
                        #print("Using feature vector filled with zeros")
                        sherlock_feature = [0] * len(selected_sherlock_feature_set) 
                    try:
                        num_stats_feature = df_num_stats_features.query(f"domain == '{row['table_name'].split('/')[-2]}' & table_name == '{row['table_name'].split('/')[-1]}' & column_name == '{row['column_name']}'")["features"].iloc[0]
                        num_stats_feature = literal_eval(num_stats_feature)
                    except:
                        #print(f"No num_feature for {row['column_name']} in table {row['table_name']}")
                        #print("Using feature vector filled with zeros")
                        num_stats_feature = [0]*15
                    num_features.append(sherlock_feature+num_stats_feature)
            
            #self.data_list.append(self.df)
            self.data_list.append(convert_Table_to_dgl_graph_enriched(self.df, tokenized_table_name, num_features).to(self.device))
            
            # add the current table as graph representation to the list of all graphs
            #self.graph_list.append(convert_Table_to_dgl_graph(self.df))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
    
    def save(self):
        print("Saving graphs...")
        path = join(
            self.save_dir, f"GitTables_{self.split}_dataset_enriched_ml{self.max_length}_{self.random_state}_{self.shuffle_cols}_{self.semantic_sim_score}.bin")
        save_graphs(path, self.data_list)

    def load(self):
        print("Loading graphs from tmp file...")
        path = join(
            self.save_dir, f"GitTables_{self.split}_dataset_enriched_ml{self.max_length}_{self.random_state}_{self.shuffle_cols}_{self.semantic_sim_score}.bin")
        self.data_list = load_graphs(path)[0]

    def has_cache(self):
        path = join(
            self.save_dir, f"GitTables_{self.split}_dataset_enriched_ml{self.max_length}_{self.random_state}_{self.shuffle_cols}_{self.semantic_sim_score}.bin")
        print(f"Has Chache? => {os.path.exists(path)}")
        print(path)
        return os.path.exists(path)
    
    def to_device(self):
        self.data_list = [graph.to(self.device) for graph in self.data_list]
    
        
        
def convert_Table_to_dgl_graph(df: pd.DataFrame, tokenized_table_name):
    """ This function is to convert the table to a graph representation where each 
     numeric column node is connected to each textual column node.
     There is also a node which represents the table name. This node is
     connected to all column nodes
    """
    # directed edges from each textual column to each numerical columns
    textual_node_ids = df[df["columns_data_type"] == "textual"].index.tolist()
    numerical_node_ids = df[df["columns_data_type"]
                            == "numerical"].index.tolist()
    graph_data = {}
    
    # table_name to each column
    source_nodes = []
    target_nodes = []
    for column_id in df.index.tolist():
        source_nodes.append(0)
        target_nodes.append(column_id)
    graph_data[('table', 'table_column', 'column')] = (torch.tensor(source_nodes), torch.tensor(target_nodes))
    
    # text to num columns
    source_nodes = []
    target_nodes = []
    for numerical_node_id in numerical_node_ids:
        for textual_node_id in textual_node_ids:
            source_nodes.append(textual_node_id)
            target_nodes.append(numerical_node_id)
    # add self-loop for column nodes
    source_nodes.extend(textual_node_ids+numerical_node_ids)
    target_nodes.extend(textual_node_ids+numerical_node_ids)
    
    graph_data[('column', 'column_column', 'column')] = (torch.tensor(source_nodes), torch.tensor(target_nodes))
    
    # add self-loop for column nodes
    #source_nodes = textual_node_ids+numerical_node_ids
    #target_nodes = textual_node_ids+numerical_node_ids
    #graph_data[('column', 'column_column', 'column')] = (torch.tensor(source_nodes), torch.tensor(target_nodes))
    
    g = dgl.heterograph(graph_data) 
    
    # set features to the nodes
    g.nodes["table"].data["data_tensor"] = torch.tensor([tokenized_table_name])
    g.nodes["column"].data["data_tensor"] = torch.LongTensor(df["data_tensor"].tolist())
    g.nodes["column"].data["label_tensor"] = torch.LongTensor(df["label_tensor"].tolist())
    
    def fill_tensor_with_zeros(tensor, desired_length, dim=0):
        current_length = tensor.size(dim)
        if current_length < desired_length:
            pad_size = [0] * tensor.dim()
            pad_size[dim] = desired_length - current_length
            tensor = torch.cat((tensor, torch.zeros(*pad_size)), dim=dim)
        return tensor
    
    return g
        
        
class GitTablesDGLDataset(DGLDataset):
    """ Dataloader that loads the graph representation with the additional node for numeric
    column that contains a vector of special numeric features (sherlock numeric feature)
    """
    # TODO: Consider that the graph is contructed with only columns from the table that has a valid semantic type.
    # In current state, all columns are considered, wether they have a semantic type or not. Large graphs leads to long code running durations, which is a problem. 
    
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        save_dir:str = join(os.environ["MAIN_DIR"], "data_loader", "tmp"),
        force_reload:bool = False,
        max_length: int = 128,
        device: torch.device = torch.device("cpu"),
        random_state:int = 1,
        split:str = "train",
        shuffle_cols:bool = False
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.random_state = random_state
        self.split = split
        self.shuffle_cols = shuffle_cols
        super(GitTablesDGLDataset_enriched, self).__init__(name="GitTables", save_dir=save_dir, force_reload=force_reload)
        
                 
    def process(self):
        print("Executing dataset generating process...")
        # get the Label Encoder to encode the semantic types 
        label_enc = get_LabelEncoder()
        
        # load sherlock features
        print("Loading sherlock feature set...")
        df_sherlock_features = pd.read_csv("/home/slangenecker/sato/extract/out/features/type_gittables/gittables_type_gittables_sherlock_features.csv")
        with open(join(os.environ["MAIN_DIR"], "data_loader","valid_sherlock_features.json"), "r") as f:
            selected_sherlock_feature_set = json.load(f)
        df_sherlock_features = df_sherlock_features[["locator", "dataset_id", "field_id", "header"]+selected_sherlock_feature_set]
        
        # load additional numerical stats features
        df_num_stats_features = pd.read_csv(join(os.environ["MAIN_DIR"],"data_loader", "features","GitTables_numerical_statistic_features.csv"))
        
        # load all tables from GitTables, tokenize every table column as described in Doduo and build the graph representation
        self.data_list = []
        
        # if idx_sport_domain > 0:
        #     break
        ## load train_valid_test split definitions
        with open(join(os.environ["GitTables"], "data", f"train_valid_test_split_{self.random_state}.json")) as f:
            train_valid_test_split = json.load(f)

        for idx_table_path, table_path in tqdm(enumerate(train_valid_test_split[self.split]), total=len(train_valid_test_split[self.split])):
            # if idx_table_path > 0:
            #     continue
            #print(table_path)
            
            # read metadata
            table_metadata = json.loads(pq.read_schema(join(os.environ["GitTables"], table_path)).metadata[b"gittables"])
            dbpedia_types = table_metadata["dbpedia_embedding_column_types"]
            dbpedia_similarities = table_metadata["dbpedia_embedding_similarities"]

            # read the table in a DF
            df_table = pd.read_parquet(join(os.environ["GitTables"], table_path))
            
            data_list = []
            
            if self.shuffle_cols:
                column_list = list(range(len(df_table.columns)))
                #random.seed(self.random_state)
                random.shuffle(column_list)
            else:
                column_list = list(range(len(df_table.columns)))
                
            for i in column_list:
                column_name = df_table.columns[i]
                
                try:
                    # check if the semantic type of the columns is in the valid semantic types that we consider in our experiments 
                    if dbpedia_types[column_name]["cleaned_label"] in label_enc.classes_:
                        # in case we want to filter assigned types regarding the similarity score later on
                        if dbpedia_similarities[column_name] > 0.0: 
                            if table_metadata["dtypes"][column_name] == "object" or table_metadata["dtypes"][column_name] == "string":
                                column_data_type = "textual"
                                column_label = dbpedia_types[column_name]["cleaned_label"]
                            else:
                                column_data_type = "numerical"
                                column_label = dbpedia_types[column_name]["cleaned_label"]
                    else:
                        continue
                    
                except Exception as e:
                    continue
                    #print(e)
                    #print(f"Not considering column: {column_name} from table: {table_path}")
                    
                data_list.append([
                    table_path,
                    column_name,
                    column_data_type,
                    column_label,
                    " ".join([str(x) for x in df_table.iloc[:, i].dropna().tolist()]) # get the data for each column. The values are concatenated with " " between the individual values
                ])
                
            self.df = pd.DataFrame(data_list, columns=[
                                    "table_name", "column_name", "columns_data_type", "column_label", "data"])
            if len(self.df) == 0:
                print(f"Table {table_path} has no columns with assigned semantic types!")
                continue
            if len(self.df[self.df["columns_data_type"] == "numerical"].index.tolist()) == 0:
                print(f"Table {table_path} has no numerical columns. Continue...")
                continue
            # self.df["data_tensor"] = self.df["data"].apply(
            #     lambda x: torch.LongTensor(
            #         tokenizer.encode(
            #             x, return_tensors="pt", add_special_tokens=True, max_length=max_length + 2, truncation=True)).to(
            #                 device)).tolist()
            
            self.df = self.df.dropna().reset_index(drop=True)
                       
            self.df["data_tensor"] = self.tokenizer([seq for seq in self.df["data"].tolist()], padding="max_length", max_length=self.max_length + 2, truncation=True)["input_ids"]

            self.df["label_tensor"] = self.df["column_label"].apply(
                lambda x: torch.LongTensor(label_enc.transform([x])).to(self.device))

            # tokenize table name
            tokenized_table_name = self.tokenizer(table_path.split("/")[-1].split(".parquet")[0], padding="max_length", max_length=20, truncation=True)["input_ids"]
            
            #self.data_list.append(self.df)
            self.data_list.append(convert_Table_to_dgl_graph(self.df, tokenized_table_name).to(self.device))
            
            # add the current table as graph representation to the list of all graphs
            #self.graph_list.append(convert_Table_to_dgl_graph(self.df))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
    
    def save(self):
        print("Saving graphs...")
        path = join(self.save_dir, f"GitTables_{self.split}_dataset_ml{self.max_length}_{self.random_state}_{self.shuffle_cols}.bin")
        save_graphs(path, self.data_list)
    
    def load(self):
        print("Loading graphs from tmp file...")
        path = join(self.save_dir, f"GitTables_{self.split}_dataset_ml{self.max_length}_{self.random_state}_{self.shuffle_cols}.bin")
        self.data_list = load_graphs(path)[0]
        
    def has_cache(self):
        path = join(self.save_dir, f"GitTables_{self.split}_dataset_ml{self.max_length}_{self.random_state}_{self.shuffle_cols}.bin")
        print(f"Has Chache? => {os.path.exists(path)}")
        print(path)
        return os.path.exists(path)
    
    def to_device(self):
        self.data_list = [graph.to(self.device) for graph in self.data_list]
        
        
        
def convert_Table_to_dgl_graph_enriched_sep_col_dtype_nodes(df: pd.DataFrame, tokenized_table_name, num_features):
    """ This function is to convert the table to a graph representation where each 
     numeric column node is connected to each textual column node. Additionally a numeric node
     is also connected to another node, which represents the special numeric value features of the 
     numerical column (sherlock features). There is also a node which represents the table name. This node is
     connected to all column nodes
    """
    # directed edges from each textual column to each numerical columns
    textual_node_ids = df[df["columns_data_type"] == 0].index.tolist()
    numerical_node_ids = df[df["columns_data_type"] == 1].index.tolist()
    graph_data = {}
    
    # table_name to each column
    if len(textual_node_ids) > 0:
        graph_data[('table', 'provide_info', 'text_column')] = (torch.tensor([0] * len(textual_node_ids)), torch.tensor(range(len(textual_node_ids))))
    graph_data[('table', 'provide_info', 'num_column')] = (torch.tensor([0] * len(numerical_node_ids)), torch.tensor(range(len(numerical_node_ids))))
    
    # text to num columns
    if len(textual_node_ids) > 0:
        textual_to_numerical_source = torch.tensor(range(len(textual_node_ids))).repeat(len(numerical_node_ids)).to(torch.int64)
        textual_to_numerical_target = torch.tensor(range(len(numerical_node_ids))).repeat(len(textual_node_ids)).to(torch.int64)
        graph_data[('text_column', 'provide_info', 'num_column')] = (textual_to_numerical_source, textual_to_numerical_target)
    
    # add self-loop for column nodes
    #textual_to_numerical_source = torch.cat((textual_to_numerical_source, torch.tensor(textual_node_ids + numerical_node_ids))).to(torch.int64)
    #textual_to_numerical_target = torch.cat((textual_to_numerical_target, torch.tensor(textual_node_ids + numerical_node_ids))).to(torch.int64)
    if len(textual_node_ids) > 0:
        graph_data[('text_column', 'provide_info', 'text_column')] = (torch.tensor(range(len(textual_node_ids))).to(torch.int64), torch.tensor(range(len(textual_node_ids))).to(torch.int64))
    graph_data[('num_column', 'provide_info', 'num_column')] = (torch.tensor(range(len(numerical_node_ids))).to(torch.int64), torch.tensor(range(len(numerical_node_ids))).to(torch.int64))
    
    # numerical feature node to numerical column
    # build numerical feature node ids from 0 to len(numerical_node_ids) with the usage of idx
    numerical_feature_node_source = torch.arange(len(numerical_node_ids))
    numerical_feature_node_target = torch.arange(len(numerical_node_ids))
    graph_data[('num_feature_node', 'provide_info', 'num_column')] = (numerical_feature_node_source, numerical_feature_node_target)
    

    g = dgl.heterograph(graph_data)
    
    # set features to the nodes
    g.nodes["table"].data["data_tensor"] = torch.tensor([tokenized_table_name])
    if len(textual_node_ids) > 0:
        g.nodes["text_column"].data["data_tensor"] = torch.LongTensor(df[df["columns_data_type"] == 0]["data_tensor"].tolist())
        g.nodes["text_column"].data["label_tensor"] = torch.LongTensor(df[df["columns_data_type"] == 0]["label_tensor"].tolist())
        g.nodes["text_column"].data["col_type_tensor"] = torch.LongTensor(df[df["columns_data_type"] == 0]["columns_data_type"].tolist())
    g.nodes["num_column"].data["data_tensor"] = torch.LongTensor(df[df["columns_data_type"] == 1]["data_tensor"].tolist())
    g.nodes["num_column"].data["label_tensor"] = torch.LongTensor(df[df["columns_data_type"] == 1]["label_tensor"].tolist())
    g.nodes["num_column"].data["col_type_tensor"] = torch.LongTensor(df[df["columns_data_type"] == 1]["columns_data_type"].tolist())
    g.nodes["num_feature_node"].data["data_tensor"] = torch.tensor(num_features, dtype=torch.float64)
    
    return g
        
class GitTablesDGLDataset_enriched_sep_col_dtype_nodes(DGLDataset):
    """ Dataloader that loads the graph representation with the additional node for numeric
    column that contains a vector of special numeric features (sherlock numeric feature)
    """

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        save_dir:str = join(os.environ["MAIN_DIR"], "data_loader", "tmp"),
        force_reload:bool = False,
        max_length: int = 128,
        device: torch.device = torch.device("cpu"),
        random_state:int = 1,
        split:str = "train",
        shuffle_cols:bool = False
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.random_state = random_state
        self.split = split
        self.shuffle_cols = shuffle_cols
        super(GitTablesDGLDataset_enriched_sep_col_dtype_nodes, self).__init__(name="GitTables_sep", save_dir=save_dir, force_reload=force_reload)
        
                 
    def process(self):
        print("Executing dataset generating process...")
        # get the Label Encoder to encode the semantic types 
        label_enc = get_LabelEncoder()
        
        # load sherlock features
        with open(join(os.environ["MAIN_DIR"], "data_loader","valid_sherlock_features.json"), "r") as f:
            selected_sherlock_feature_set = json.load(f)
        print("Loading sherlock feature set...")
        df_sherlock_features = pd.read_csv("/ext/daten-wi/slangenecker/sato/extract/out/features/type_gittables/GitTables_type_gittables_sherlock_features.csv", usecols=["locator", "dataset_id", "field_id", "header", "header_c"]+selected_sherlock_feature_set)
        
        # load additional numerical stats features
        df_num_stats_features = pd.read_csv(join(os.environ["MAIN_DIR"],"data_loader", "features","GitTables_numerical_statistic_features.csv"))
        
        # load all tables from GitTables, tokenize every table column as described in Doduo and build the graph representation
        self.data_list = []
        
        # if idx_sport_domain > 0:
        #     break
        ## load train_valid_test split definitions
        with open(join(os.environ["GitTables"], "data", f"train_valid_test_split_{self.random_state}.json")) as f:
            train_valid_test_split = json.load(f)

        for idx_table_path, table_path in tqdm(enumerate(train_valid_test_split[self.split]), total=len(train_valid_test_split[self.split])):
            # if idx_table_path > 2:
            #     continue
            #print(table_path)
            
            # read metadata
            table_metadata = json.loads(pq.read_schema(join(os.environ["GitTables"], table_path)).metadata[b"gittables"])
            dbpedia_types = table_metadata["dbpedia_embedding_column_types"]
            dbpedia_similarities = table_metadata["dbpedia_embedding_similarities"]

            # read the table in a DF
            df_table = pd.read_parquet(join(os.environ["GitTables"], table_path))
            
            data_list = []
            
            if self.shuffle_cols:
                column_list = list(range(len(df_table.columns)))
                #random.seed(self.random_state)
                random.shuffle(column_list)
            else:
                column_list = list(range(len(df_table.columns)))
                
            for i in column_list:
                column_name = df_table.columns[i]
                
                try:
                    # check if the semantic type of the columns is in the valid semantic types that we consider in our experiments 
                    # in case we want to filter assigned types regarding the similarity score later on
                    if len(df_table[column_name].dropna()) > 0: # filter out columns that contain only nan values
                        if dbpedia_similarities[column_name] >= 0.8: # filter semantic type similarity scores under a threshold
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
                    
                data_list.append([
                    table_path,
                    column_name,
                    column_data_type,
                    column_label,
                    " ".join([str(x) for x in df_table.iloc[:, i].dropna().tolist()])
                ])
                
            self.df = pd.DataFrame(data_list, columns=[
                                    "table_name", "column_name", "columns_data_type", "column_label", "data"])
            if len(self.df) == 0:
                #print(f"Table {table_path} has no columns with assigned semantic types!")
                continue
            if len(self.df[self.df["columns_data_type"] == 1].index.tolist()) < 2:
                # if table has less than 2 numerical column, we skip a this point
                #print(f"Table {table_path} has no numerical columns. Continue...")
                continue
             
            # self.df["data_tensor"] = self.df["data"].apply(
            #     lambda x: torch.LongTensor(
            #         tokenizer.encode(
            #             x, return_tensors="pt", add_special_tokens=True, max_length=max_length + 2, truncation=True)).to(
            #                 device)).tolist()
            
            self.df = self.df.dropna().reset_index(drop=True)
                       
            self.df["data_tensor"] = self.tokenizer([seq for seq in self.df["data"].tolist()], padding="max_length", max_length=self.max_length + 2, truncation=True)["input_ids"]

            self.df["label_tensor"] = self.df["column_label"].apply(
                lambda x: torch.LongTensor(label_enc.transform([x])).to(self.device))

            # tokenize table name
            tokenized_table_name = self.tokenizer(table_path.split("/")[-1].split(".parquet")[0], padding="max_length", max_length=20, truncation=True)["input_ids"]
            # build numerical features for each numerical table column
            num_features = []
            for idx, row in self.df.iterrows():
                if row['columns_data_type'] == 1:
                    try:
                        sherlock_feature = df_sherlock_features.query(f"locator == '{row['table_name'].split('/')[-2]}' & dataset_id == '{row['table_name'].split('/')[-1]}' & header == '{row['column_name']}'")[selected_sherlock_feature_set].iloc[0].tolist()
                    except:
                        #print(f"No sherlock features for {row['column_name']} in table {row['table_name']}")
                        #print("Using feature vector filled with zeros")
                        sherlock_feature = [0] * len(selected_sherlock_feature_set) 
                    try:
                        num_stats_feature = df_num_stats_features.query(f"domain == '{row['table_name'].split('/')[-2]}' & table_name == '{row['table_name'].split('/')[-1]}' & column_name == '{row['column_name']}'")["features"].iloc[0]
                        num_stats_feature = literal_eval(num_stats_feature)
                    except:
                        #print(f"No num_feature for {row['column_name']} in table {row['table_name']}")
                        #print("Using feature vector filled with zeros")
                        num_stats_feature = [0]*15
                    num_features.append(sherlock_feature+num_stats_feature)
            
            #self.data_list.append(self.df)
            self.data_list.append(convert_Table_to_dgl_graph_enriched_sep_col_dtype_nodes(self.df, tokenized_table_name, num_features).to(self.device))
            
            # add the current table as graph representation to the list of all graphs
            #self.graph_list.append(convert_Table_to_dgl_graph(self.df))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
    
    def save(self):
        print("Saving graphs...")
        path = join(
            self.save_dir, f"GitTables_sep_{self.split}_dataset_enriched_ml{self.max_length}_{self.random_state}_{self.shuffle_cols}.bin")
        save_graphs(path, self.data_list)

    def load(self):
        print("Loading graphs from tmp file...")
        path = join(
            self.save_dir, f"GitTables_sep_{self.split}_dataset_enriched_ml{self.max_length}_{self.random_state}_{self.shuffle_cols}.bin")
        self.data_list = load_graphs(path)[0]

    def has_cache(self):
        path = join(
            self.save_dir, f"GitTables_sep_{self.split}_dataset_enriched_ml{self.max_length}_{self.random_state}_{self.shuffle_cols}.bin")
        print(f"Has Chache? => {os.path.exists(path)}")
        print(path)
        return os.path.exists(path)
    
    def to_device(self):
        self.data_list = [graph.to(self.device) for graph in self.data_list]
        
        
        
        
def convert_Table_to_dgl_graph_enriched_shadow_num_nodes(df: pd.DataFrame, tokenized_table_name, num_features):
    """ This function is to convert the table to a graph representation where each 
     numeric column node is connected to each textual column node. Additionally a numeric node
     is also connected to another node, which represents the special numeric value features of the 
     numerical column (sherlock features). There is also a node which represents the table name. This node is
     connected to all column nodes. 
     Additionally, a numerical column node is also connected to all other numerical columns nodes with an incoming edge to
     provide context information from neighbouring numerical columns. 
    """
    # extract node ids
    textual_node_ids = df[df["columns_data_type"] == 0].index.tolist()
    numerical_node_ids = df[df["columns_data_type"] == 1].index.tolist()
    graph_data = {}
    
    # table_name to each column
    table_to_column_source = torch.tensor([0] * len(df))
    table_to_column_target = torch.tensor(df.index.tolist())
    graph_data[('table', 'table_column', 'column')] = (table_to_column_source, table_to_column_target)
    
    # text to num columns
    textual_to_numerical_source = torch.tensor(textual_node_ids).repeat(len(numerical_node_ids)).to(torch.int64)
    textual_to_numerical_target = torch.tensor(numerical_node_ids * len(textual_node_ids)).to(torch.int64)
    graph_data[('column', 'text_num_col', 'column')] = (textual_to_numerical_source, textual_to_numerical_target)
    
    # num to num columns
    numerical_to_numerical_source = []
    numerical_to_numerical_target = []
    for i in numerical_node_ids:
        for j in numerical_node_ids:
            if i != j:  # Skip connecting a node to itself
                numerical_to_numerical_source.append(i)
                numerical_to_numerical_target.append(j)
    numerical_to_numerical_source = torch.tensor(numerical_to_numerical_source).to(torch.int64)
    numerical_to_numerical_target = torch.tensor(numerical_to_numerical_target).to(torch.int64)
    graph_data[('column', 'num_num_col', 'column')] = (numerical_to_numerical_source, numerical_to_numerical_target)
    
    # add self-loop for column nodes    
    graph_data[('column', 'self', 'column')] = (torch.tensor(textual_node_ids + numerical_node_ids), torch.tensor(textual_node_ids + numerical_node_ids))
        
    
    # numerical feature node to numerical column
    # build numerical feature node ids from 0 to len(numerical_node_ids) with the usage of idx
    numerical_feature_node_source = torch.arange(len(numerical_node_ids))
    numerical_feature_node_target = torch.tensor(numerical_node_ids)
    graph_data[('num_feature_node', 'num_feature_column', 'column')] = (numerical_feature_node_source, numerical_feature_node_target)
    

    g = dgl.heterograph(graph_data)
    
    # set features to the nodes
    g.nodes["table"].data["data_tensor"] = torch.tensor([tokenized_table_name])
    g.nodes["column"].data["data_tensor"] = torch.LongTensor(df["data_tensor"].tolist())
    g.nodes["column"].data["label_tensor"] = torch.LongTensor(df["label_tensor"].tolist())
    g.nodes["column"].data["col_type_tensor"] = torch.LongTensor(df["columns_data_type"].tolist())
    g.nodes["num_feature_node"].data["data_tensor"] = torch.tensor(num_features, dtype=torch.float64)
    
    return g
        
class GitTablesDGLDataset_enriched_shadow_num_nodes(DGLDataset):
    """ Dataloader that loads the graph representation with the additional node for numeric
    column that contains a vector of special numeric features (sherlock numeric feature)
    """

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        save_dir:str = join(os.environ["MAIN_DIR"], "data_loader", "tmp"),
        force_reload:bool = False,
        max_length: int = 128,
        device: torch.device = torch.device("cpu"),
        random_state:int = 1,
        split:str = "train",
        shuffle_cols:bool = False,
        semantic_sim_score:float = 0.7
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.random_state = random_state
        self.split = split
        self.shuffle_cols = shuffle_cols
        self.semantic_sim_score = semantic_sim_score
        super(GitTablesDGLDataset_enriched_shadow_num_nodes, self).__init__(name="GitTables_shadow", save_dir=save_dir, force_reload=force_reload)
        
                 
    def process(self):
        print("Executing dataset generating process...")
        # get the Label Encoder to encode the semantic types 
        label_enc = get_LabelEncoder()
        
        # load sherlock features
        with open(join(os.environ["MAIN_DIR"], "data_loader","valid_sherlock_features.json"), "r") as f:
            selected_sherlock_feature_set = json.load(f)
        print("Loading sherlock feature set...")
        df_sherlock_features = pd.read_csv("/ext/daten-wi/slangenecker/sato/extract/out/features/type_gittables/GitTables_type_gittables_sherlock_features.csv", usecols=["locator", "dataset_id", "field_id", "header", "header_c"]+selected_sherlock_feature_set)
        
        # load additional numerical stats features
        df_num_stats_features = pd.read_csv(join(os.environ["MAIN_DIR"],"data_loader", "features",f"GitTables_numerical_statistic_features_{self.semantic_sim_score}.csv"))
        
        # load all tables from GitTables, tokenize every table column as described in Doduo and build the graph representation
        self.data_list = []
        
        # if idx_sport_domain > 0:
        #     break
        ## load train_valid_test split definitions
        with open(join(os.environ["GitTables"], "data", f"train_valid_test_split_{self.random_state}_{self.semantic_sim_score}.json")) as f:
            train_valid_test_split = json.load(f)

        for idx_table_path, table_path in tqdm(enumerate(train_valid_test_split[self.split]), total=len(train_valid_test_split[self.split])):
            # if idx_table_path > 2:
            #     continue
            #print(table_path)
            
            # read metadata
            table_metadata = json.loads(pq.read_schema(join(os.environ["GitTables"], table_path)).metadata[b"gittables"])
            dbpedia_types = table_metadata["dbpedia_embedding_column_types"]
            dbpedia_similarities = table_metadata["dbpedia_embedding_similarities"]

            # read the table in a DF
            df_table = pd.read_parquet(join(os.environ["GitTables"], table_path))
            
            data_list = []
            
            if self.shuffle_cols:
                column_list = list(range(len(df_table.columns)))
                #random.seed(self.random_state)
                random.shuffle(column_list)
            else:
                column_list = list(range(len(df_table.columns)))
                
            for i in column_list:
                column_name = df_table.columns[i]
                
                try:
                    # check if the semantic type of the columns is in the valid semantic types that we consider in our experiments 
                    # in case we want to filter assigned types regarding the similarity score later on
                    if len(df_table[column_name].dropna()) > 0: # filter out columns that contain only nan values
                        if dbpedia_similarities[column_name] >= self.semantic_sim_score: # filter semantic type similarity scores under a threshold
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
                    
                data_list.append([
                    table_path,
                    column_name,
                    column_data_type,
                    column_label,
                    " ".join([str(x) for x in df_table.iloc[:, i].dropna().tolist()])
                ])
                
            self.df = pd.DataFrame(data_list, columns=[
                                    "table_name", "column_name", "columns_data_type", "column_label", "data"])
            if len(self.df) == 0:
                #print(f"Table {table_path} has no columns with assigned semantic types!")
                continue
            if len(self.df[self.df["columns_data_type"] == 1].index.tolist()) < 2:
                # if table has less than 2 numerical column, we skip a this point
                #print(f"Table {table_path} has no numerical columns. Continue...")
                continue
             
            # self.df["data_tensor"] = self.df["data"].apply(
            #     lambda x: torch.LongTensor(
            #         tokenizer.encode(
            #             x, return_tensors="pt", add_special_tokens=True, max_length=max_length + 2, truncation=True)).to(
            #                 device)).tolist()
            
            self.df = self.df.dropna().reset_index(drop=True)
                       
            self.df["data_tensor"] = self.tokenizer([seq for seq in self.df["data"].tolist()], padding="max_length", max_length=self.max_length + 2, truncation=True)["input_ids"]

            self.df["label_tensor"] = self.df["column_label"].apply(
                lambda x: torch.LongTensor(label_enc.transform([x])).to(self.device))

            # tokenize table name
            tokenized_table_name = self.tokenizer(table_path.split("/")[-1].split(".parquet")[0], padding="max_length", max_length=20, truncation=True)["input_ids"]
            # build numerical features for each numerical table column
            num_features = []
            for idx, row in self.df.iterrows():
                if row['columns_data_type'] == 1:
                    try:
                        sherlock_feature = df_sherlock_features.query(f"locator == '{row['table_name'].split('/')[-2]}' & dataset_id == '{row['table_name'].split('/')[-1]}' & header == '{row['column_name']}'")[selected_sherlock_feature_set].iloc[0].tolist()
                    except:
                        #print(f"No sherlock features for {row['column_name']} in table {row['table_name']}")
                        #print("Using feature vector filled with zeros")
                        sherlock_feature = [0] * len(selected_sherlock_feature_set) 
                    try:
                        num_stats_feature = df_num_stats_features.query(f"domain == '{row['table_name'].split('/')[-2]}' & table_name == '{row['table_name'].split('/')[-1]}' & column_name == '{row['column_name']}'")["features"].iloc[0]
                        num_stats_feature = literal_eval(num_stats_feature)
                    except:
                        #print(f"No num_feature for {row['column_name']} in table {row['table_name']}")
                        #print("Using feature vector filled with zeros")
                        num_stats_feature = [0]*15
                    num_features.append(sherlock_feature+num_stats_feature)
            
            #self.data_list.append(self.df)
            self.data_list.append(convert_Table_to_dgl_graph_enriched_shadow_num_nodes(self.df, tokenized_table_name, num_features).to(self.device))
            
            # add the current table as graph representation to the list of all graphs
            #self.graph_list.append(convert_Table_to_dgl_graph(self.df))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
    
    def save(self):
        print("Saving graphs...")
        path = join(
            self.save_dir, f"GitTables_shadow_nums_{self.split}_dataset_enriched_ml{self.max_length}_{self.random_state}_{self.shuffle_cols}_{self.semantic_sim_score}.bin")
        save_graphs(path, self.data_list)

    def load(self):
        print("Loading graphs from tmp file...")
        path = join(
            self.save_dir, f"GitTables_shadow_nums_{self.split}_dataset_enriched_ml{self.max_length}_{self.random_state}_{self.shuffle_cols}_{self.semantic_sim_score}.bin")
        self.data_list = load_graphs(path)[0]

    def has_cache(self):
        path = join(
            self.save_dir, f"GitTables_shadow_nums_{self.split}_dataset_enriched_ml{self.max_length}_{self.random_state}_{self.shuffle_cols}_{self.semantic_sim_score}.bin")
        print(f"Has Chache? => {os.path.exists(path)}")
        print(path)
        return os.path.exists(path)
    
    def to_device(self):
        self.data_list = [graph.to(self.device) for graph in self.data_list]