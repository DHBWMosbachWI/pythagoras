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
from dgl import save_graphs, load_graphs
from random import randrange


def get_all_textual_semantic_types(sport_domains: list = ["baseball", "basketball", "football", "hockey", "soccer"]):
    textual_semantic_types = []
    for sports_domain in sport_domains:
        with open(join(os.environ["SportsTables"], sports_domain, "metadata.json")) as f:
            metadata = json.load(f)
            for table_key in metadata.keys():
                textual_semantic_types.extend(
                    list(metadata[table_key]["textual_cols"].values()))

    return list(set([x for x in textual_semantic_types if x is not None]))


def get_all_numerical_semantic_types(sport_domains: list = ["baseball", "basketball", "football", "hockey", "soccer"]):
    numerical_semantic_types = []
    for sports_domain in sport_domains:
        with open(join(os.environ["SportsTables"], sports_domain, "metadata.json")) as f:
            metadata = json.load(f)
            for table_key in metadata.keys():
                numerical_semantic_types.extend(
                    list(metadata[table_key]["numerical_cols"].values()))

    return list(set(x for x in numerical_semantic_types if x is not None))


def get_LabelEncoder(sport_domains: list = ["baseball", "basketball", "football", "hockey", "soccer"]):
    all_semantic_types = get_all_textual_semantic_types() + \
        get_all_numerical_semantic_types()
    label_enc = LabelEncoder()
    label_enc.fit(all_semantic_types)
    return label_enc

def load_metadata_for_sportdomain(sport_domain:str):
    """This function loads the metadata (columns <-> semantic type) 
    for all tablecolumn in the given sportdomain

    Args:
        sport_domain (str): Name of the sportdomain (e.g. basketball)

    Returns:
        dict: dictionary which maps columns to semantic types
    """
    with open(join(os.environ["SportsTables"], sport_domain, "metadata.json")) as f:
        metadata = json.load(f)
    return metadata

def load_train_test_split_for_sportdomain(sport_domain:str, random_state:int):
    """This function loads the train valid test split specifications for the given 
    sportdomain

    Args:
        sport_domain (str): Name of the sportdomain (e.g. basketball)
        random_state (int): The random_state which should be used to load the splitting definitions

    Returns:
        dict: dictionary which defines the tablesets for training, validation and test
    """
    with open(join(os.environ["SportsTables"], sport_domain, f"train_valid_test_split_{random_state}.json")) as f:
        train_valid_test_split = json.load(f)
    return train_valid_test_split

def load_semantic_type_occurrences():
    """This function loads the dataframe which contains each semantic type in the SportsTables corpus 
    and their occurrences

    Returns:
        panadas.DataFrame: A DataFrame with the columns [semantic_type, count]
    """
    df = pd.read_csv("SportsTables_semantic_type_occurrences.csv")
    return df


def convert_Table_to_dgl_graph(df: pd.DataFrame):
    # directed edges from each textual column to each numerical columns
    textual_node_ids = df[df["columns_data_type"] == "textual"].index.tolist()
    numerical_node_ids = df[df["columns_data_type"]
                            == "numerical"].index.tolist()
    source_nodes = []
    target_nodes = []
    for numerical_node_id in numerical_node_ids:
        for textual_node_id in textual_node_ids:
            source_nodes.append(textual_node_id)
            target_nodes.append(numerical_node_id)

    g = dgl.graph((source_nodes, target_nodes), num_nodes=len(df))
    # maybe we dont need the node data here and provide the node feature later during training procedure
    g.ndata["data_tensor"] =  torch.LongTensor(df["data_tensor"].tolist())
    g.ndata["label_tensor"] = torch.LongTensor(df["label_tensor"].tolist())
    #g.ndata["table_name"] = torch.LongTensor(df["table_name"].tolist())
    #g.ndata["column_names"] = torch.LongTensor(df["column_name"].tolist())
    
    return g

def convert_Table_to_dgl_graph_with_table_name(df: pd.DataFrame, table_name_ids: list):
    # directed edges from each textual column to each numerical columns
    textual_node_ids = df[df["columns_data_type"] == "textual"].index.tolist()
    numerical_node_ids = df[df["columns_data_type"]
                            == "numerical"].index.tolist()
    # table_name node id has the the highest number
    table_name_node_id = len(df)
    
    source_nodes = []
    target_nodes = []
    for numerical_node_id in numerical_node_ids:
        # assign edges from table_name node to the numerical nodes
        source_nodes.append(table_name_node_id)
        target_nodes.append(numerical_node_id)
        
        # assign edges from each individual textual node to every numerical node
        for textual_node_id in textual_node_ids:
            source_nodes.append(textual_node_id)
            target_nodes.append(numerical_node_id)
        
    # assign edges from table_name node to the textual nodes
    for textual_node_id in textual_node_ids:
        source_nodes.append(table_name_node_id)
        target_nodes.append(textual_node_id)  

    g = dgl.graph((source_nodes, target_nodes), num_nodes=len(df)+1)
    # maybe we dont need the node data here and provide the node feature later during training procedure
    g.ndata["data_tensor"] =  torch.LongTensor(df["data_tensor"].tolist()+[table_name_ids])
    g.ndata["label_tensor"] = torch.LongTensor(df["label_tensor"].tolist()+[0])
    #g.ndata["table_name"] = torch.LongTensor(df["table_name"].tolist())
    #g.ndata["column_names"] = torch.LongTensor(df["column_name"].tolist())
    
    return g





class SportsTablesDataset(Dataset):
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        max_length: int = 128,
        device: torch.device = torch.device("cpu"),
        sport_domains: list = ["baseball", "basketball",
                               "football", "hockey", "soccer"]
    ):
        # get the Label Encoder to encode the semantic types 
        label_enc = get_LabelEncoder()

        # load all tables from SportsTables, tokenize every table column as described in Doduo and build the graph representation
        self.data_list = []
        for idx_sport_domain, sport_domain in enumerate(sport_domains):
            if idx_sport_domain > 0:
                break
            # load metadata.json containing semantic types of the columns
            with open(join(os.environ["SportsTables"], sport_domain, "metadata.json")) as f:
                metadata = json.load(f)

            for idx_table_path, table_path in enumerate(glob(join(os.environ["SportsTables"], sport_domain, "*.csv"))):
                # if idx_table_path != 1:
                #     continue
                table_name = table_path.split("/")[-1].split(".csv")[0]
                ## search for correct in key in metadata
                for key in metadata.keys():
                    if table_name in key:
                        table_metadata_key = key

                df = pd.read_csv(table_path)
                data_list = []
                for i in range(len(df.columns)):
                    column_name = df.columns[i]
                    # search for defined columns data type and semantic label in metadata
                    if column_name in metadata[table_metadata_key]["textual_cols"].keys():
                        column_data_type = "textual"
                        column_label = metadata[table_metadata_key]["textual_cols"][column_name]
                    elif column_name in metadata[table_metadata_key]["numerical_cols"].keys():
                        column_data_type = "numerical"
                        column_label = metadata[table_metadata_key]["numerical_cols"][column_name]

                    data_list.append([
                        table_name,  # table name
                        column_name,  # column name
                        column_data_type,
                        column_label,
                        " ".join([str(x)
                                 for x in df.iloc[:, i].dropna().tolist()]),
                    ])
                self.df = pd.DataFrame(data_list, columns=[
                                       "table_name", "column_name", "columns_data_type", "column_label", "data"])
                # self.df["data_tensor"] = self.df["data"].apply(
                #     lambda x: torch.LongTensor(
                #         tokenizer.encode(
                #             x, return_tensors="pt", add_special_tokens=True, max_length=max_length + 2, truncation=True)).to(
                #                 device)).tolist()

                self.df["data_tensor"] = tokenizer([seq for seq in self.df["data"].tolist()], padding=True, max_length=max_length + 2, truncation=True)["input_ids"]
                
                self.df = self.df.dropna().reset_index(drop=True)
                self.df["label_tensor"] = self.df["column_label"].apply(
                    lambda x: torch.LongTensor(label_enc.transform([x])).to(device))
                
                
                self.data_list.append(convert_Table_to_dgl_graph(self.df))
                
                # add the current table as graph representation to the list of all graphs
                #self.graph_list.append(convert_Table_to_dgl_graph(self.df))
                

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]



class SportsTablesDGLDataset(DGLDataset):
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        max_length: int = 128,
        device: torch.device = torch.device("cpu"),
        sport_domains: list = ["baseball", "basketball",
                               "football", "hockey", "soccer"],
        random_state:int = 1,
        split:str = "train",
        shuffle_cols:bool = False
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.sport_domains = sport_domains
        self.random_state = random_state
        self.split = split
        self.shuffle_cols = shuffle_cols
        super().__init__(name="SportsTables")
        
                 
    def process(self):
        # get the Label Encoder to encode the semantic types 
        label_enc = get_LabelEncoder()

        # load all tables from SportsTables, tokenize every table column as described in Doduo and build the graph representation
        self.data_list = []
        for idx_sport_domain, sport_domain in enumerate(self.sport_domains):
            # if idx_sport_domain > 0:
            #     break
            # load metadata.json containing semantic types of the columns
            with open(join(os.environ["SportsTables"], sport_domain, "metadata.json")) as f:
                metadata = json.load(f)
            with open(join(os.environ["SportsTables"], sport_domain, f"train_valid_test_split_{self.random_state}.json")) as f:
                train_valid_test_split = json.load(f)

            for idx_table_path, table_name_full in enumerate(train_valid_test_split[self.split]):
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
                data_list = []
                if self.shuffle_cols:
                    column_list = list(range(len(df.columns)))
                    #random.seed(self.random_state)
                    random.shuffle(column_list)
                else:
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
                    
                    data_list.append([
                        table_name,  # table name
                        column_name,  # column name
                        column_data_type,
                        column_label,
                        " ".join([str(x)
                                 for x in df.iloc[:, i].dropna().tolist()]),
                    ])
                self.df = pd.DataFrame(data_list, columns=[
                                       "table_name", "column_name", "columns_data_type", "column_label", "data"])
                if len(self.df) == 0:
                    print(f"Table {table_name} has no columns with assigned semantic types!")
                    continue
                # self.df["data_tensor"] = self.df["data"].apply(
                #     lambda x: torch.LongTensor(
                #         tokenizer.encode(
                #             x, return_tensors="pt", add_special_tokens=True, max_length=max_length + 2, truncation=True)).to(
                #                 device)).tolist()

                self.df["data_tensor"] = self.tokenizer([seq for seq in self.df["data"].tolist()], padding=True, max_length=self.max_length + 2, truncation=True)["input_ids"]
                
                self.df = self.df.dropna().reset_index(drop=True)
                self.df["label_tensor"] = self.df["column_label"].apply(
                    lambda x: torch.LongTensor(label_enc.transform([x])).to(self.device))
                
                
                #self.data_list.append(self.df)
                self.data_list.append(convert_Table_to_dgl_graph(self.df).to(self.device))
                
                # add the current table as graph representation to the list of all graphs
                #self.graph_list.append(convert_Table_to_dgl_graph(self.df))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
    

class SportsTablesDGLDataset_with_table_name(DGLDataset):
    """ Dataloader that loads the graph representation of a table with the additional node
    that represents the table name
    """
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        max_length: int = 128,
        device: torch.device = torch.device("cpu"),
        sport_domains: list = ["baseball", "basketball",
                               "football", "hockey", "soccer"],
        random_state:int = 1,
        split:str = "train",
        shuffle_cols:bool = False
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.sport_domains = sport_domains
        self.random_state = random_state
        self.split = split
        self.shuffle_cols = shuffle_cols
        super().__init__(name="SportsTables")
        
                 
    def process(self):
        # get the Label Encoder to encode the semantic types 
        label_enc = get_LabelEncoder()

        # load all tables from SportsTables, tokenize every table column as described in Doduo and build the graph representation
        self.data_list = []
        for idx_sport_domain, sport_domain in enumerate(self.sport_domains):
            # if idx_sport_domain > 0:
            #     break
            # load metadata.json containing semantic types of the columns
            with open(join(os.environ["SportsTables"], sport_domain, "metadata.json")) as f:
                metadata = json.load(f)
            with open(join(os.environ["SportsTables"], sport_domain, f"train_valid_test_split_{self.random_state}.json")) as f:
                train_valid_test_split = json.load(f)

            for idx_table_path, table_name_full in enumerate(train_valid_test_split[self.split]):
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
                data_list = []
                if self.shuffle_cols:
                    column_list = list(range(len(df.columns)))
                    #random.seed(self.random_state)
                    random.shuffle(column_list)
                else:
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
                    
                    data_list.append([
                        table_name,  # table name
                        column_name,  # column name
                        column_data_type,
                        column_label,
                        " ".join([str(x)
                                 for x in df.iloc[:, i].dropna().tolist()]),
                    ])
                self.df = pd.DataFrame(data_list, columns=[
                                       "table_name", "column_name", "columns_data_type", "column_label", "data"])
                if len(self.df) == 0:
                    print(f"Table {table_name} has no columns with assigned semantic types!")
                    continue

                # Put table name in the sequence of the column values to get same tensor length/sizes for a tablename as well as for the columns
                # This is neccesary for the dgl grap dataset were node features had to have the same feature vector sizes
                # last element in list is the table_name
                tokenized_inputs = self.tokenizer([seq for seq in self.df["data"].tolist()+[table_name]], padding=True, max_length=self.max_length + 2, truncation=True)["input_ids"]
                self.df["data_tensor"] = tokenized_inputs[:-1]
                
                self.df = self.df.dropna().reset_index(drop=True)
                self.df["label_tensor"] = self.df["column_label"].apply(
                    lambda x: torch.LongTensor(label_enc.transform([x])).to(self.device))
                
                
                #self.data_list.append(self.df)
                self.data_list.append(convert_Table_to_dgl_graph_with_table_name(self.df, tokenized_inputs[-1]).to(self.device))
                
                # add the current table as graph representation to the list of all graphs
                #self.graph_list.append(convert_Table_to_dgl_graph(self.df))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
    
def convert_Table_to_dgl_graph_enriched(df: pd.DataFrame, tokenized_table_name, num_features, table_to_column, column_to_column, numStatsFeat_to_column):
    """ This function is to convert the table to a graph representation where each 
     numeric column node is connected to each textual column node. Additionally a numeric node
     is also connected to another node, which represents the special numeric value features of the 
     numerical column (sherlock features). There is also a node which represents the table name. This node is
     connected to all column nodes
    """
    # extract node ids for textual as well as for numerical nodes
    textual_node_ids = df[df["columns_data_type"] == 0].index.tolist()
    numerical_node_ids = df[df["columns_data_type"] == 1].index.tolist()
    graph_data = {}
    
    # table_name to each column
    if table_to_column:
        table_to_column_source = torch.tensor([0] * len(df))
        table_to_column_target = torch.tensor(df.index.tolist())
        graph_data[('table', 'table_column', 'column')] = (table_to_column_source, table_to_column_target)
    
    # text to num columns
    if column_to_column:
        textual_to_numerical_source = torch.tensor(textual_node_ids).repeat(len(numerical_node_ids)).to(torch.int64)
        textual_to_numerical_target = torch.tensor(numerical_node_ids * len(textual_node_ids)).to(torch.int64)
        
        # add self-loop for column nodes
        textual_to_numerical_source = torch.cat((textual_to_numerical_source, torch.tensor(textual_node_ids + numerical_node_ids))).to(torch.int64)
        textual_to_numerical_target = torch.cat((textual_to_numerical_target, torch.tensor(textual_node_ids + numerical_node_ids))).to(torch.int64)
        
        graph_data[('column', 'column_column', 'column')] = (textual_to_numerical_source, textual_to_numerical_target)
    else:
        # we wanna keep self-loop for column nodes in any case
        graph_data[('column', 'column_column', 'column')] = (torch.tensor(textual_node_ids + numerical_node_ids), torch.tensor(textual_node_ids + numerical_node_ids))
    
    # numerical feature node to numerical column
    # build numerical feature node ids from 0 to len(numerical_node_ids) with the usage of idx
    if numStatsFeat_to_column:
        numerical_feature_node_source = torch.arange(len(numerical_node_ids))
        numerical_feature_node_target = torch.tensor(numerical_node_ids)
        graph_data[('num_feature_node', 'num_feature_column', 'column')] = (numerical_feature_node_source, numerical_feature_node_target)
    

    g = dgl.heterograph(graph_data)
    
    # set features to the nodes
    if table_to_column:
        g.nodes["table"].data["data_tensor"] = torch.tensor([tokenized_table_name])
    g.nodes["column"].data["data_tensor"] = torch.LongTensor(df["data_tensor"].tolist())
    g.nodes["column"].data["label_tensor"] = torch.LongTensor(df["label_tensor"].tolist())
    g.nodes["column"].data["col_type_tensor"] = torch.LongTensor(df["columns_data_type"].tolist())
    if numStatsFeat_to_column:
        g.nodes["num_feature_node"].data["data_tensor"] = torch.tensor(num_features, dtype=torch.float64)
    
    return g
    
class SportsTablesDGLDataset_enriched(DGLDataset):
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
        sport_domains: list = ["baseball", "basketball",
                               "football", "hockey", "soccer"],
        random_state:int = 1,
        split:str = "train",
        shuffle_cols:bool = False,
        # ablation studies setting:
        table_to_column: bool = True, # set table to column connections
        column_to_column: bool = True, # set column to column connections
        numStatsFeat_to_column:bool = True # set num_stats_feat_nodes to numerical column connections
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.sport_domains = sport_domains
        self.random_state = random_state
        self.split = split
        self.shuffle_cols = shuffle_cols
        # ablation studies setting:
        self.table_to_column = table_to_column
        self.column_to_column = column_to_column
        self.numStatsFeat_to_column = numStatsFeat_to_column
        
        super(SportsTablesDGLDataset_enriched, self).__init__(name="SportsTables", save_dir=save_dir, force_reload=force_reload)
        
                 
    def process(self):
        # get the Label Encoder to encode the semantic types 
        label_enc = get_LabelEncoder()
        
        # load sherlock features
        df_sherlock_features = pd.read_csv("/home/slangenecker/sato/extract/out/features/type_SportsTables/SportsTables_type_SportsTables_sherlock_features.csv")
        with open(join(os.environ["MAIN_DIR"], "data_loader","valid_sherlock_features.json"), "r") as f:
            selected_sherlock_feature_set = json.load(f)
        df_sherlock_features = df_sherlock_features[["locator", "dataset_id", "field_id", "header"]+selected_sherlock_feature_set]
        
        # load additional numerical stats features
        df_num_stats_features = pd.read_csv(join(os.environ["MAIN_DIR"],"data_loader", "features","SportsTables_numerical_statistic_features.csv"))
        
        # load all tables from SportsTables, tokenize every table column as described in Doduo and build the graph representation
        self.data_list = []
        for idx_sport_domain, sport_domain in enumerate(self.sport_domains):
            # if idx_sport_domain > 0:
            #     break
            # load metadata.json containing semantic types of the columns
            with open(join(os.environ["SportsTables"], sport_domain, "metadata.json")) as f:
                metadata = json.load(f)
            with open(join(os.environ["SportsTables"], sport_domain, f"train_valid_test_split_{self.random_state}.json")) as f:
                train_valid_test_split = json.load(f)

            for idx_table_path, table_name_full in enumerate(train_valid_test_split[self.split]):
                # if idx_table_path != 0:
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
                data_list = []
                if self.shuffle_cols:
                    column_list = list(range(len(df.columns)))
                    #random.seed(self.random_state)
                    random.shuffle(column_list)
                else:
                    column_list = list(range(len(df.columns)))
                    
                for i in column_list:
                    column_name = df.columns[i]
                    # search for defined columns data type and semantic label in metadata
                    if column_name in metadata[table_metadata_key]["textual_cols"].keys():
                        column_data_type = 0 # => "textual"
                        column_label = metadata[table_metadata_key]["textual_cols"][column_name]
                    elif column_name in metadata[table_metadata_key]["numerical_cols"].keys():
                        column_data_type = 1 # => "numerical"
                        column_label = metadata[table_metadata_key]["numerical_cols"][column_name]
                    else:
                        print(f"Column {df.columns[i]} in {table_name} not labeled in metadata.json!")
                        continue
                    
                    data_list.append([
                        table_name,  # table name
                        column_name,  # column name
                        column_data_type,
                        column_label,
                        " ".join([str(x)
                                 for x in df.iloc[:, i].dropna().tolist()]),
                    ])
                self.df = pd.DataFrame(data_list, columns=[
                                       "table_name", "column_name", "columns_data_type", "column_label", "data"])
                if len(self.df) == 0:
                    print(f"Table {table_name} has no columns with assigned semantic types!")
                    continue
                # self.df["data_tensor"] = self.df["data"].apply(
                #     lambda x: torch.LongTensor(
                #         tokenizer.encode(
                #             x, return_tensors="pt", add_special_tokens=True, max_length=max_length + 2, truncation=True)).to(
                #                 device)).tolist()
                
                self.df = self.df.dropna().reset_index(drop=True)
                
                ## assign numerical feature set to the columns
                
                self.df["data_tensor"] = self.tokenizer([seq for seq in self.df["data"].tolist()], padding=True, max_length=self.max_length + 2, truncation=True)["input_ids"]
                #self.df["data_tensor"] = self.tokenizer([seq for seq in self.df["data"].tolist()], padding="max_length", max_length=self.max_length + 2, truncation=True)["input_ids"]
                self.df["label_tensor"] = self.df["column_label"].apply(
                    lambda x: torch.LongTensor(label_enc.transform([x])).to(self.device))
                
                # tokenize table name
                tokenized_table_name = self.tokenizer(table_name)["input_ids"]
                #tokenized_table_name = self.tokenizer(table_name, padding="max_length", max_length=20, truncation=True)["input_ids"]
                
                # build numerical features for each numerical table column
                num_features = []
                for idx, row in self.df.iterrows():
                    if row['columns_data_type'] == 1: # "numerical":
                        try:
                            sherlock_feature = df_sherlock_features.query(f"locator == '{sport_domain}' & dataset_id == '{row['table_name']}.csv' & header == '{row['column_name']}'")[selected_sherlock_feature_set].iloc[0].tolist()
                            num_stats_feature = df_num_stats_features.query(f"domain == '{sport_domain}' & table_name == '{row['table_name']}.csv' & column_name == '{row['column_name']}'")["features"].iloc[0]
                            try:
                                num_stats_feature = literal_eval(num_stats_feature)
                            except:
                                print(f"No num_feature for {row['column_name']} in table {row['table_name']}.csv")
                                print("Using feature vector filled with zeros")
                                num_stats_feature = [0]*15
                            num_features.append(sherlock_feature+num_stats_feature)
                        except Exception as err:
                            print(err)
                            print(f"No sherlock or num_feature for {row['column_name']} in table {row['table_name']}.csv")
                
                #self.data_list.append(self.df)
                self.data_list.append(convert_Table_to_dgl_graph_enriched(self.df, tokenized_table_name, num_features, self.table_to_column, self.column_to_column, self.numStatsFeat_to_column).to(self.device))
                
                # add the current table as graph representation to the list of all graphs
                #self.graph_list.append(convert_Table_to_dgl_graph(self.df))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
    
    def save(self):
        print("Saving graphs...")
        path = join(
            self.save_dir, f"SportsTables_{self.split}_dataset_enriched_ml{self.max_length}_{self.random_state}_{self.shuffle_cols}_{self.table_to_column}_{self.column_to_column}_{self.numStatsFeat_to_column}.bin")
        save_graphs(path, self.data_list)

    def load(self):
        print("Loading graphs from tmp file...")
        path = join(
            self.save_dir, f"SportsTables_{self.split}_dataset_enriched_ml{self.max_length}_{self.random_state}_{self.shuffle_cols}_{self.table_to_column}_{self.column_to_column}_{self.numStatsFeat_to_column}.bin")
        self.data_list = load_graphs(path)[0]

    def has_cache(self):
        path = join(
            self.save_dir, f"SportsTables_{self.split}_dataset_enriched_ml{self.max_length}_{self.random_state}_{self.shuffle_cols}_{self.table_to_column}_{self.column_to_column}_{self.numStatsFeat_to_column}.bin")
        print(f"Has Chache? => {os.path.exists(path)}")
        print(path)
        return os.path.exists(path)
    
    def to_device(self):
        self.data_list = [graph.to(self.device) for graph in self.data_list]
        
class SportsTablesDGLDataset_enriched_column_names(DGLDataset):
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
        sport_domains: list = ["baseball", "basketball",
                               "football", "hockey", "soccer"],
        random_state:int = 1,
        split:str = "train",
        shuffle_cols:bool = False,
        gpt_gen_colnames = False,
        # ablation studies setting:
        table_to_column: bool = True, # set table to column connections
        column_to_column: bool = True, # set column to column connections
        numStatsFeat_to_column:bool = True # set num_stats_feat_nodes to numerical column connections
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.sport_domains = sport_domains
        self.random_state = random_state
        self.split = split
        self.shuffle_cols = shuffle_cols
        # ablation studies setting:
        self.table_to_column = table_to_column
        self.column_to_column = column_to_column
        self.numStatsFeat_to_column = numStatsFeat_to_column
        self.gpt_gen_colnames = gpt_gen_colnames
        
        super(SportsTablesDGLDataset_enriched_column_names, self).__init__(name="SportsTables", save_dir=save_dir, force_reload=force_reload)
        
                 
    def process(self):
        # get the Label Encoder to encode the semantic types 
        label_enc = get_LabelEncoder()
        
        # load gpt generated column names if neccessary
        if self.gpt_gen_colnames:
            with open(join(os.environ["MAIN_DIR"],"gpt","SportsTables_semantic_type_abbreviations.json")) as f:
                gpt_colnames = json.load(f)
        
        # load sherlock features
        df_sherlock_features = pd.read_csv("/home/slangenecker/sato/extract/out/features/type_SportsTables/SportsTables_type_SportsTables_sherlock_features.csv")
        with open(join(os.environ["MAIN_DIR"], "data_loader","valid_sherlock_features.json"), "r") as f:
            selected_sherlock_feature_set = json.load(f)
        df_sherlock_features = df_sherlock_features[["locator", "dataset_id", "field_id", "header"]+selected_sherlock_feature_set]
        
        # load additional numerical stats features
        df_num_stats_features = pd.read_csv(join(os.environ["MAIN_DIR"],"data_loader", "features","SportsTables_numerical_statistic_features.csv"))
        
        # load all tables from SportsTables, tokenize every table column as described in Doduo and build the graph representation
        self.data_list = []
        for idx_sport_domain, sport_domain in enumerate(self.sport_domains):
            # if idx_sport_domain > 0:
            #     break
            # load metadata.json containing semantic types of the columns
            with open(join(os.environ["SportsTables"], sport_domain, "metadata.json")) as f:
                metadata = json.load(f)
            with open(join(os.environ["SportsTables"], sport_domain, f"train_valid_test_split_{self.random_state}.json")) as f:
                train_valid_test_split = json.load(f)

            for idx_table_path, table_name_full in enumerate(train_valid_test_split[self.split]):
                # if idx_table_path != 0:
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
                data_list = []
                if self.shuffle_cols:
                    column_list = list(range(len(df.columns)))
                    #random.seed(self.random_state)
                    random.shuffle(column_list)
                else:
                    column_list = list(range(len(df.columns)))
                    
                for i in column_list:
                    column_name = df.columns[i]
                    # search for defined columns data type and semantic label in metadata
                    if column_name in metadata[table_metadata_key]["textual_cols"].keys():
                        column_data_type = 0 # => "textual"
                        column_label = metadata[table_metadata_key]["textual_cols"][column_name]
                    elif column_name in metadata[table_metadata_key]["numerical_cols"].keys():
                        column_data_type = 1 # => "numerical"
                        column_label = metadata[table_metadata_key]["numerical_cols"][column_name]
                    else:
                        print(f"Column {df.columns[i]} in {table_name} not labeled in metadata.json!")
                        continue
                    
                    
                    
                    data_list.append([
                        table_name,  # table name
                        column_name,  # column name
                        column_data_type,
                        column_label,
                        " ".join([str(x)
                                 for x in df.iloc[:, i].dropna().tolist()]),
                    ])
                self.df = pd.DataFrame(data_list, columns=[
                                       "table_name", "column_name", "columns_data_type", "column_label", "data"])
                if len(self.df) == 0:
                    print(f"Table {table_name} has no columns with assigned semantic types!")
                    continue
                # self.df["data_tensor"] = self.df["data"].apply(
                #     lambda x: torch.LongTensor(
                #         tokenizer.encode(
                #             x, return_tensors="pt", add_special_tokens=True, max_length=max_length + 2, truncation=True)).to(
                #                 device)).tolist()
                
                self.df = self.df.dropna().reset_index(drop=True)
                
                # sign gpt generate colname if enabled. select randomly one name out of 10 provided
                if self.gpt_gen_colnames:
                    self.df["gpt_colname"] = [gpt_colnames[collabel][randrange(10)] for collabel in self.df["column_label"].tolist()]
                    self.df["data"] = self.df["gpt_colname"] + " " + self.df["data"]
                else:
                    ## add column name to the data separated with a space
                    self.df["data"] = self.df["column_name"] + " " + self.df["data"]
                
                self.df["data_tensor"] = self.tokenizer([seq for seq in self.df["data"].tolist()], padding=True, max_length=self.max_length + 2, truncation=True)["input_ids"]
                #self.df["data_tensor"] = self.tokenizer([seq for seq in self.df["data"].tolist()], padding="max_length", max_length=self.max_length + 2, truncation=True)["input_ids"]
                self.df["label_tensor"] = self.df["column_label"].apply(
                    lambda x: torch.LongTensor(label_enc.transform([x])).to(self.device))
                
                # tokenize table name
                tokenized_table_name = self.tokenizer(table_name)["input_ids"]
                #tokenized_table_name = self.tokenizer(table_name, padding="max_length", max_length=20, truncation=True)["input_ids"]
                
                # build numerical features for each numerical table column
                num_features = []
                for idx, row in self.df.iterrows():
                    if row['columns_data_type'] == 1: # "numerical":
                        try:
                            sherlock_feature = df_sherlock_features.query(f"locator == '{sport_domain}' & dataset_id == '{row['table_name']}.csv' & header == '{row['column_name']}'")[selected_sherlock_feature_set].iloc[0].tolist()
                            num_stats_feature = df_num_stats_features.query(f"domain == '{sport_domain}' & table_name == '{row['table_name']}.csv' & column_name == '{row['column_name']}'")["features"].iloc[0]
                            try:
                                num_stats_feature = literal_eval(num_stats_feature)
                            except:
                                print(f"No num_feature for {row['column_name']} in table {row['table_name']}.csv")
                                print("Using feature vector filled with zeros")
                                num_stats_feature = [0]*15
                            num_features.append(sherlock_feature+num_stats_feature)
                        except Exception as err:
                            print(err)
                            print(f"No sherlock or num_feature for {row['column_name']} in table {row['table_name']}.csv")
                
                #self.data_list.append(self.df)
                self.data_list.append(convert_Table_to_dgl_graph_enriched(self.df, tokenized_table_name, num_features, self.table_to_column, self.column_to_column, self.numStatsFeat_to_column).to(self.device))
                
                # add the current table as graph representation to the list of all graphs
                #self.graph_list.append(convert_Table_to_dgl_graph(self.df))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
    
    def save(self):
        print("Saving graphs...")
        path = join(
            self.save_dir, f"SportsTables_{self.split}_dataset_enriched_colnames_gptcolnames{self.gpt_gen_colnames}_ml{self.max_length}_{self.random_state}_{self.shuffle_cols}_{self.table_to_column}_{self.column_to_column}_{self.numStatsFeat_to_column}.bin")
        save_graphs(path, self.data_list)

    def load(self):
        print("Loading graphs from tmp file...")
        path = join(
            self.save_dir, f"SportsTables_{self.split}_dataset_enriched_colnames_gptcolnames{self.gpt_gen_colnames}_ml{self.max_length}_{self.random_state}_{self.shuffle_cols}_{self.table_to_column}_{self.column_to_column}_{self.numStatsFeat_to_column}.bin")
        self.data_list = load_graphs(path)[0]

    def has_cache(self):
        path = join(
            self.save_dir, f"SportsTables_{self.split}_dataset_enriched_colnames_gptcolnames{self.gpt_gen_colnames}_ml{self.max_length}_{self.random_state}_{self.shuffle_cols}_{self.table_to_column}_{self.column_to_column}_{self.numStatsFeat_to_column}.bin")
        print(f"Has Chache? => {os.path.exists(path)}")
        print(path)
        return os.path.exists(path)
    
    def to_device(self):
        self.data_list = [graph.to(self.device) for graph in self.data_list]
    
def convert_Table_to_dgl_graph_enriched_shadow_num_nodes(df: pd.DataFrame, tokenized_table_name, num_features, one_etype):
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
    if one_etype:
        graph_data[('table', 'edge', 'column')] = (table_to_column_source, table_to_column_target)
    else:
        graph_data[('table', 'table_column', 'column')] = (table_to_column_source, table_to_column_target)
    
    # text to num columns
    textual_to_numerical_source = torch.tensor(textual_node_ids).repeat(len(numerical_node_ids)).to(torch.int64)
    textual_to_numerical_target = torch.tensor(numerical_node_ids * len(textual_node_ids)).to(torch.int64)
    if one_etype:
        graph_data[('column', 'edge', 'column')] = (textual_to_numerical_source, textual_to_numerical_target)
    else:
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
    if one_etype:
        graph_data[('column', 'edge', 'column')] = (numerical_to_numerical_source, numerical_to_numerical_target)
    else:
        graph_data[('column', 'num_num_col', 'column')] = (numerical_to_numerical_source, numerical_to_numerical_target)
    
    # add self-loop for column nodes
    if one_etype:
        graph_data[('column', 'edge', 'column')] = (torch.tensor(textual_node_ids + numerical_node_ids), torch.tensor(textual_node_ids + numerical_node_ids))
    else:  
        graph_data[('column', 'self', 'column')] = (torch.tensor(textual_node_ids + numerical_node_ids), torch.tensor(textual_node_ids + numerical_node_ids))
        
    
    # numerical feature node to numerical column
    # build numerical feature node ids from 0 to len(numerical_node_ids) with the usage of idx
    numerical_feature_node_source = torch.arange(len(numerical_node_ids))
    numerical_feature_node_target = torch.tensor(numerical_node_ids)
    if one_etype:
        graph_data[('num_feature_node', 'edge', 'column')] = (numerical_feature_node_source, numerical_feature_node_target)
    else:
        graph_data[('num_feature_node', 'num_feature_column', 'column')] = (numerical_feature_node_source, numerical_feature_node_target)
    

    g = dgl.heterograph(graph_data)
    
    # set features to the nodes
    g.nodes["table"].data["data_tensor"] = torch.tensor([tokenized_table_name])
    g.nodes["column"].data["data_tensor"] = torch.LongTensor(df["data_tensor"].tolist())
    g.nodes["column"].data["label_tensor"] = torch.LongTensor(df["label_tensor"].tolist())
    g.nodes["column"].data["col_type_tensor"] = torch.LongTensor(df["columns_data_type"].tolist())
    g.nodes["num_feature_node"].data["data_tensor"] = torch.tensor(num_features, dtype=torch.float64)
    
    return g
    
class SportsTablesDGLDataset_enriched_shadow_num_nodes(DGLDataset):
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        save_dir:str = join(os.environ["MAIN_DIR"], "data_loader", "tmp"),
        force_reload:bool = False,
        max_length: int = 128,
        device: torch.device = torch.device("cpu"),
        sport_domains: list = ["baseball", "basketball", 
                               "football", "hockey", "soccer"],
        random_state:int = 1,
        split: str = "train",
        shuffle_cols: bool = False,
        one_etype: bool = False
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.sport_domains = sport_domains
        self.random_state = random_state
        self.split = split
        self.shuffle_cols = shuffle_cols
        self.one_etype = one_etype
        super().__init__(name="SportsTables")
        
    def process(self):
        # get the Label Encoder to encode the semantic types 
        label_enc = get_LabelEncoder()
        
        # load sherlock features
        df_sherlock_features = pd.read_csv("/home/slangenecker/sato/extract/out/features/type_SportsTables/SportsTables_type_SportsTables_sherlock_features.csv")
        with open(join(os.environ["MAIN_DIR"], "data_loader","valid_sherlock_features.json"), "r") as f:
            selected_sherlock_feature_set = json.load(f)
        df_sherlock_features = df_sherlock_features[["locator", "dataset_id", "field_id", "header"]+selected_sherlock_feature_set]
        
        # load additional numerical stats features
        df_num_stats_features = pd.read_csv(join(os.environ["MAIN_DIR"],"data_loader", "features","SportsTables_numerical_statistic_features.csv"))
        
        # load all tables from SportsTables, tokenize every table column as described in Doduo and build the graph representation
        self.data_list = []
        for idx_sport_domain, sport_domain in enumerate(self.sport_domains):
            # if idx_sport_domain > 0:
            #     break
            # load metadata.json containing semantic types of the columns
            with open(join(os.environ["SportsTables"], sport_domain, "metadata.json")) as f:
                metadata = json.load(f)
            with open(join(os.environ["SportsTables"], sport_domain, f"train_valid_test_split_{self.random_state}.json")) as f:
                train_valid_test_split = json.load(f)

            for idx_table_path, table_name_full in enumerate(train_valid_test_split[self.split]):
                # if idx_table_path != 0:
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
                data_list = []
                if self.shuffle_cols:
                    column_list = list(range(len(df.columns)))
                    #random.seed(self.random_state)
                    random.shuffle(column_list)
                else:
                    column_list = list(range(len(df.columns)))
                    
                for i in column_list:
                    column_name = df.columns[i]
                    # search for defined columns data type and semantic label in metadata
                    if column_name in metadata[table_metadata_key]["textual_cols"].keys():
                        column_data_type = 0 # => "textual"
                        column_label = metadata[table_metadata_key]["textual_cols"][column_name]
                    elif column_name in metadata[table_metadata_key]["numerical_cols"].keys():
                        column_data_type = 1 # => "numerical"
                        column_label = metadata[table_metadata_key]["numerical_cols"][column_name]
                    else:
                        print(f"Column {df.columns[i]} in {table_name} not labeled in metadata.json!")
                        continue
                    
                    data_list.append([
                        table_name,  # table name
                        column_name,  # column name
                        column_data_type,
                        column_label,
                        " ".join([str(x)
                                 for x in df.iloc[:, i].dropna().tolist()]),
                    ])
                self.df = pd.DataFrame(data_list, columns=[
                                       "table_name", "column_name", "columns_data_type", "column_label", "data"])
                if len(self.df) == 0:
                    print(f"Table {table_name} has no columns with assigned semantic types!")
                    continue
                # self.df["data_tensor"] = self.df["data"].apply(
                #     lambda x: torch.LongTensor(
                #         tokenizer.encode(
                #             x, return_tensors="pt", add_special_tokens=True, max_length=max_length + 2, truncation=True)).to(
                #                 device)).tolist()
                
                self.df = self.df.dropna().reset_index(drop=True)
                
                ## assign numerical feature set to the columns
                
                self.df["data_tensor"] = self.tokenizer([seq for seq in self.df["data"].tolist()], padding=True, max_length=self.max_length + 2, truncation=True)["input_ids"]
                #self.df["data_tensor"] = self.tokenizer([seq for seq in self.df["data"].tolist()], padding="max_length", max_length=self.max_length + 2, truncation=True)["input_ids"]
                self.df["label_tensor"] = self.df["column_label"].apply(
                    lambda x: torch.LongTensor(label_enc.transform([x])).to(self.device))
                
                # tokenize table name
                tokenized_table_name = self.tokenizer(table_name)["input_ids"]
                #tokenized_table_name = self.tokenizer(table_name, padding="max_length", max_length=20, truncation=True)["input_ids"]
                
                
                # build numerical features for each numerical table column
                num_features = []
                for idx, row in self.df.iterrows():
                    if row['columns_data_type'] == 1: # "numerical":
                        try:
                            sherlock_feature = df_sherlock_features.query(f"locator == '{sport_domain}' & dataset_id == '{row['table_name']}.csv' & header == '{row['column_name']}'")[selected_sherlock_feature_set].iloc[0].tolist()
                            num_stats_feature = df_num_stats_features.query(f"domain == '{sport_domain}' & table_name == '{row['table_name']}.csv' & column_name == '{row['column_name']}'")["features"].iloc[0]
                            try:
                                num_stats_feature = literal_eval(num_stats_feature)
                            except:
                                print(f"No num_feature for {row['column_name']} in table {row['table_name']}.csv")
                                print("Using feature vector filled with zeros")
                                num_stats_feature = [0]*15
                            num_features.append(sherlock_feature+num_stats_feature)
                        except Exception as err:
                            print(err)
                            print(f"No sherlock or num_feature for {row['column_name']} in table {row['table_name']}.csv")
                
                #self.data_list.append(self.df)
                self.data_list.append(convert_Table_to_dgl_graph_enriched_shadow_num_nodes(self.df, tokenized_table_name, num_features, self.one_etype).to(self.device))
                
                # add the current table as graph representation to the list of all graphs
                #self.graph_list.append(convert_Table_to_dgl_graph(self.df))
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
    
    def save(self):
        print("Saving graphs...")
        path = join(
            self.save_dir, f"SportsTables_{self.split}_dataset_enriched_shadow_num_nodes_ml{self.max_length}_{self.random_state}_{self.shuffle_cols}_{self.one_etype}.bin")
        save_graphs(path, self.data_list)

    def load(self):
        print("Loading graphs from tmp file...")
        path = join(
            self.save_dir, f"SportsTables_{self.split}_dataset_enriched_shadow_num_nodes_ml{self.max_length}_{self.random_state}_{self.shuffle_cols}_{self.one_etype}.bin")
        self.data_list = load_graphs(path)[0]

    def has_cache(self):
        path = join(
            self.save_dir, f"SportsTables_{self.split}_dataset_enriched_shadow_num_nodes_ml{self.max_length}_{self.random_state}_{self.shuffle_cols}_{self.one_etype}.bin")
        print(f"Has Chache? => {os.path.exists(path)}")
        print(path)
        return os.path.exists(path)
    
    def to_device(self):
        self.data_list = [graph.to(self.device) for graph in self.data_list]

    
        
    
class SportsTablesDGLDataset_df(DGLDataset):
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        max_length: int = 128,
        device: torch.device = torch.device("cpu"),
        sport_domains: list = ["baseball", "basketball",
                               "football", "hockey", "soccer"],
        random_state:int = 1,
        split:str = "train",
        shuffle_cols:bool = False
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.sport_domains = sport_domains
        self.random_state = random_state
        self.split = split
        self.shuffle_cols = shuffle_cols
        super().__init__(name="SportsTables")
        
                 
    def process(self):
        # get the Label Encoder to encode the semantic types 
        label_enc = get_LabelEncoder()

        # load all tables from SportsTables, tokenize every table column as described in Doduo and build the graph representation
        self.data_list = []
        for idx_sport_domain, sport_domain in enumerate(self.sport_domains):
            # if idx_sport_domain > 0:
            #     break
            # load metadata.json containing semantic types of the columns
            with open(join(os.environ["SportsTables"], sport_domain, "metadata.json")) as f:
                metadata = json.load(f)
            with open(join(os.environ["SportsTables"], sport_domain, f"train_valid_test_split_{self.random_state}.json")) as f:
                train_valid_test_split = json.load(f)

            for idx_table_path, table_name_full in enumerate(train_valid_test_split[self.split]):
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
                data_list = []
                if self.shuffle_cols:
                    column_list = list(range(len(df.columns)))
                    #random.seed(self.random_state)
                    random.shuffle(column_list)
                else:
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
                    
                    data_list.append([
                        table_name,  # table name
                        column_name,  # column name
                        column_data_type,
                        column_label,
                        " ".join([str(x)
                                 for x in df.iloc[:, i].dropna().tolist()]),
                    ])
                self.df = pd.DataFrame(data_list, columns=[
                                       "table_name", "column_name", "columns_data_type", "column_label", "data"])
                if len(self.df) == 0:
                    print(f"Table {table_name} has no columns with assigned semantic types!")
                    continue
                # self.df["data_tensor"] = self.df["data"].apply(
                #     lambda x: torch.LongTensor(
                #         tokenizer.encode(
                #             x, return_tensors="pt", add_special_tokens=True, max_length=max_length + 2, truncation=True)).to(
                #                 device)).tolist()

                self.df["data_tensor"] = self.tokenizer([seq for seq in self.df["data"].tolist()], padding=True, max_length=self.max_length + 2, truncation=True)["input_ids"]
                
                self.df = self.df.dropna().reset_index(drop=True)
                self.df["label_tensor"] = self.df["column_label"].apply(
                    lambda x: torch.LongTensor(label_enc.transform([x])).to(self.device))
                
                
                self.data_list.append(self.df)
                #self.data_list.append(convert_Table_to_dgl_graph(self.df).to(self.device))
                
                # add the current table as graph representation to the list of all graphs
                #self.graph_list.append(convert_Table_to_dgl_graph(self.df))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
    
