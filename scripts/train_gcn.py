import sys
sys.path.append("..")
import argparse
import torch
import os
from os.path import join

from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from model.gcn import CA_GCN, CA_GCN_Tablewise, CA_GCN_Conv1, CA_GCN_Tablewise_Conv1, CA_GAT, CA_GAT_Tablewise, CA_GCN_Conv1_enriched, CA_GAT_enriched, CA_GCN_Conv3_enriched, CA_GAT_enriched_shadow_nums
from data_loader.SportsDB_data_loader import SportsTablesDGLDataset, get_LabelEncoder, SportsTablesDGLDataset_with_table_name, SportsTablesDGLDataset_enriched, SportsTablesDGLDataset_enriched_shadow_num_nodes, SportsTablesDGLDataset_enriched_column_names
from data_loader.GitTables_data_loader import GitTablesDGLDataset_enriched, GitTablesDGLDataset_enriched_sep_col_dtype_nodes, GitTablesDGLDataset_enriched_shadow_num_nodes
from torch.nn import CrossEntropyLoss
from time import time
import dgl
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm, trange

from dgl.dataloading import GraphDataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import json

import logging


if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    print(f"Device: {device}")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bert_shortcut_name",
        default="bert-base-uncased",
        type=str,
        help="Huggingface model shortcut name"
    )
    parser.add_argument(
        "--model_architecture",
        default="CA_GCN",
        type=str,
        help="The name of the modelclass with which the training should be executed",
        choices=["CA_GCN", "CA_GCN_Conv1", "CA_GAT", "CA_GCN_Conv1_enriched", "CA_GAT_enriched", "CA_GCN_Conv3_enriched", "CA_GCN_Conv1_enriched_no_bert_for_nums", "CA_GAT_enriched_shadow_nums"]
    )
    parser.add_argument(
        "--table_graph_representation",
        default="columns",
        type=str,
        help="The graph representation for a table that shouls be used. For example only use columns as graph nodes. Or use column nodes and an additional node for the table_name and so on.",
        choices=["columns", "columns+table_name", "enriched", "enriched_sep", "enriched_shadow_nums", "enriched_column_names", "enriched_gpt_column_names"]
    )
    parser.add_argument(
        "--test_size",
        default = .2,
        type=float,
        help="Size of the test split e.g. 0.2 for 8/2 split"
    )
    parser.add_argument(
        "--max_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
    )
    parser.add_argument(
        "--num_classes",
        default=462,
        type=int,
        help="The number of classes e.g. the number of different semantic types the model has to learn."
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Batch Size"
    )
    parser.add_argument(
        "--num_train_epochs",
        default=10,
        type=int,
        help="Number of epochs for training"
    )
    parser.add_argument(
        "--random_seed",
        default=1,
        type=int,
        help="Random seed"
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-5,
        type=float,
        help="Learning rate of the optimizer for training"
    )
    parser.add_argument(
        "--gcn_hidden_feats",
        default=512,
        type=int,
        help="Size of the hidden tensors of the GraphConv Layers"
    )
    parser.add_argument(
        "--sport_domains",
        type=str,
        nargs="+",
        default = ["baseball"],
        choices=["baseball", "basketball", "football", "hockey", "soccer"]
    )
    parser.add_argument(
        "--shuffle_cols",
        default = False,
        action="store_true",
        help="Shuffle column order of the table before serialization"
    )
    parser.add_argument(
        "--tablewise", 
        default = False,
        action="store_true",
        help="Coumnwise or Tablewise serialization"
    )
    parser.add_argument(
        "--data_corpus",
        default="SportsTables",
        choices=["SportsTables", "GitTables"]
    )
    parser.add_argument(
        "--layers_to_freeze",
        type=int,
        default=10
    )
    parser.add_argument(
        "--table_to_column", 
        default = False,
        action="store_true",
        help="Coumnwise or Tablewise serialization"
    )
    parser.add_argument(
        "--column_to_column", 
        default = False,
        action="store_true",
        help="Coumnwise or Tablewise serialization"
    )
    parser.add_argument(
        "--numStatsFeat_to_column", 
        default = False,
        action="store_true",
        help="Coumnwise or Tablewise serialization"
    )
    parser.add_argument(
        "--one_etype",
        default = False,
        action="store_true",
        help="If true, than the graph contains only one etype. Otherwise we have multiple etype like table_to_column, column_to_column..."
    )

    
    # Parsing all given arguments
    args = parser.parse_args()
    
    print("args={}".format(json.dumps(vars(args))))
    
    sport_domains = args.sport_domains
    # sport_domains only in SportsTables corpus
    # TODO: Fix this behaviour. Its really bad implemented right now with the variable sport_domains in case we wana use other
    # corporas without sport_domains.
    if args.data_corpus != "SportsTables":
        sport_domains = None
    
    table_graph_representation = args.table_graph_representation

    
    # Check if the directory exists
    if not os.path.exists(join("..", "output", args.data_corpus, f"{args.model_architecture}")):
        # If it doesn't exist, create it
        os.makedirs(join("..", "output", args.data_corpus, f"{args.model_architecture}"))
    # Check if the directory exists
    if not os.path.exists(join("..", "output", args.data_corpus, f"{args.model_architecture}",table_graph_representation)):
        # If it doesn't exist, create it
        os.makedirs(join("..", "output", args.data_corpus, f"{args.model_architecture}",table_graph_representation))
    
    # build tag name of the model for savings
    if args.tablewise == False:
        tag_name = join("..", "output", args.data_corpus, f"{args.model_architecture}",table_graph_representation, f"single_{args.bert_shortcut_name}_ts{args.test_size}_ml{args.max_length}_nc{args.num_classes}_bs{args.batch_size}_rs{args.random_seed}_lr{args.learning_rate}_hf{args.gcn_hidden_feats}_{sport_domains}_sc{args.shuffle_cols}_lf{args.layers_to_freeze}_ttc{args.table_to_column}_ctc{args.column_to_column}_nSFtc{args.numStatsFeat_to_column}_oetype{args.one_etype}")
    else:
        tag_name = join("..", "output", args.data_corpus, f"{args.model_architecture}",table_graph_representation, f"table_{args.bert_shortcut_name}_ts{args.test_size}_ml{args.max_length}_nc{args.num_classes}_bs{args.batch_size}_rs{args.random_seed}_lr{args.learning_rate}_hf{args.gcn_hidden_feats}_{sport_domains}_sc{args.shuffle_cols}_lf{args.layers_to_freeze}")
    if args.data_corpus == "GitTables":
        sem_type_sim_score = 0.7
        tag_name = tag_name + f"_ss{sem_type_sim_score}"
    print(tag_name)
    
    ## logging set-up
    logging.basicConfig(filename=f"{tag_name}.log", filemode="w", format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    logging.debug("args={}".format(json.dumps(vars(args))))
    logging.debug(f"Device: {device}")
    logging.debug(f"Model tag name: {tag_name}")
    
    
    # Load tokenizer and initiate the NN model
    tokenizer = BertTokenizer.from_pretrained(args.bert_shortcut_name)
    if args.model_architecture == "CA_GCN":
        if args.tablewise == False:
            model = CA_GCN(args.bert_shortcut_name, args.gcn_hidden_feats, args.num_classes).to(device)
        else:
            model = CA_GCN_Tablewise(args.bert_shortcut_name, args.gcn_hidden_feats, args.num_classes).to(device)
    elif args.model_architecture == "CA_GCN_Conv1":
        if args.tablewise == False:
            model = CA_GCN_Conv1(args.bert_shortcut_name, args.gcn_hidden_feats, args.num_classes).to(device)
        else:
            #model = CA_GCN_Tablewise(args.bert_shortcut_name, args.gcn_hidden_feats, args.num_classes).to(device)
            model = CA_GCN_Tablewise_Conv1(args.bert_shortcut_name, args.gcn_hidden_feats, args.num_classes).to(device)
    elif args.model_architecture == "CA_GAT":
        if args.tablewise == False:
            model = CA_GAT(args.bert_shortcut_name, args.gcn_hidden_feats, args.num_classes).to(device)
        else:
            #model = CA_GCN_Tablewise(args.bert_shortcut_name, args.gcn_hidden_feats, args.num_classes).to(device)
            model = CA_GAT_Tablewise(args.bert_shortcut_name, args.gcn_hidden_feats, args.num_classes).to(device)
    elif args.model_architecture == "CA_GCN_Conv1_enriched":
        if args.tablewise == False:
            model = CA_GCN_Conv1_enriched(args.bert_shortcut_name, args.gcn_hidden_feats, args.gcn_hidden_feats, args.num_classes, args.table_to_column, args.column_to_column, args.numStatsFeat_to_column).to(device)
        else:
            #model = CA_GCN_Tablewise(args.bert_shortcut_name, args.gcn_hidden_feats, args.num_classes).to(device)
            #model = CA_GAT_Tablewise(args.bert_shortcut_name, args.gcn_hidden_feats, args.gcn_hidden_feats, args.num_classes).to(device)
            print("There is no valid model!")
            exit()
    elif args.model_architecture == "CA_GCN_Conv3_enriched":
        if args.tablewise == False:
            model = CA_GCN_Conv3_enriched(args.bert_shortcut_name, args.gcn_hidden_feats, args.gcn_hidden_feats, args.num_classes).to(device)
        else:
            #model = CA_GCN_Tablewise(args.bert_shortcut_name, args.gcn_hidden_feats, args.num_classes).to(device)
            #model = CA_GAT_Tablewise(args.bert_shortcut_name, args.gcn_hidden_feats, args.gcn_hidden_feats, args.num_classes).to(device)
            print("There is no valid model!")
            exit()
    elif args.model_architecture == "CA_GAT_enriched":
        if args.tablewise == False:
            model = CA_GAT_enriched(args.bert_shortcut_name, args.gcn_hidden_feats, args.gcn_hidden_feats, args.num_classes, args.table_to_column, args.column_to_column, args.numStatsFeat_to_column).to(device)
        else:
            #model = CA_GCN_Tablewise(args.bert_shortcut_name, args.gcn_hidden_feats, args.num_classes).to(device)
            #model = CA_GAT_Tablewise(args.bert_shortcut_name, args.gcn_hidden_feats, args.gcn_hidden_feats, args.num_classes).to(device)
            print("There is no valid model!")
            exit()
    elif args.model_architecture == "CA_GCN_Conv1_enriched_no_bert_for_nums":
        if args.tablewise == False:
            model = CA_GCN_Conv1_enriched_no_bert_for_nums(args.bert_shortcut_name, args.gcn_hidden_feats, args.gcn_hidden_feats, args.num_classes).to(device)
        else:
            #model = CA_GCN_Tablewise(args.bert_shortcut_name, args.gcn_hidden_feats, args.num_classes).to(device)
            #model = CA_GAT_Tablewise(args.bert_shortcut_name, args.gcn_hidden_feats, args.gcn_hidden_feats, args.num_classes).to(device)
            print("There is no valid model!")
            exit()
    elif args.model_architecture == "CA_GAT_enriched_shadow_nums":
        if args.tablewise == False:
            model = CA_GAT_enriched_shadow_nums(args.bert_shortcut_name, args.gcn_hidden_feats, args.gcn_hidden_feats, args.num_classes, args.one_etype).to(device)
        else:
            #model = CA_GCN_Tablewise(args.bert_shortcut_name, args.gcn_hidden_feats, args.num_classes).to(device)
            #model = CA_GAT_Tablewise(args.bert_shortcut_name, args.gcn_hidden_feats, args.gcn_hidden_feats, args.num_classes).to(device)
            print("There is no valid model!")
            exit()
            
    # Load the datasets for training & validation
    if args.data_corpus == "SportsTables":
        if args.model_architecture in ["CA_GCN", "CA_GCN_Conv1", "CA_GAT"]:
            if table_graph_representation == "columns":
                train_dataset = SportsTablesDGLDataset(tokenizer=tokenizer, max_length=args.max_length, device=device, sport_domains=sport_domains, random_state=args.random_seed, split="train", shuffle_cols=args.shuffle_cols)
                valid_dataset = SportsTablesDGLDataset(tokenizer=tokenizer, max_length=args.max_length, device=device, sport_domains=sport_domains, random_state=args.random_seed, split="valid", shuffle_cols=args.shuffle_cols)
                train_dataloader = GraphDataLoader(train_dataset, batch_size=args.batch_size, drop_last=False)
                valid_dataloader = GraphDataLoader(valid_dataset, batch_size=args.batch_size, drop_last=False)
            elif table_graph_representation == "columns+table_name":
                train_dataset = SportsTablesDGLDataset_with_table_name(tokenizer=tokenizer, max_length=args.max_length, device=device, sport_domains=sport_domains, random_state=args.random_seed, split="train", shuffle_cols=args.shuffle_cols)
                valid_dataset = SportsTablesDGLDataset_with_table_name(tokenizer=tokenizer, max_length=args.max_length, device=device, sport_domains=sport_domains, random_state=args.random_seed, split="valid", shuffle_cols=args.shuffle_cols)
                train_dataloader = GraphDataLoader(train_dataset, batch_size=args.batch_size, drop_last=False)
                valid_dataloader = GraphDataLoader(valid_dataset, batch_size=args.batch_size, drop_last=False)
        elif args.model_architecture in ["CA_GCN_Conv1_enriched", "CA_GAT_enriched"]:
            if table_graph_representation == "enriched":
                train_dataset = SportsTablesDGLDataset_enriched(tokenizer=tokenizer, max_length=args.max_length, device=device, sport_domains=sport_domains, random_state=args.random_seed, split="train", shuffle_cols=args.shuffle_cols, table_to_column=args.table_to_column, column_to_column=args.column_to_column, numStatsFeat_to_column=args.numStatsFeat_to_column)
                train_dataset.to_device()
                
                valid_dataset = SportsTablesDGLDataset_enriched(tokenizer=tokenizer, max_length=args.max_length, device=device, sport_domains=sport_domains, random_state=args.random_seed, split="valid", shuffle_cols=args.shuffle_cols, table_to_column=args.table_to_column, column_to_column=args.column_to_column, numStatsFeat_to_column=args.numStatsFeat_to_column)
                valid_dataset.to_device()
                
                train_dataloader = GraphDataLoader(train_dataset, batch_size=args.batch_size, drop_last=False)
                valid_dataloader = GraphDataLoader(valid_dataset, batch_size=args.batch_size, drop_last=False)
            elif table_graph_representation == "enriched_column_names":
                train_dataset = SportsTablesDGLDataset_enriched_column_names(tokenizer=tokenizer, max_length=args.max_length, device=device, sport_domains=sport_domains, random_state=args.random_seed, split="train", shuffle_cols=args.shuffle_cols, table_to_column=args.table_to_column, column_to_column=args.column_to_column, numStatsFeat_to_column=args.numStatsFeat_to_column)
                train_dataset.to_device()
                
                valid_dataset = SportsTablesDGLDataset_enriched_column_names(tokenizer=tokenizer, max_length=args.max_length, device=device, sport_domains=sport_domains, random_state=args.random_seed, split="valid", shuffle_cols=args.shuffle_cols, table_to_column=args.table_to_column, column_to_column=args.column_to_column, numStatsFeat_to_column=args.numStatsFeat_to_column)
                valid_dataset.to_device()
                
                train_dataloader = GraphDataLoader(train_dataset, batch_size=args.batch_size, drop_last=False)
                valid_dataloader = GraphDataLoader(valid_dataset, batch_size=args.batch_size, drop_last=False)
            elif table_graph_representation == "enriched_gpt_column_names":
                train_dataset = SportsTablesDGLDataset_enriched_column_names(tokenizer=tokenizer, max_length=args.max_length, device=device, sport_domains=sport_domains, random_state=args.random_seed, split="train", shuffle_cols=args.shuffle_cols, table_to_column=args.table_to_column, column_to_column=args.column_to_column, numStatsFeat_to_column=args.numStatsFeat_to_column, gpt_gen_colnames=True)
                train_dataset.to_device()
                
                valid_dataset = SportsTablesDGLDataset_enriched_column_names(tokenizer=tokenizer, max_length=args.max_length, device=device, sport_domains=sport_domains, random_state=args.random_seed, split="valid", shuffle_cols=args.shuffle_cols, table_to_column=args.table_to_column, column_to_column=args.column_to_column, numStatsFeat_to_column=args.numStatsFeat_to_column, gpt_gen_colnames=True)
                valid_dataset.to_device()
                
                train_dataloader = GraphDataLoader(train_dataset, batch_size=args.batch_size, drop_last=False)
                valid_dataloader = GraphDataLoader(valid_dataset, batch_size=args.batch_size, drop_last=False)
        elif args.model_architecture in ["CA_GAT_enriched_shadow_nums"]:
            if table_graph_representation == "enriched_shadow_nums":
                train_dataset = SportsTablesDGLDataset_enriched_shadow_num_nodes(tokenizer=tokenizer, max_length=args.max_length, device=device, sport_domains=sport_domains, random_state=args.random_seed, split="train", shuffle_cols=args.shuffle_cols, one_etype=args.one_etype)
                train_dataset.to_device()
                
                valid_dataset = SportsTablesDGLDataset_enriched_shadow_num_nodes(tokenizer=tokenizer, max_length=args.max_length, device=device, sport_domains=sport_domains, random_state=args.random_seed, split="valid", shuffle_cols=args.shuffle_cols, one_etype=args.one_etype)
                valid_dataset.to_device()
                
                train_dataloader = GraphDataLoader(train_dataset, batch_size=args.batch_size, drop_last=False)
                valid_dataloader = GraphDataLoader(valid_dataset, batch_size=args.batch_size, drop_last=False)
    elif args.data_corpus == "GitTables":
        if args.model_architecture in ["CA_GCN", "CA_GCN_Conv1", "CA_GAT"]:
            pass
        elif args.model_architecture in ["CA_GCN_Conv1_enriched", "CA_GCN_Conv3_enriched", "CA_GAT_enriched"]:
            if table_graph_representation == "enriched":
                train_dataset = GitTablesDGLDataset_enriched(tokenizer=tokenizer, max_length=args.max_length, device=device, random_state=args.random_seed, split="train", force_reload=False, semantic_sim_score=sem_type_sim_score)
                train_dataset.to_device()
                # TODO: This is a fast solution to fix the dtypes. It is already fixed in the dataset class. But I want to use the tmp file right now withou rebuild the dataset
                for graph in train_dataset:
                    graph.nodes["num_feature_node"].data["data_tensor"] = graph.nodes["num_feature_node"].data["data_tensor"].to(torch.float64)
                valid_dataset = GitTablesDGLDataset_enriched(tokenizer=tokenizer, max_length=args.max_length, device=device, random_state=args.random_seed, split="valid", force_reload=False, semantic_sim_score=sem_type_sim_score)
                valid_dataset.to_device()
                # TODO: This is a fast solution to fix the dtypes. It is already fixed in the dataset class. But I want to use the tmp file right now withou rebuild the dataset
                for graph in valid_dataset:
                    graph.nodes["num_feature_node"].data["data_tensor"] = graph.nodes["num_feature_node"].data["data_tensor"].to(torch.float64)
                train_dataloader = GraphDataLoader(train_dataset, batch_size=args.batch_size, drop_last=False)
                valid_dataloader = GraphDataLoader(valid_dataset, batch_size=args.batch_size, drop_last=False)
        elif args.model_architecture in ["CA_GCN_Conv1_enriched_no_bert_for_nums"]:
            if table_graph_representation == "enriched_sep":
                train_dataset = GitTablesDGLDataset_enriched_sep_col_dtype_nodes(tokenizer=tokenizer, max_length=args.max_length, device=device, random_state=args.random_seed, split="train", force_reload=False)
                train_dataset.to_device()

                valid_dataset = GitTablesDGLDataset_enriched_sep_col_dtype_nodes(tokenizer=tokenizer, max_length=args.max_length, device=device, random_state=args.random_seed, split="valid", force_reload=False)
                valid_dataset.to_device()
                
                train_dataloader = GraphDataLoader(train_dataset, batch_size=args.batch_size, drop_last=False)
                valid_dataloader = GraphDataLoader(valid_dataset, batch_size=args.batch_size, drop_last=False)
        elif args.model_architecture in ["CA_GAT_enriched_shadow_nums"]:
            if table_graph_representation == "enriched_shadow_nums":
                train_dataset = GitTablesDGLDataset_enriched_shadow_num_nodes(tokenizer=tokenizer, max_length=args.max_length, device=device, random_state=args.random_seed, split="train", force_reload=False, semantic_sim_score=sem_type_sim_score)
                train_dataset.to_device()
                
                valid_dataset = GitTablesDGLDataset_enriched_shadow_num_nodes(tokenizer=tokenizer, max_length=args.max_length, device=device, random_state=args.random_seed, split="valid", force_reload=False, semantic_sim_score=sem_type_sim_score)
                valid_dataset.to_device()
                
                train_dataloader = GraphDataLoader(train_dataset, batch_size=args.batch_size, drop_last=False)
                valid_dataloader = GraphDataLoader(valid_dataset, batch_size=args.batch_size, drop_last=False)
                
    
    # Initiate optimizer and loss function
    # TODO: specify the parameters which we want to optimize during the training
    # maybe exclude weights of bert here
    t_total = len(train_dataloader) * args.num_train_epochs
    # no_decay = ["bias", "LayerNorm.weight"]
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [
    #             p for n, p in model.named_parameters()
    #             if not any(nd in n for nd in no_decay)
    #         ],
    #         "weight_decay":
    #         0.0
    #     },
    #     {
    #         "params": [
    #             p for n, p in model.named_parameters()
    #             if any(nd in n for nd in no_decay)
    #         ],
    #         "weight_decay":
    #         0.0
    #     },
    # ]
    # Specify which layers to fine-tune

    # Freeze lower layers
    # for param in model.bert.embeddings.parameters():
    #     param.requires_grad = False

    for i, layer in enumerate(model.bert.encoder.layer):
        if i < args.layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
    # Specify which layers to fine-tune
    #fine_tuned_layers = ['num_feat_linear1', 'conv1']
    # use all non bert layer for finetuning
    fine_tuned_layers = list(set([layer[0].split(".")[0] for layer in model.named_parameters() if "bert" not in layer[0]]))

    # Separate the parameters of the model into different groups
    param_groups = [
        {"params": getattr(model, layer).parameters()} for layer in fine_tuned_layers
    ]

    # Add BERT's last layer to the parameter groups
    #param_groups.append({"params": model.bert.embeddings.parameters()})
    #TODO: I think we have to add the pooler layers to the trainable parameters:
    # In transformer-based models like BERT, the pooler layer plays a crucial role. 
    # It takes the hidden state of the first token (usually the [CLS] token) in the final layer and applies a dense layer to it. 
    # The output of this layer is then used for classification tasks
    param_groups.append({"params": model.bert.pooler.parameters()})
    
    #optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    #optimizer = AdamW(param_groups, lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)
    loss_fn = CrossEntropyLoss()
    
    # variables to store best results on val dataset
    best_vl_loss = 1000000
    best_vl_micro_f1 = -1
    best_vl_macro_f1 = -1
    loss_info_list = []
    
    ### early stopping criteriy
    early_stopp_counter = 0
    early_stopp_patience = 300
    
    # Training
    try:
        for epoch in trange(args.num_train_epochs, desc="Epoch"):
            # start time of the current epoch
            t1 = time()
            
            # switch the model to train mode
            model.train()
            
            tr_loss = 0.
            tr_pred_list = []
            tr_true_list = []
            
            vl_loss = 0.
            vl_pred_list = []
            vl_true_list = []
            
            # Training
            for batch_idx, batch in enumerate(train_dataloader):
                # logging.debug(f"Batch_idx: {batch_idx}")
                # logging.debug(f"Batch: {batch}")
                # no self loop for enriched because this is a heterogenous graph
                if table_graph_representation not in ["enriched", "enriched_sep", "enriched_shadow_nums", "enriched_column_names", "enriched_gpt_column_names"]:
                    batch = dgl.add_self_loop(batch)
                logits = model(batch)
                #logging.debug(f"Logits: {logits}")
                if table_graph_representation == "columns":
                    tr_pred_list += logits.argmax(1).cpu().detach().numpy().tolist()
                    tr_true_list += batch.ndata["label_tensor"].cpu().detach().numpy().tolist()
                    loss = loss_fn(logits, batch.ndata["label_tensor"])
                elif table_graph_representation == "columns+table_name":
                    # delete last element of the output of the model, because these represents the table_name node
                    tr_pred_list += logits[:-1].argmax(1).cpu().detach().numpy().tolist()
                    tr_true_list += batch.ndata["label_tensor"][:-1].cpu().detach().numpy().tolist()
                    loss = loss_fn(logits[:-1], batch.ndata["label_tensor"][:-1])
                elif table_graph_representation in ["enriched","enriched_column_names", "enriched_gpt_column_names"]:
                    # this is with heterogenous graphs
                    tr_pred_list += logits.argmax(1).cpu().detach().numpy().tolist()
                    #logging.debug(f"TR_PRED: {tr_pred_list}")
                    tr_true_list += batch.nodes['column'].data["label_tensor"].cpu().detach().numpy().tolist()
                    #logging.debug(f"TR_TRUE: {tr_true_list}")
                    loss = loss_fn(logits, batch.nodes['column'].data["label_tensor"])
                    #logging.debug(f"Loss: {loss}")
                elif table_graph_representation == "enriched_sep":
                    # this is with heterogenous graphs
                    tr_pred_list += logits.argmax(1).cpu().detach().numpy().tolist()
                    tr_true_list += batch.nodes['num_column'].data["label_tensor"].cpu().detach().numpy().tolist()
                    loss = loss_fn(logits, batch.nodes['num_column'].data["label_tensor"])
                elif table_graph_representation == "enriched_shadow_nums":
                    # this is with heterogenous graphs
                    tr_pred_list += logits.argmax(1).cpu().detach().numpy().tolist()
                    #logging.debug(f"TR_PRED: {tr_pred_list}")
                    tr_true_list += batch.nodes['column'].data["label_tensor"].cpu().detach().numpy().tolist()
                    #logging.debug(f"TR_TRUE: {tr_true_list}")
                    loss = loss_fn(logits, batch.nodes['column'].data["label_tensor"])
                    #logging.debug(f"Loss: {loss}")
                    
                loss.backward()
                tr_loss += loss.item()
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                # Access and log the current learning rate
                current_lr = optimizer.param_groups[0]['lr']
                if batch_idx % 1000 == 0:
                    current = batch_idx * args.batch_size
                    if table_graph_representation == "columns":
                        logging.debug(f"F1 Macro: {f1_score(batch.ndata['label_tensor'].cpu().detach().numpy().tolist(), logits.argmax(1).cpu().detach().numpy().tolist(), average='macro')} "+
                            f"F1 Weighted: {f1_score(batch.ndata['label_tensor'].cpu().detach().numpy().tolist(), logits.argmax(1).cpu().detach().numpy().tolist(), average='weighted')} "+
                            f"[{current}/{len(train_dataloader)}]")
                    elif table_graph_representation == "columns+table_name":
                        # delete last element of the output of the model, because these represents the table_name node
                        logging.debug(f"F1 Macro: {f1_score(batch.ndata['label_tensor'][:-1].cpu().detach().numpy().tolist(), logits[:-1].argmax(1).cpu().detach().numpy().tolist(), average='macro')} "+
                            f"F1 Weighted: {f1_score(batch.ndata['label_tensor'][:-1].cpu().detach().numpy().tolist(), logits[:-1].argmax(1).cpu().detach().numpy().tolist(), average='weighted')} "+
                            f"[{current}/{len(train_dataloader)}]")
                    elif table_graph_representation in ["enriched","enriched_column_names", "enriched_gpt_column_names"]:
                        logging.debug(f"F1 Macro: {f1_score(batch.nodes['column'].data['label_tensor'].cpu().detach().numpy().tolist(), logits.argmax(1).cpu().detach().numpy().tolist(), average='macro')} "+
                            f"F1 Weighted: {f1_score(batch.nodes['column'].data['label_tensor'].cpu().detach().numpy().tolist(), logits.argmax(1).cpu().detach().numpy().tolist(), average='weighted')} "+
                            f"[{current}/{len(train_dataloader)}]")
                        logging.debug(f"Learning-Rate: {current_lr}")
                    elif table_graph_representation == "enriched_sep":
                        logging.debug(f"F1 Macro: {f1_score(batch.nodes['num_column'].data['label_tensor'].cpu().detach().numpy().tolist(), logits.argmax(1).cpu().detach().numpy().tolist(), average='macro')} "+
                            f"F1 Weighted: {f1_score(batch.nodes['num_column'].data['label_tensor'].cpu().detach().numpy().tolist(), logits.argmax(1).cpu().detach().numpy().tolist(), average='weighted')} "+
                            f"[{current}/{len(train_dataloader)}]")
                        logging.debug(f"Learning-Rate: {current_lr}")
                
            tr_loss /= (len(train_dataloader) / args.batch_size)
            
            tr_micro_f1 = f1_score(tr_true_list, tr_pred_list, average="micro")
            tr_macro_f1 = f1_score(tr_true_list, tr_pred_list, average="macro")
            tr_weighted_f1 = f1_score(tr_true_list, tr_pred_list, average="weighted")
            tr_class_f1 = f1_score(tr_true_list, tr_pred_list, average=None, labels=np.arange(args.num_classes))
            
            # Validation
            model.eval()
            for batch_idx, batch in enumerate(valid_dataloader):
                if table_graph_representation not in ["enriched", "enriched_sep", "enriched_shadow_nums", "enriched_column_names", "enriched_gpt_column_names"]:
                    batch = dgl.add_self_loop(batch)
                logits = model(batch)
                
                if table_graph_representation == "columns":
                    vl_pred_list += logits.argmax(1).cpu().detach().numpy().tolist()
                    vl_true_list += batch.ndata["label_tensor"].cpu().detach().numpy().tolist()
                    
                    loss = loss_fn(logits, batch.ndata["label_tensor"])  
                elif table_graph_representation == "columns+table_name":
                    # delete last element of the output of the model, because these represents the table_name node
                    vl_pred_list += logits[:-1].argmax(1).cpu().detach().numpy().tolist()
                    vl_true_list += batch.ndata["label_tensor"][:-1].cpu().detach().numpy().tolist()
                    
                    loss = loss_fn(logits[:-1], batch.ndata["label_tensor"][:-1])
                elif table_graph_representation in ["enriched","enriched_column_names", "enriched_gpt_column_names"]:
                    vl_pred_list += logits.argmax(1).cpu().detach().numpy().tolist()
                    vl_true_list += batch.nodes['column'].data["label_tensor"].cpu().detach().numpy().tolist()
                    
                    loss = loss_fn(logits, batch.nodes['column'].data["label_tensor"])
                elif table_graph_representation == "enriched_sep":
                    vl_pred_list += logits.argmax(1).cpu().detach().numpy().tolist()
                    vl_true_list += batch.nodes['num_column'].data["label_tensor"].cpu().detach().numpy().tolist()
                    
                    loss = loss_fn(logits, batch.nodes['num_column'].data["label_tensor"])  
                elif table_graph_representation == "enriched_shadow_nums":
                    # this is with heterogenous graphs
                    vl_pred_list += logits.argmax(1).cpu().detach().numpy().tolist()
                    vl_true_list += batch.nodes['column'].data["label_tensor"].cpu().detach().numpy().tolist()
                    
                    loss = loss_fn(logits, batch.nodes['column'].data["label_tensor"])          
                    
                
                vl_loss += loss.item()
            
            vl_loss /= (len(valid_dataloader) / args.batch_size)
        
            vl_micro_f1 = f1_score(vl_true_list, vl_pred_list, average='micro')
            vl_macro_f1 = f1_score(vl_true_list, vl_pred_list, average='macro')
            vl_weighted_f1 = f1_score(vl_true_list, vl_pred_list, average="weighted")
            vl_class_f1 = f1_score(vl_true_list, vl_pred_list, average=None, labels=np.arange(args.num_classes))
            
            # store best micro f1 on validation set
            if vl_micro_f1 > best_vl_micro_f1:
                best_vl_micro_f1 = vl_micro_f1
                
                model_savepath = tag_name+f"_best_micro_f1.pt"
                torch.save(model.state_dict(), model_savepath)
            if vl_macro_f1 > best_vl_macro_f1:
                best_vl_macro_f1 = vl_macro_f1
                
                model_savepath = tag_name+f"_best_marco_f1.pt"
                torch.save(model.state_dict(), model_savepath)
                
            loss_info_list.append([
                tr_loss, tr_macro_f1, tr_micro_f1, tr_weighted_f1, vl_loss, vl_macro_f1, vl_micro_f1, vl_weighted_f1
            ])
            
            t2 = time()
            logging.debug(
                f"Epoch {epoch}: tr_loss={tr_loss:.7f} tr_macro_f1={tr_macro_f1:.4f} tr_micro_f1={tr_micro_f1:.4f} "+
                f"vl_loss={vl_loss:.7f} vl_macro_f1={vl_macro_f1:.4f} vl_micro_f1={vl_micro_f1:.4f} ({(t2 - t1):.2f} sec.)")
            
            ## early stopp
            if best_vl_loss > vl_loss:
                early_stopp_counter = 0
                best_vl_loss = vl_loss
            else:
                logging.debug(f"No improvement of loss on validation dataset during the epoch. Counter at: {early_stopp_counter}/{early_stopp_patience}")
                early_stopp_counter +=1
            if early_stopp_counter > early_stopp_patience:
                logging.debug(f"Early stopp at epoch: {epoch} / best_vl_loss: {best_vl_loss}")
                break
    except Exception as e:
        # logging.debug(batch_idx)
        # logging.debug(batch)
        logging.error(e, exc_info=True)
    
    loss_info_df = pd.DataFrame(loss_info_list,
                                    columns=[
                                        "tr_loss", "tr_f1_macro_f1",
                                        "tr_f1_micro_f1", "tr_f1_weighted_f1", "vl_loss",
                                        "vl_f1_macro_f1", "vl_f1_micro_f1", "vl_f1_weighted_f1"
                                    ])
    loss_info_df.to_csv(f"{tag_name}_loss_info.csv")
        