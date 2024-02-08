import sys
sys.path.append("..")
from ast import literal_eval
from transformers import BertTokenizer
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.dataloading import GraphDataLoader
import dgl
from data_loader.SportsDB_data_loader import SportsTablesDGLDataset, SportsTablesDGLDataset_with_table_name, SportsTablesDGLDataset_enriched, SportsTablesDGLDataset_enriched_column_names
from data_loader.GitTables_data_loader import GitTablesDGLDataset_enriched, GitTablesDGLDataset_enriched_shadow_num_nodes
from os.path import join
import torch
import argparse
import logging
from model.gcn import CA_GCN, CA_GCN_Tablewise, CA_GCN_Conv1, CA_GCN_Tablewise_Conv1, CA_GAT, CA_GAT_Tablewise, CA_GCN_Conv1_enriched, CA_GAT_enriched, CA_GAT_enriched_shadow_nums



def parse_model_params(model_name: str, data_corpus:str):
    model_name_splitted = model_name.split("_")
    tablewise = True if model_name_splitted[0] == "table" else False
    bert_shortcut_name = model_name_splitted[1]
    test_size = float(model_name_splitted[2].split("ts")[1])
    max_length = int(model_name_splitted[3].split("ml")[1])
    num_classes = int(model_name_splitted[4].split("nc")[1])
    batch_size = int(model_name_splitted[5].split("bs")[1])
    random_state = int(model_name_splitted[6].split("rs")[1])
    learning_rate = float(model_name_splitted[7].split("lr")[1])
    gcn_hidden_feats = int(model_name_splitted[8].split("hf")[1])
    # For GitTables we do not have sportdomains
    # TODO: Hacky solution right now
    if model_name_splitted[9] == "[]":
        sport_domains = "None"
    else:
        sport_domains = literal_eval(model_name_splitted[9])
    shuffle_cols = literal_eval(model_name_splitted[10].split("sc")[1])
    layers_freezed = int(model_name_splitted[11].split("lf")[1])
    if data_corpus == "GitTables":
        sem_type_sim_score = float(model_name_splitted[12].split("ss")[1])
    
        return tablewise, bert_shortcut_name, test_size, max_length, num_classes, batch_size, random_state, learning_rate, gcn_hidden_feats, sport_domains, shuffle_cols, layers_freezed, sem_type_sim_score
    
    if data_corpus == "SportsTables":
        table_to_column = False if model_name_splitted[12].split("ttc")[1] == "False" else True
        column_to_column = False if model_name_splitted[13].split("ctc")[1] == "False" else True
        numStatsFeat_to_column = False if model_name_splitted[14].split("nSFtc")[1] == "False" else True

        return tablewise, bert_shortcut_name, test_size, max_length, num_classes, batch_size, random_state, learning_rate, gcn_hidden_feats, sport_domains, shuffle_cols, layers_freezed, table_to_column, column_to_column, numStatsFeat_to_column 


if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default="bert-base-uncased_0.2_510_462_1_1_5e-05_512_best_marco_f1.pt",
        type=str,
        help="The name of the model in the output folder."
    )
    parser.add_argument(
        "--model_architecture",
        default="CA_GCN",
        type=str,
        help="The name of the modelclass with which the training should be executed",
        choices=["CA_GCN", "CA_GCN_Conv1", "CA_GAT", "CA_GCN_Conv1_enriched", "CA_GAT_enriched_shadow_nums"]
    )
    parser.add_argument(
        "--table_graph_representation",
        default="columns",
        type=str,
        help="The graph representation for a table that shouls be used. For example only use columns as graph nodes. Or use column nodes and an additional node for the table_name and so on.",
        choices=["columns", "columns+table_name", "enriched", "enriched_shadow_nums", "enriched_column_names", "enriched_gpt_column_names"]
    )
    parser.add_argument(
        "--data_corpus",
        default="SportsTables",
        choices=["SportsTables", "GitTables"]
    )

    #model_name = "bert-base-uncased_0.2_510_462_1_1_5e-05_512_best_marco_f1.pt"

    # Parsing all given arguments
    args = parser.parse_args()
    table_graph_representation = args.table_graph_representation
    print(args.model_name)

    if args.data_corpus == "GitTables":
        args.tablewise, args.bert_shortcut_name, args.test_size, args.max_length, args.num_classes, args.batch_size, args.random_state, args.learning_rate, args.gcn_hidden_feats, args.sport_domains, args.shuffle_cols, args.layers_freezed, args.sem_type_sim_score = parse_model_params(
            args.model_name, args.data_corpus)
    if args.data_corpus == "SportsTables":
        args.tablewise, args.bert_shortcut_name, args.test_size, args.max_length, args.num_classes, args.batch_size, args.random_state, args.learning_rate, args.gcn_hidden_feats, args.sport_domains, args.shuffle_cols, args.layers_freezed, args.table_to_column, args.column_to_column, args.numStatsFeat_to_column = parse_model_params(
            args.model_name, args.data_corpus)

    print("args={}".format(json.dumps(vars(args))))

    # logging set-up
    logging.basicConfig(filename=join("..", "output",args.data_corpus, f"{args.model_architecture}", table_graph_representation, "classification_report",
                        f"{args.model_name}.log"), filemode="w", format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    logging.debug("args={}".format(json.dumps(vars(args))))
    logging.debug(f"Device: {device}")
    logging.debug(f"Model tag name: {args.model_name}")

    #model_name = args.model_name
    #sport_domains = args.sport_domains

    # Load tokenizer and load the NN model
    tokenizer = BertTokenizer.from_pretrained(args.bert_shortcut_name)

    if args.model_architecture == "CA_GCN":
        if args.tablewise == False:
            model = CA_GCN(args.bert_shortcut_name,
                           args.gcn_hidden_feats, args.num_classes).to(device)
        else:
            model = CA_GCN_Tablewise(
                args.bert_shortcut_name, args.gcn_hidden_feats, args.num_classes).to(device)
    elif args.model_architecture == "CA_GCN_Conv1":
        if args.tablewise == False:
            model = CA_GCN_Conv1(
                args.bert_shortcut_name, args.gcn_hidden_feats, args.num_classes).to(device)
        else:
            #model = CA_GCN_Tablewise(args.bert_shortcut_name, args.gcn_hidden_feats, args.num_classes).to(device)
            model = CA_GCN_Tablewise_Conv1(
                args.bert_shortcut_name, args.gcn_hidden_feats, args.num_classes).to(device)
    elif args.model_architecture == "CA_GAT":
        if args.tablewise == False:
            model = CA_GAT(args.bert_shortcut_name,
                           args.gcn_hidden_feats, args.num_classes).to(device)
        else:
            #model = CA_GCN_Tablewise(args.bert_shortcut_name, args.gcn_hidden_feats, args.num_classes).to(device)
            model = CA_GAT_Tablewise(
                args.bert_shortcut_name, args.gcn_hidden_feats, args.num_classes).to(device)
    elif args.model_architecture == "CA_GCN_Conv1_enriched":
        if args.tablewise == False:
            model = CA_GCN_Conv1_enriched(
                args.bert_shortcut_name, args.gcn_hidden_feats, args.gcn_hidden_feats, args.num_classes, args.table_to_column, args.column_to_column, args.numStatsFeat_to_column).to(device)
        else:
            pass
            #model = CA_GCN_Tablewise(args.bert_shortcut_name, args.gcn_hidden_feats, args.num_classes).to(device)
            #model = CA_GCN_Tablewise_Conv1(args.bert_shortcut_name, args.gcn_hidden_feats, args.num_classes).to(device)
    elif args.model_architecture == "CA_GAT_enriched":
        if args.tablewise == False:
            model = CA_GAT_enriched(
                args.bert_shortcut_name, args.gcn_hidden_feats, args.gcn_hidden_feats, args.num_classes, args.table_to_column, args.column_to_column, args.numStatsFeat_to_column).to(device)
        else:
            pass
            #model = CA_GCN_Tablewise(args.bert_shortcut_name, args.gcn_hidden_feats, args.num_classes).to(device)
            #model = CA_GCN_Tablewise_Conv1(args.bert_shortcut_name, args.gcn_hidden_feats, args.num_classes).to(device)
    elif args.model_architecture == "CA_GAT_enriched_shadow_nums":
        if args.tablewise == False:
            model = CA_GAT_enriched_shadow_nums(
                args.bert_shortcut_name, args.gcn_hidden_feats, args.gcn_hidden_feats, args.num_classes).to(device)
        else:
            pass
    
    if args.data_corpus == "SportsTables":
        logging.debug(f'Loading model: {join("..", "output", args.data_corpus, f"{args.model_architecture}", table_graph_representation, f"single_bert-base-uncased_ts0.2_ml{args.max_length}_nc{args.num_classes}_bs{args.batch_size}_rs{args.random_state}_lr{args.learning_rate}_hf{args.gcn_hidden_feats}_{args.sport_domains}_sc{args.shuffle_cols}_lf{args.layers_freezed}_ttc{args.table_to_column}_ctc{args.column_to_column}_nSFtc{args.numStatsFeat_to_column}_best_micro_f1.pt")}')
        model.load_state_dict(torch.load(
            join("..", "output",args.data_corpus, f"{args.model_architecture}", table_graph_representation, f"single_bert-base-uncased_ts0.2_ml{args.max_length}_nc{args.num_classes}_bs{args.batch_size}_rs{args.random_state}_lr{args.learning_rate}_hf{args.gcn_hidden_feats}_{args.sport_domains}_sc{args.shuffle_cols}_lf{args.layers_freezed}_ttc{args.table_to_column}_ctc{args.column_to_column}_nSFtc{args.numStatsFeat_to_column}_best_micro_f1.pt")))

    if args.data_corpus == "GitTables":
        logging.debug(f'Loading model: {join("..", "output", args.data_corpus, f"{args.model_architecture}", table_graph_representation, f"single_bert-base-uncased_ts0.2_ml{args.max_length}_nc{args.num_classes}_bs{args.batch_size}_rs{args.random_state}_lr{args.learning_rate}_hf{args.gcn_hidden_feats}_{args.sport_domains}_sc{args.shuffle_cols}_lf{args.layers_freezed}_ss{args.sem_type_sim_score}_best_micro_f1.pt")}')
        model.load_state_dict(torch.load(
            join("..", "output",args.data_corpus, f"{args.model_architecture}", table_graph_representation, f"single_bert-base-uncased_ts0.2_ml{args.max_length}_nc{args.num_classes}_bs{args.batch_size}_rs{args.random_state}_lr{args.learning_rate}_hf{args.gcn_hidden_feats}_{args.sport_domains}_sc{args.shuffle_cols}_lf{args.layers_freezed}_ss{args.sem_type_sim_score}_best_micro_f1.pt")))

    # Load the datasets for training & validation
    if args.data_corpus == "SportsTables":
        if table_graph_representation == "columns":
            test_dataset = SportsTablesDGLDataset(tokenizer=tokenizer, max_length=args.max_length, device=device,
                                                sport_domains=args.sport_domains, random_state=args.random_state, split="test", shuffle_cols=args.shuffle_cols)
        elif table_graph_representation == "columns+table_name":
            test_dataset = SportsTablesDGLDataset_with_table_name(tokenizer=tokenizer, max_length=args.max_length, device=device,
                                                                sport_domains=args.sport_domains, random_state=args.random_state, split="test", shuffle_cols=args.shuffle_cols)
        elif table_graph_representation == "enriched":
            test_dataset = SportsTablesDGLDataset_enriched(tokenizer=tokenizer, max_length=args.max_length, device=device,
                                                        sport_domains=args.sport_domains, random_state=args.random_state, split="test", shuffle_cols=args.shuffle_cols, 
                                                        table_to_column=args.table_to_column, column_to_column=args.column_to_column, numStatsFeat_to_column=args.numStatsFeat_to_column)
        elif table_graph_representation == "enriched_column_names":
            test_dataset = SportsTablesDGLDataset_enriched_column_names(tokenizer=tokenizer, max_length=args.max_length, device=device,
                                                        sport_domains=args.sport_domains, random_state=args.random_state, split="test", shuffle_cols=args.shuffle_cols, 
                                                        table_to_column=args.table_to_column, column_to_column=args.column_to_column, numStatsFeat_to_column=args.numStatsFeat_to_column)
            test_dataset.to_device()
        elif table_graph_representation == "enriched_gpt_column_names":
            test_dataset = SportsTablesDGLDataset_enriched_column_names(tokenizer=tokenizer, max_length=args.max_length, device=device,
                                                        sport_domains=args.sport_domains, random_state=args.random_state, split="test", shuffle_cols=args.shuffle_cols, 
                                                        table_to_column=args.table_to_column, column_to_column=args.column_to_column, numStatsFeat_to_column=args.numStatsFeat_to_column, gpt_gen_colnames=True) 
            test_dataset.to_device()
    elif args.data_corpus == "GitTables":
        if table_graph_representation == "columns":
            pass
        elif table_graph_representation == "columns+table_name":
            pass
        elif table_graph_representation == "enriched":
            test_dataset = GitTablesDGLDataset_enriched(tokenizer=tokenizer, max_length=args.max_length, 
                                                        device=device, random_state=args.random_state, 
                                                        split="test", force_reload=False, shuffle_cols=args.shuffle_cols)
            test_dataset.to_device()
        elif table_graph_representation == "enriched_shadow_nums":
            test_dataset = GitTablesDGLDataset_enriched_shadow_num_nodes(tokenizer=tokenizer, max_length=args.max_length, 
                                                        device=device, random_state=args.random_state, 
                                                        split="test", force_reload=False, shuffle_cols=args.shuffle_cols)
            test_dataset.to_device()
    test_dataloader = GraphDataLoader(
        test_dataset, batch_size=args.batch_size, drop_last=False)

    # Validation
    model.eval()
    vl_pred_list = []
    vl_true_list = []
    vl_col_type_list = []
    for batch_idx, batch in enumerate(test_dataloader):
        if table_graph_representation not in  ["enriched", "enriched_shadow_nums", "enriched_column_names", "enriched_gpt_column_names"]:
            batch = dgl.add_self_loop(batch)

        logits = model(batch)

        if table_graph_representation == "columns":
            vl_pred_list += logits.argmax(1).cpu().detach().numpy().tolist()
            vl_true_list += batch.ndata["label_tensor"].cpu(
            ).detach().numpy().tolist()
        elif table_graph_representation == "columns+table_name":
            vl_pred_list += logits[:-
                                   1].argmax(1).cpu().detach().numpy().tolist()
            vl_true_list += batch.ndata["label_tensor"][:-
                                                        1].cpu().detach().numpy().tolist()
        elif table_graph_representation in ["enriched", "enriched_column_names", "enriched_gpt_column_names"]:
            vl_pred_list += logits.argmax(1).cpu().detach().numpy().tolist()
            vl_true_list += batch.nodes['column'].data["label_tensor"].cpu(
                ).detach().numpy().tolist()
            vl_col_type_list += batch.nodes['column'].data["col_type_tensor"].cpu(
                ).detach().numpy().tolist()
        elif table_graph_representation == "enriched_shadow_nums":
            vl_pred_list += logits.argmax(1).cpu().detach().numpy().tolist()
            vl_true_list += batch.nodes['column'].data["label_tensor"].cpu(
                ).detach().numpy().tolist()
            vl_col_type_list += batch.nodes['column'].data["col_type_tensor"].cpu(
                ).detach().numpy().tolist()

    # get label encoder to have the semantic types in the classification report
    if args.data_corpus == "SportsTables":
        from data_loader.SportsDB_data_loader import get_LabelEncoder
    elif args.data_corpus == "GitTables":
        from data_loader.GitTables_data_loader import get_LabelEncoder
    label_encoder = get_LabelEncoder()

    vl_pred_list = label_encoder.inverse_transform(vl_pred_list)
    vl_true_list = label_encoder.inverse_transform(vl_true_list)

    # save predictions 1:1
    with open(join("..", "output", args.data_corpus, f"{args.model_architecture}", table_graph_representation, "classification_report", f"{args.model_name}_predictions.json"), "w") as f:
        json.dump({"y_true": vl_true_list.tolist(), "y_pred": vl_pred_list.tolist(), "y_col_type": vl_col_type_list}, f)

    class_report = classification_report(
        vl_true_list, vl_pred_list, output_dict=True)

    with open(join("..", "output", args.data_corpus, f"{args.model_architecture}", table_graph_representation, "classification_report", f"{args.model_name}.json"), "w") as f:
        json.dump(class_report, f)
