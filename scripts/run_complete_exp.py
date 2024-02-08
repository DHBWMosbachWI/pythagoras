# TODO: Run experminent with the model GAT_enriched
# TODO: Run experiment without updating the weights of the BERT Model
# TODO: Cut semantic types for which we have only one example in SportsTable.
# For example, there is football.team.championship_wins which only occurs ones. 
# If these types are in the test-set. The GCN Model always predicts the type soccer.player.name 
import argparse
import os
from dotenv import load_dotenv
load_dotenv(override=True)

# Define the cuda device if you want
os.environ['CUDA_VISIBLE_DEVICES']='0'
#os.environ['CUDA_LAUNCH_BLOCKING']='1'


#data_corpus = "GitTables"  # choice of ["SportsTables", "GitTables"]
data_corpus = "SportsTables" 

if data_corpus == "GitTables":
    num_classes = 219 # 104  # GitTables =>  219; SportsTables => 462
    max_length = 254 # SportTables: 254 / GitTables: 62
    learning_rate = 5e-5 # SportTables: 1e-5 / GitTables: 
    gcn_hidden_feats = 512 # SprotsTable => 512 / GitTables => 1024
    batch_size = 3
    #sport_domains_m = [["'baseball'", "'basketball'",
    #                   "'football'", "'hockey'", "'soccer'"]] # for training
    sport_domains_m =  [[]] # for validation
    layers_to_freeze = 4
elif data_corpus == "SportsTables":
    num_classes = 462  # GitTables =>  779; SportsTables => 462
    max_length = 254 # SportTables: 254 / GitTables: 62
    learning_rate = 1e-4 # SportTables: 1e-5 / GitTables: 
    gcn_hidden_feats = 512 # SprotsTable => 512 / GitTables => 1024
    batch_size = 1
    #sport_domains_m = [[]]
    #sport_domains_m = 'None'
    #sport_domains_m = [["'baseball'", "'basketball'",
    #                   "'football'", "'hockey'", "'soccer'"]]  # for training
    sport_domains_m = [
        ["'baseball', 'basketball', 'football', 'hockey', 'soccer'"]]  # for validation
    #sport_domains_m = [["'baseball'"],["'basketball'"], ["'football'"], ["'hockey'"], ["'soccer'"]]
    #sport_domains_m = ["'football'"]
    layers_to_freeze = 12

#model_architecture = "CA_GAT"
#model_architecture = "CA_GCN_Conv1"
model_architecture = "CA_GCN_Conv1_enriched"
#model_architecture = "CA_GCN_Conv1_enriched_no_bert_for_nums"
#model_architecture = "CA_GCN_Conv3_enriched"
#model_architecture = "CA_GAT_enriched"
#model_architecture = "CA_GAT_enriched_shadow_nums"
#table_graph_representation = "columns+table_name"
#table_graph_representation = "enriched"
#table_graph_representation = "enriched_sep" # choice of ["columns", enriched]
#table_graph_representation = "enriched_column_names" # choice of ["columns", enriched]
table_graph_representation = "enriched_gpt_column_names"

num_train_epochs = 500

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    for sport_domains in sport_domains_m:
        for random_state in [1]:

            # # singlewise training
            # os.system(
            #     f"{os.environ['PYTHON']} train_gcn.py --model_architecture {model_architecture} --layers_to_freeze {layers_to_freeze} --table_graph_representation {table_graph_representation} --num_classes {num_classes} --gcn_hidden_feats {gcn_hidden_feats} --data_corpus {data_corpus} --learning_rate {learning_rate} --max_length {max_length} --num_train_epochs {num_train_epochs} --batch_size {batch_size} --random_seed {random_state} --sport_domains {' '.join(sport_domains)} --table_to_column --column_to_column --numStatsFeat_to_column"
            # )
            # without tablename node connection
            # os.system(
            #     f"{os.environ['PYTHON']} train_gcn.py --model_architecture {model_architecture} --layers_to_freeze {layers_to_freeze} --table_graph_representation {table_graph_representation} --num_classes {num_classes} --gcn_hidden_feats {gcn_hidden_feats} --data_corpus {data_corpus} --learning_rate {learning_rate} --max_length {max_length} --num_train_epochs {num_train_epochs} --batch_size {batch_size} --random_seed {random_state} --sport_domains {' '.join(sport_domains)} --column_to_column --numStatsFeat_to_column"
            # )
            # without textuel column to numerical column connection
            # os.system(
            #     f"{os.environ['PYTHON']} train_gcn.py --model_architecture {model_architecture} --layers_to_freeze {layers_to_freeze} --table_graph_representation {table_graph_representation} --num_classes {num_classes} --gcn_hidden_feats {gcn_hidden_feats} --data_corpus {data_corpus} --learning_rate {learning_rate} --max_length {max_length} --num_train_epochs {num_train_epochs} --batch_size {batch_size} --random_seed {random_state} --sport_domains {' '.join(sport_domains)} --table_to_column --numStatsFeat_to_column"
            # )
            # wihtout numStaFeat node to column connection
            # os.system(
            #     f"{os.environ['PYTHON']} train_gcn.py --model_architecture {model_architecture} --layers_to_freeze {layers_to_freeze} --table_graph_representation {table_graph_representation} --num_classes {num_classes} --gcn_hidden_feats {gcn_hidden_feats} --data_corpus {data_corpus} --learning_rate {learning_rate} --max_length {max_length} --num_train_epochs {num_train_epochs} --batch_size {batch_size} --random_seed {random_state} --sport_domains {' '.join(sport_domains)} --table_to_column --column_to_column"
            # )
            # wihtout all connections
            # os.system(
            #     f"{os.environ['PYTHON']} train_gcn.py --model_architecture {model_architecture} --layers_to_freeze {layers_to_freeze} --table_graph_representation {table_graph_representation} --num_classes {num_classes} --gcn_hidden_feats {gcn_hidden_feats} --data_corpus {data_corpus} --learning_rate {learning_rate} --max_length {max_length} --num_train_epochs {num_train_epochs} --batch_size {batch_size} --random_seed {random_state} --sport_domains {' '.join(sport_domains)}"
            # )
            # only num Feat Stats
            # os.system(
            #     f"{os.environ['PYTHON']} train_gcn.py --model_architecture {model_architecture} --layers_to_freeze {layers_to_freeze} --table_graph_representation {table_graph_representation} --num_classes {num_classes} --gcn_hidden_feats {gcn_hidden_feats} --data_corpus {data_corpus} --learning_rate {learning_rate} --max_length {max_length} --num_train_epochs {num_train_epochs} --batch_size {batch_size} --random_seed {random_state} --sport_domains {' '.join(sport_domains)} --numStatsFeat_to_column"
            # )
            # singlewise validation
            ## SportTables
            os.system(
                f"{os.environ['PYTHON']} validate_gcn.py --data_corpus {data_corpus} --model_architecture {model_architecture} --table_graph_representation {table_graph_representation} --model_name single_bert-base-uncased_ts0.2_ml{max_length}_nc{num_classes}_bs{batch_size}_rs{random_state}_lr{learning_rate}_hf{gcn_hidden_feats}_{sport_domains}_scFalse_lf{layers_to_freeze}_ttcTrue_ctcTrue_nSFtcTrue_best_micro_f1.pt"
            )
            ## GitTables
            # os.system(
            #     f"{os.environ['PYTHON']} validate_gcn.py --data_corpus {data_corpus} --model_architecture {model_architecture} --table_graph_representation {table_graph_representation} --model_name single_bert-base-uncased_ts0.2_ml{max_length}_nc{num_classes}_bs{batch_size}_rs{random_state}_lr{learning_rate}_hf{gcn_hidden_feats}_{sport_domains}_scFalse_lf{layers_to_freeze}_ss0.7_best_micro_f1.pt"
            # )

            # # tablewise training
            # os.system(
            #     f"{os.environ['PYTHON']} train_gcn.py --model_architecture {model_architecture} --learning_rate {learning_rate} --tablewise --max_length {10} --num_train_epochs {num_train_epochs} --batch_size {batch_size} --random_seed {random_state} --sport_domains {' '.join(sport_domains)}"
            # )
            # #tablewise validation
            # os.system(
            #     f"{os.environ['PYTHON']} validate_gcn.py --model_architecture {model_architecture} --model_name table_bert-base-uncased_ts0.2_ml{10}_nc462_bs1_rs{random_state}_lr{learning_rate}_hf512_{sport_domains}_scFalse_best_micro_f1.pt"
            # )

            # ## tablewise training shuffle cols
            # os.system(
            #     f"{os.environ['PYTHON']} train_gcn.py --model_architecture {model_architecture} --learning_rate {learning_rate} --tablewise --shuffle_cols --max_length {10} --num_train_epochs {num_train_epochs} --batch_size {batch_size} --random_seed {random_state} --sport_domains {' '.join(sport_domains)}"
            # )
            # # ## tablewise validation shuffle cols
            # os.system(
            #     f"{os.environ['PYTHON']} validate_gcn.py --model_architecture {model_architecture} --model_name table_bert-base-uncased_ts0.2_ml{10}_nc462_bs1_rs{random_state}_lr{learning_rate}_hf512_{sport_domains}_scTrue_best_micro_f1.pt"
            # )
