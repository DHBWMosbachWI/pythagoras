import dgl
import torch
import torch.nn.functional as F

from transformers import BertModel, BertConfig
from functools import reduce
import operator
import numpy as np


class CA_GCN(torch.nn.Module):
    def __init__(self, bert_shortcut_name, hidden_feats, num_classes):
        super().__init__()
        
        # load the pre-trained bert language model
        self.bert = BertModel.from_pretrained(bert_shortcut_name)
        self.dropout = torch.nn.Dropout(self.bert.config.hidden_dropout_prob)
        
        # layers of the GNN
        self.conv1 = dgl.nn.GraphConv(in_feats=self.bert.config.hidden_size, out_feats=hidden_feats)
        self.conv2 = dgl.nn.GraphConv(in_feats=hidden_feats, out_feats=hidden_feats)
        self.conv3 = dgl.nn.GraphConv(in_feats=hidden_feats, out_feats=num_classes)
    
    def forward(self, graph):
        # foward pass of the bert model
        input_ids = graph.ndata["data_tensor"]
        
        output = self.bert(input_ids=input_ids)
        pooled_output = output[1]
        pooled_output = self.dropout(pooled_output)
         
        # hidden_state_output = self.dropout(output["last_hidden_state"])
        # maybe use pooled output from bert?
        # polled_output = self.dropout(output[1]) # pooler output is at position 1
        
        # GNN forward pass
        h = self.conv1(graph, pooled_output) # hidden state of CLS token is on first position
        h = F.relu(h)
        h = self.conv2(graph, h)
        h = F.relu(h)
        gnn_output = self.conv3(graph, h)
        
        return gnn_output
    
class CA_GCN_Conv1(torch.nn.Module):
    def __init__(self, bert_shortcut_name, hidden_feats, num_classes):
        super().__init__()
        
        # load the pre-trained bert language model
        self.bert = BertModel.from_pretrained(bert_shortcut_name)
        self.dropout = torch.nn.Dropout(self.bert.config.hidden_dropout_prob)
        
        # layers of the GNN
        self.conv1 = dgl.nn.GraphConv(in_feats=self.bert.config.hidden_size, out_feats=num_classes)
    
    def forward(self, graph):
        # foward pass of the bert model
        input_ids = graph.ndata["data_tensor"]
        
        output = self.bert(input_ids=input_ids)
        pooled_output = output[1]
        pooled_output = self.dropout(pooled_output)
         
        # hidden_state_output = self.dropout(output["last_hidden_state"])
        # maybe use pooled output from bert?
        # polled_output = self.dropout(output[1]) # pooler output is at position 1
        
        # GNN forward pass
        gnn_output = self.conv1(graph, pooled_output) # hidden state of CLS token is on first position
        # h = F.relu(h)
        # h = self.conv2(graph, h)
        # h = F.relu(h)
        # gnn_output = self.conv3(graph, h)
        
        return gnn_output
    

class CA_GCN_Tablewise(torch.nn.Module):
    def __init__(self, bert_shortcut_name, hidden_feats, num_classes):
        super().__init__()
        
        # load the pre-trained bert language model
        self.bert = BertModel.from_pretrained(bert_shortcut_name)
        self.dropout = torch.nn.Dropout(self.bert.config.hidden_dropout_prob)
        
        # layers of the GNN
        self.conv1 = dgl.nn.GraphConv(in_feats=self.bert.config.hidden_size, out_feats=hidden_feats)
        self.conv2 = dgl.nn.GraphConv(in_feats=hidden_feats, out_feats=hidden_feats)
        self.conv3 = dgl.nn.GraphConv(in_feats=hidden_feats, out_feats=num_classes)
    
    def forward(self, graph):
        # foward pass of the bert model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        token_ids_list = graph.ndata["data_tensor"].tolist()
        token_ids = torch.LongTensor(reduce(operator.add, token_ids_list)).to(device)
        cls_index_list = torch.LongTensor([0] + np.cumsum(
            np.array([len(x) for x in token_ids_list])).tolist()[:-1]).to(device)
            
        
        output = self.bert(input_ids=token_ids.unsqueeze(0))
        
        # selected the tensors of [CLS] tokens from the output
        col_rep_tensors = torch.index_select(output[0][0], 0, cls_index_list)
        pooled_output = self.dropout(col_rep_tensors) 
         
        # hidden_state_output = self.dropout(output["last_hidden_state"])
        # maybe use pooled output from bert?
        # polled_output = self.dropout(output[1]) # pooler output is at position 1
        
        # GNN forward pass
        h = self.conv1(graph, pooled_output) # hidden state of CLS token is on first position
        h = F.relu(h)
        h = self.conv2(graph, h)
        h = F.relu(h)
        gnn_output = self.conv3(graph, h)
        
        return gnn_output
    
    
class CA_GCN_Tablewise_Conv1(torch.nn.Module):
    def __init__(self, bert_shortcut_name, hidden_feats, num_classes):
        super().__init__()
        
        # load the pre-trained bert language model
        self.bert = BertModel.from_pretrained(bert_shortcut_name)
        self.dropout = torch.nn.Dropout(self.bert.config.hidden_dropout_prob)
        
        # layers of the GNN
        self.conv1 = dgl.nn.GraphConv(in_feats=self.bert.config.hidden_size, out_feats=num_classes)
    
    def forward(self, graph):
        # foward pass of the bert model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        token_ids_list = graph.ndata["data_tensor"].tolist()
        token_ids = torch.LongTensor(reduce(operator.add, token_ids_list)).to(device)
        cls_index_list = torch.LongTensor([0] + np.cumsum(
            np.array([len(x) for x in token_ids_list])).tolist()[:-1]).to(device)
            
        
        output = self.bert(input_ids=token_ids.unsqueeze(0))
        
        # selected the tensors of [CLS] tokens from the output
        col_rep_tensors = torch.index_select(output[0][0], 0, cls_index_list)
        pooled_output = self.dropout(col_rep_tensors) 
         
        # hidden_state_output = self.dropout(output["last_hidden_state"])
        # maybe use pooled output from bert?
        # polled_output = self.dropout(output[1]) # pooler output is at position 1
        
        # GNN forward pass
        gnn_output = self.conv1(graph, pooled_output) # hidden state of CLS token is on first position
        # h = F.relu(h)
        # h = self.conv2(graph, h)
        # h = F.relu(h)
        # gnn_output = self.conv3(graph, h)
        
        return gnn_output
    
    
    
##### GATConv Models
class CA_GAT(torch.nn.Module):
    def __init__(self, bert_shortcut_name, hidden_feats, num_classes):
        super().__init__()
        
        # load the pre-trained bert language model
        self.bert = BertModel.from_pretrained(bert_shortcut_name)
        self.dropout = torch.nn.Dropout(self.bert.config.hidden_dropout_prob)
        
        # layers of the GNN
        self.conv1 = dgl.nn.GATConv(in_feats=self.bert.config.hidden_size, out_feats=num_classes, num_heads=1)
    
    def forward(self, graph):
        # foward pass of the bert model
        input_ids = graph.ndata["data_tensor"]
        
        output = self.bert(input_ids=input_ids)
        pooled_output = output[1]
        pooled_output = self.dropout(pooled_output)
         
        # hidden_state_output = self.dropout(output["last_hidden_state"])
        # maybe use pooled output from bert?
        # polled_output = self.dropout(output[1]) # pooler output is at position 1
        
        # GNN forward pass
        gnn_output = self.conv1(graph, pooled_output) # hidden state of CLS token is on first position
        # h = F.relu(h)
        # h = self.conv2(graph, h)
        # h = F.relu(h)
        # gnn_output = self.conv3(graph, h)
        
        gnn_output = torch.squeeze(gnn_output)
        return gnn_output
    

class CA_GAT_Tablewise(torch.nn.Module):
    def __init__(self, bert_shortcut_name, hidden_feats, num_classes):
        super().__init__()
        
        # load the pre-trained bert language model
        self.bert = BertModel.from_pretrained(bert_shortcut_name)
        self.dropout = torch.nn.Dropout(self.bert.config.hidden_dropout_prob)
        
        # layers of the GNN
        self.conv1 = dgl.nn.GATConv(in_feats=self.bert.config.hidden_size, out_feats=num_classes, num_heads=1)
    
    def forward(self, graph):
        # foward pass of the bert model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        token_ids_list = graph.ndata["data_tensor"].tolist()
        token_ids = torch.LongTensor(reduce(operator.add, token_ids_list)).to(device)
        cls_index_list = torch.LongTensor([0] + np.cumsum(
            np.array([len(x) for x in token_ids_list])).tolist()[:-1]).to(device)
            
        
        output = self.bert(input_ids=token_ids.unsqueeze(0))
        
        # selected the tensors of [CLS] tokens from the output
        col_rep_tensors = torch.index_select(output[0][0], 0, cls_index_list)
        pooled_output = self.dropout(col_rep_tensors) 
         
        # hidden_state_output = self.dropout(output["last_hidden_state"])
        # maybe use pooled output from bert?
        # polled_output = self.dropout(output[1]) # pooler output is at position 1
        
        # GNN forward pass
        gnn_output = self.conv1(graph, pooled_output) # hidden state of CLS token is on first position
        # h = F.relu(h)
        # h = self.conv2(graph, h)
        # h = F.relu(h)
        # gnn_output = self.conv3(graph, h)
        gnn_output = torch.squeeze(gnn_output)
        
        return gnn_output
    
## NN-Model. Be aware that we use heteroenous graph representations of the table here
class CA_GCN_Conv1_enriched(torch.nn.Module):
    def __init__(self, bert_shortcut_name, hidden_feats, num_feat_embedding_dim, num_classes, table_to_column, column_to_column, numStatsFeat_to_column):
        super().__init__()
        self.table_to_column = table_to_column 
        self.column_to_column = column_to_column
        self.numStatsFeat_to_column = numStatsFeat_to_column
        
        # load the pre-trained bert language model
        self.bert = BertModel.from_pretrained(bert_shortcut_name)
        self.dropout = torch.nn.Dropout(self.bert.config.hidden_dropout_prob)
        
        ## MLP for the numerical feature set
        # 192 => number of numerical_features
        # TODO: Check if applying Batch Norm to the input is good
        # TODO: Check how many linear layer will be good
        #self.bn1 = torch.nn.BatchNorm1d(num_features=192)
        #self.num_feat_linear1 = torch.nn.Linear(192,num_feat_embedding_dim)
        self.num_feat_linear1 = torch.nn.Linear(192,self.bert.config.hidden_size)
        #self.num_dp1 = torch.nn.Dropout(0.3)
        #self.num_feat_linear2 = torch.nn.Linear(num_feat_embedding_dim, self.bert.config.hidden_size)
        #self.num_dp2 = torch.nn.Dropout(0.3)
        #self.num_feat_linear3 = torch.nn.Linear(num_feat_embedding_dim, self.bert.config.hidden_size)
        
        # layers of the GNN
        # self.conv1 = dgl.nn.GraphConv(in_feats=self.bert.config.hidden_size, out_feats=num_classes)
        self.conv1 = dgl.nn.HeteroGraphConv({
            'table_column':dgl.nn.GraphConv(in_feats=self.bert.config.hidden_size, out_feats=num_classes),
            'column_column': dgl.nn.GraphConv(in_feats=self.bert.config.hidden_size, out_feats=num_classes),
            'num_feature_column': dgl.nn.GraphConv(in_feats=self.bert.config.hidden_size, out_feats=num_classes)
        }, aggregate="sum") # specify the aggregation method

    def forward(self, graph):
        # foward pass of the bert model
        # get all column repreentation vectors from the BERT model
        input_ids = graph.nodes["column"].data["data_tensor"]
        output = self.bert(input_ids=input_ids)
        pooled_output = output[1] # pooled output is at position 1
        pooled_output = self.dropout(pooled_output)
        
        # update the graph data
        graph.nodes["column"].data["data_tensor"] = pooled_output   
        
        # get table
        if self.table_to_column:
            input_ids = graph.nodes["table"].data["data_tensor"]
            output = self.bert(input_ids=input_ids)
            pooled_output_table = output[1] # pooled output is at position 1
            pooled_output_table = self.dropout(pooled_output_table)
            
            graph.nodes["table"].data["data_tensor"] = pooled_output_table
        
        if self.numStatsFeat_to_column:    
            # forward pass of the MLP for the numerical feature set
            #num_feat_out = self.bn1(graph.nodes["num_feature_node"].data["data_tensor"].float())
            num_feat_out = self.num_feat_linear1(graph.nodes["num_feature_node"].data["data_tensor"].float())
            #num_feat_out = self.num_feat_linear1(num_feat_out)
            #num_feat_out = F.relu(num_feat_out)
            #num_feat_out = self.num_dp1(num_feat_out)
            #num_feat_out = self.num_feat_linear2(num_feat_out)
            #num_feat_out = F.relu(num_feat_out)
            #num_feat_out = self.num_dp2(num_feat_out)
            #num_feat_out = self.num_feat_linear3(num_feat_out)
            
            graph.nodes["num_feature_node"].data["data_tensor"] = num_feat_out
         
        # prepare input data structure for HeteroGraphConv
        if (self.table_to_column == False) and (self.numStatsFeat_to_column == True):
            hetero_graph_conv_input = {
                'column': graph.nodes["column"].data["data_tensor"],
                'num_feature_node': graph.nodes["num_feature_node"].data["data_tensor"]
            }
        elif (self.numStatsFeat_to_column == False) and (self.table_to_column == True):
            hetero_graph_conv_input = {
                'column': graph.nodes["column"].data["data_tensor"],
                'table': graph.nodes["table"].data["data_tensor"]
            }
        elif (self.numStatsFeat_to_column == False) and (self.table_to_column == False) and (self.column_to_column == False):
            hetero_graph_conv_input = {
                'column': graph.nodes["column"].data["data_tensor"]
            }
        elif (self.numStatsFeat_to_column == True) and (self.table_to_column == False) and (self.column_to_column == False):
            hetero_graph_conv_input = {
                'column': graph.nodes["column"].data["data_tensor"],
                'num_feature_node': graph.nodes["num_feature_node"].data["data_tensor"]
            }   
        else:
            hetero_graph_conv_input = {
                'column': graph.nodes["column"].data["data_tensor"],
                'num_feature_node': graph.nodes["num_feature_node"].data["data_tensor"],
                'table': graph.nodes["table"].data["data_tensor"]
            }
            
        
        # GNN forward pass
        gnn_output = self.conv1(graph, hetero_graph_conv_input) # hidden state of CLS token is on first position
        #h = {k: F.relu(v) for k, v in gnn_output.items()}
        
        # h = F.relu(h)
        # h = self.conv2(graph, h)
        # h = F.relu(h)
        # gnn_output = self.conv3(graph, h)
        
        #return graph, pooled_output_table, pooled_output,num_feat_out, gnn_output, hetero_graph_conv_input
        return gnn_output['column']
    
    
## TODO: This class is not finished yet. It not clear if the forware method is correct in current implementation.
## Its unclear how we must pass the updated graph to the next conv layer.
class CA_GCN_Conv3_enriched(torch.nn.Module):
    def __init__(self, bert_shortcut_name, hidden_feats, num_feat_embedding_dim, num_classes):
        super().__init__()
        
        # load the pre-trained bert language model
        self.bert = BertModel.from_pretrained(bert_shortcut_name)
        self.dropout = torch.nn.Dropout(self.bert.config.hidden_dropout_prob)
        
        ## MLP for the numerical feature set
        # 192 => number of numerical_features
        # TODO: Check if applying Batch Norm to the input is good
        # TODO: Check how many linear layer will be good
        #self.bn1 = torch.nn.BatchNorm1d(num_features=192)
        #self.num_feat_linear1 = torch.nn.Linear(192,num_feat_embedding_dim)
        self.num_feat_linear1 = torch.nn.Linear(192,self.bert.config.hidden_size)
        #self.num_dp1 = torch.nn.Dropout(0.3)
        #self.num_feat_linear2 = torch.nn.Linear(num_feat_embedding_dim, num_feat_embedding_dim)
        #self.num_dp2 = torch.nn.Dropout(0.3)
        #self.num_feat_linear3 = torch.nn.Linear(num_feat_embedding_dim, self.bert.config.hidden_size)
        
        # layers of the GNN
        # self.conv1 = dgl.nn.GraphConv(in_feats=self.bert.config.hidden_size, out_feats=num_classes)
        self.conv1 = dgl.nn.HeteroGraphConv({
            'table_column':dgl.nn.GraphConv(in_feats=self.bert.config.hidden_size, out_feats=hidden_feats),
            'column_column': dgl.nn.GraphConv(in_feats=self.bert.config.hidden_size, out_feats=hidden_feats),
            'num_feature_column': dgl.nn.GraphConv(in_feats=self.bert.config.hidden_size, out_feats=hidden_feats)
        }, aggregate="sum") # specify the aggregation method
        
        self.conv2 = dgl.nn.HeteroGraphConv({
            'table_column':dgl.nn.GraphConv(in_feats=hidden_feats, out_feats=hidden_feats),
            'column_column': dgl.nn.GraphConv(in_feats=hidden_feats, out_feats=hidden_feats),
            'num_feature_column': dgl.nn.GraphConv(in_feats=hidden_feats, out_feats=hidden_feats)
        }, aggregate="sum") # specify the aggregation method
        
        self.conv3 = dgl.nn.HeteroGraphConv({
            'table_column':dgl.nn.GraphConv(in_feats=hidden_feats, out_feats=num_classes),
            'column_column': dgl.nn.GraphConv(in_feats=hidden_feats, out_feats=num_classes),
            'num_feature_column': dgl.nn.GraphConv(in_feats=hidden_feats, out_feats=num_classes)
        }, aggregate="sum") # specify the aggregation method

    def forward(self, graph):
        # foward pass of the bert model
        # get all column repreentation vectors from the BERT model
        input_ids = graph.nodes["column"].data["data_tensor"]
        output = self.bert(input_ids=input_ids)
        pooled_output = output[1] # pooled output is at position 1
        pooled_output = self.dropout(pooled_output)
        # get table
        input_ids = graph.nodes["table"].data["data_tensor"]
        output = self.bert(input_ids=input_ids)
        pooled_output_table = output[1] # pooled output is at position 1
        pooled_output_table = self.dropout(pooled_output_table)
        
        # forward pass of the MLP for the numerical feature set
        #num_feat_out = self.bn1(graph.nodes["num_feature_node"].data["data_tensor"].float())
        #num_feat_out = self.num_feat_linear1(graph.nodes["num_feature_node"].data["data_tensor"].float())
        #num_feat_out = self.num_feat_linear1(num_feat_out)
        #num_feat_out = F.relu(num_feat_out)
        #num_feat_out = self.num_dp1(num_feat_out)
        #num_feat_out = self.num_feat_linear2(num_feat_out)
        #num_feat_out = F.relu(num_feat_out)
        #num_feat_out = self.num_dp2(num_feat_out)
        #num_feat_out = self.num_feat_linear3(num_feat_out)
        
        # update the graph data
        graph.nodes["column"].data["data_tensor"] = pooled_output
        #graph.nodes["num_feature_node"].data["data_tensor"] = num_feat_out
        graph.nodes["table"].data["data_tensor"] = pooled_output_table
         
        # prepare input data structure for HeteroGraphConv
        hetero_graph_conv_input = {
            'column': graph.nodes["column"].data["data_tensor"],
        #    'num_feature_node': graph.nodes["num_feature_node"].data["data_tensor"],
            'table': graph.nodes["table"].data["data_tensor"]
        }
        
        # GNN forward pass
        h = self.conv1(graph, hetero_graph_conv_input) # hidden state of CLS token is on first position
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}
        gnn_output = self.conv3(graph, h)
        
        #return graph, pooled_output_table, pooled_output,num_feat_out, gnn_output, hetero_graph_conv_input
        return gnn_output['column']

## NN-Model. Be aware that we use heteroenous graph representations of the table here
class CA_GAT_enriched(torch.nn.Module):
    def __init__(self, bert_shortcut_name, hidden_feats, num_feat_embedding_dim, num_classes, table_to_column, column_to_column, numStatsFeat_to_column):
        super().__init__()
        self.table_to_column = table_to_column 
        self.column_to_column = column_to_column
        self.numStatsFeat_to_column = numStatsFeat_to_column
        
        # load the pre-trained bert language model
        self.bert = BertModel.from_pretrained(bert_shortcut_name)
        self.dropout = torch.nn.Dropout(self.bert.config.hidden_dropout_prob)
        
        ## MLP for the numerical feature set
        # 192 => number of numerical_features
        # TODO: Check if applying Batch Norm to the input is good
        # TODO: Check how many linear layer will be good
        #self.bn1 = torch.nn.BatchNorm1d(num_features=192)
        #self.num_feat_linear1 = torch.nn.Linear(192,num_feat_embedding_dim)
        self.num_feat_linear1 = torch.nn.Linear(192,self.bert.config.hidden_size)
    
        # layers of the GNN
        # self.conv1 = dgl.nn.GraphConv(in_feats=self.bert.config.hidden_size, out_feats=num_classes)
        self.conv1 = dgl.nn.HeteroGraphConv({
            'table_column':dgl.nn.GATConv(in_feats=self.bert.config.hidden_size, out_feats=num_classes, num_heads=1),
            'column_column': dgl.nn.GATConv(in_feats=self.bert.config.hidden_size, out_feats=num_classes, num_heads=1),
            'num_feature_column': dgl.nn.GATConv(in_feats=self.bert.config.hidden_size, out_feats=num_classes, num_heads=1)
        }, aggregate="sum") # specify the aggregation method

    def forward(self, graph):
        # foward pass of the bert model
        # get all column repreentation vectors from the BERT model
        input_ids = graph.nodes["column"].data["data_tensor"]
        output = self.bert(input_ids=input_ids)
        pooled_output = output[1] # pooled output is at position 1
        pooled_output = self.dropout(pooled_output)
        
        # update the graph data
        graph.nodes["column"].data["data_tensor"] = pooled_output
        
        # get table
        if self.table_to_column:
            input_ids = graph.nodes["table"].data["data_tensor"]
            output = self.bert(input_ids=input_ids)
            pooled_output_table = output[1] # pooled output is at position 1
            pooled_output_table = self.dropout(pooled_output_table)
            
            graph.nodes["table"].data["data_tensor"] = pooled_output_table
        
        # forward pass of the MLP for the numerical feature set
        #num_feat_out = self.bn1(graph.nodes["num_feature_node"].data["data_tensor"].float())
        if self.numStatsFeat_to_column:
            num_feat_out = self.num_feat_linear1(graph.nodes["num_feature_node"].data["data_tensor"].float())

            graph.nodes["num_feature_node"].data["data_tensor"] = num_feat_out
        
         
        # prepare input data structure for HeteroGraphConv
        if self.table_to_column == False:
            hetero_graph_conv_input = {
                'column': graph.nodes["column"].data["data_tensor"],
                'num_feature_node': graph.nodes["num_feature_node"].data["data_tensor"]
            }
        elif self.numStatsFeat_to_column == False:
            hetero_graph_conv_input = {
                'column': graph.nodes["column"].data["data_tensor"],
                'table': graph.nodes["table"].data["data_tensor"]
            }
        else:
            hetero_graph_conv_input = {
                'column': graph.nodes["column"].data["data_tensor"],
                'num_feature_node': graph.nodes["num_feature_node"].data["data_tensor"],
                'table': graph.nodes["table"].data["data_tensor"]
            }
        
        # GNN forward pass
        gnn_output = self.conv1(graph, hetero_graph_conv_input) # hidden state of CLS token is on first position

        return gnn_output['column'].squeeze(1)
    
class CA_GAT_enriched_shadow_nums(torch.nn.Module):
    def __init__(self, bert_shortcut_name, hidden_feats, num_feat_embedding_dim, num_classes, one_etype):
        super().__init__()
        
        # load the pre-trained bert language model
        self.bert = BertModel.from_pretrained(bert_shortcut_name)
        self.dropout = torch.nn.Dropout(self.bert.config.hidden_dropout_prob)
        
        ## MLP for the numerical feature set
        # 192 => number of numerical_features
        # TODO: Check if applying Batch Norm to the input is good
        # TODO: Check how many linear layer will be good
        #self.bn1 = torch.nn.BatchNorm1d(num_features=192)
        #self.num_feat_linear1 = torch.nn.Linear(192,num_feat_embedding_dim)
        self.num_feat_linear1 = torch.nn.Linear(192,self.bert.config.hidden_size)
        #self.num_dp1 = torch.nn.Dropout(0.3)
        #self.num_feat_linear2 = torch.nn.Linear(num_feat_embedding_dim, self.bert.config.hidden_size)
        #self.num_dp2 = torch.nn.Dropout(0.3)
        #self.num_feat_linear3 = torch.nn.Linear(num_feat_embedding_dim, self.bert.config.hidden_size)
        
        # layers of the GNN
        # self.conv1 = dgl.nn.GraphConv(in_feats=self.bert.config.hidden_size, out_feats=num_classes)
        if one_etype:
            self.conv1 = dgl.nn.HeteroGraphConv({
                'edge': dgl.nn.GATConv(in_feats=self.bert.config.hidden_size, out_feats=num_classes, num_heads=1)
            }, aggregate="sum") # specify the aggregation method
        else:
            self.conv1 = dgl.nn.HeteroGraphConv({
                'table_column': dgl.nn.GATConv(in_feats=self.bert.config.hidden_size, out_feats=num_classes, num_heads=1),
                'text_num_col': dgl.nn.GATConv(in_feats=self.bert.config.hidden_size, out_feats=num_classes, num_heads=1),
                'num_num_col': dgl.nn.GATConv(in_feats=self.bert.config.hidden_size, out_feats=num_classes, num_heads=1),
                'self': dgl.nn.GATConv(in_feats=self.bert.config.hidden_size, out_feats=num_classes, num_heads=1),
                'num_feature_column': dgl.nn.GATConv(in_feats=self.bert.config.hidden_size, out_feats=num_classes, num_heads=1)
            }, aggregate="sum") # specify the aggregation method

    def forward(self, graph):
        # foward pass of the bert model
        # get all column repreentation vectors from the BERT model
        input_ids = graph.nodes["column"].data["data_tensor"]
        output = self.bert(input_ids=input_ids)
        pooled_output = output[1] # pooled output is at position 1
        pooled_output = self.dropout(pooled_output)
        # get table
        input_ids = graph.nodes["table"].data["data_tensor"]
        output = self.bert(input_ids=input_ids)
        pooled_output_table = output[1] # pooled output is at position 1
        pooled_output_table = self.dropout(pooled_output_table)
        
        # forward pass of the MLP for the numerical feature set
        #num_feat_out = self.bn1(graph.nodes["num_feature_node"].data["data_tensor"].float())
        num_feat_out = self.num_feat_linear1(graph.nodes["num_feature_node"].data["data_tensor"].float())
        #num_feat_out = self.num_feat_linear1(num_feat_out)
        #num_feat_out = F.relu(num_feat_out)
        #num_feat_out = self.num_dp1(num_feat_out)
        #num_feat_out = self.num_feat_linear2(num_feat_out)
        #num_feat_out = F.relu(num_feat_out)
        #num_feat_out = self.num_dp2(num_feat_out)
        #num_feat_out = self.num_feat_linear3(num_feat_out)
        
        # update the graph data
        graph.nodes["column"].data["data_tensor"] = pooled_output
        graph.nodes["num_feature_node"].data["data_tensor"] = num_feat_out
        graph.nodes["table"].data["data_tensor"] = pooled_output_table
         
        # prepare input data structure for HeteroGraphConv
        hetero_graph_conv_input = {
            'column': graph.nodes["column"].data["data_tensor"],
            'num_feature_node': graph.nodes["num_feature_node"].data["data_tensor"],
            'table': graph.nodes["table"].data["data_tensor"]
        }
            
        # GNN forward pass
        gnn_output = self.conv1(graph, hetero_graph_conv_input) # hidden state of CLS token is on first position
        #h = {k: F.relu(v) for k, v in gnn_output.items()}    
        
        # h = F.relu(h)
        # h = self.conv2(graph, h)
        # h = F.relu(h)
        # gnn_output = self.conv3(graph, h)
        
        #return graph, pooled_output_table, pooled_output,num_feat_out, gnn_output, hetero_graph_conv_input
        return gnn_output["column"].squeeze(1) # squeeze(1) is to convert the tensor from format [classes, heads, length] => [classes, length] 
    
    