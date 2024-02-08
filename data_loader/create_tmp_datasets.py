#from GitTables_data_loader import GitTablesDGLDataset_enriched, GitTablesDGLDataset_enriched_shadow_num_nodes
from SportsDB_data_loader import SportsTablesDGLDataset_enriched_column_names
from transformers import BertTokenizer, BertModel
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

for random_state in [1,2,3,4,5]:
    #train_dataset = GitTablesDGLDataset_enriched_shadow_num_nodes(tokenizer=tokenizer, max_length=254, random_state=random_state, shuffle_cols=False, device=device, split="train", force_reload=True)
    #valid_dataset = GitTablesDGLDataset_enriched_shadow_num_nodes(tokenizer=tokenizer, max_length=254, random_state=random_state, shuffle_cols=False, device=device, split="valid", force_reload=True)
    #test_dataset = GitTablesDGLDataset_enriched_shadow_num_nodes(tokenizer=tokenizer, max_length=254, random_state=random_state, shuffle_cols=False, device=device, split="test", force_reload=True)
    
    train_dataset = SportsTablesDGLDataset_enriched_column_names(tokenizer=tokenizer, max_length=254, random_state=random_state, shuffle_cols=False, device=device, split="train", force_reload=True, gpt_gen_colnames=True)
    valid_dataset = SportsTablesDGLDataset_enriched_column_names(tokenizer=tokenizer, max_length=254, random_state=random_state, shuffle_cols=False, device=device, split="valid", force_reload=True, gpt_gen_colnames=True)
    test_dataset = SportsTablesDGLDataset_enriched_column_names(tokenizer=tokenizer, max_length=254, random_state=random_state, shuffle_cols=False, device=device, split="test", force_reload=True, gpt_gen_colnames=True)