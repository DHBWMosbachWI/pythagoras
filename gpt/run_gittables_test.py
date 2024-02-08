## Test columnwise model
from dotenv import load_dotenv
load_dotenv(override=True)
import os
import openai
openai.api_key = os.environ["OPENAI_API_KEY"]
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm
import json

random_state = 1
shuffle_cols = True
modifications = "columnwise"

# load test data
if modifications == None:
    test_dataset = pd.read_json(f"./training_data/GitTables_test_{shuffle_cols}_{random_state}.jsonl", lines=True)
else:
    test_dataset = pd.read_json(f"./training_data/GitTables_test_{shuffle_cols}_{random_state}_0.7_{modifications}.jsonl", lines=True)

model_dic = {
    "tablewise": {
    },
    "columnwise":{
        "False":{
            "1":""
        },
        "True":{
            "1":"ada:ft-data-and-ai-systems-tu-darmstadt:gittables-true-1-0-7-columnwise-2023-11-07-15-24-43",
        }
    }
}

# load model
if modifications == None:
    model = model_dic["tablewise"][str(shuffle_cols)][str(random_state)]
else:
    model = model_dic[f"{modifications}"][str(shuffle_cols)][str(random_state)]
print(f"Loaded model: {model}")  

# test model
total_true_labels = []
total_predicted_labels = []
for i in tqdm(range(0,len(test_dataset))):
    # if i > 0:
    #     break    
    response = openai.Completion.create(
        model= model,
        temperature = 0.8,
        max_tokens=1,
        prompt= test_dataset.iloc[i]["prompt"]
    )
    # print(test_dataset.iloc[i]["prompt"])
    # print(response)
    #number_char_of_predictions = len(test_dataset.iloc[i]["completion"])
    true_label = str(test_dataset.iloc[i]["completion"])
    
    #predicted_label = response["choices"][0]["text"][:number_char_of_predictions]
    predicted_label = response["choices"][0]["text"].replace(" ", "")
        
    total_true_labels.append(true_label)
    total_predicted_labels.append(predicted_label)
    
class_report = classification_report(
    total_true_labels, total_predicted_labels, output_dict=True)

with open(f"./results/GitTables_classification_report_{shuffle_cols}_{random_state}_0.7_{modifications}.json", "w") as f:
    json.dump(class_report, f)
    
with open(f"./results/GitTables_predictions_{shuffle_cols}_{random_state}_0.7_{modifications}.json", "w") as f:
    json.dump({"y_pred":total_predicted_labels, "y_true":total_true_labels}, f)