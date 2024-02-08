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
shuffle_cols = False
sport_domain = "overall"
modifications = "columnwise"

# load test data
if modifications == None:
    test_dataset = pd.read_json(f"./training_data/{sport_domain}_test_{shuffle_cols}_{random_state}.jsonl", lines=True)
else:
    test_dataset = pd.read_json(f"./training_data/{sport_domain}_test_{shuffle_cols}_{random_state}_{modifications}.jsonl", lines=True)

model_dic = {
    "overall": {
        "False": {
            "1": "ada:ft-personal:overall-false-1-2023-03-27-15-30-00",
            "2": "ada:ft-personal:overall-false-2-2023-03-27-17-27-58"
        },
        "True": {
            "1": "ada:ft-personal:overall-true-1-2023-03-27-15-56-42",
            "2": "ada:ft-data-and-ai-systems-tu-darmstadt:overall-true-2-2023-03-28-08-47-27"
        }
    },
    "overall_columnwise":{
        "False":{
            "1":"ada:ft-data-and-ai-systems-tu-darmstadt:overall-train-false-columnwisee-2023-03-30-21-05-01"
        }
    },
    "overall_transposed":{
        "True":{
            "1":"ada:ft-data-and-ai-systems-tu-darmstadt:overall-true-1-transposed-2023-03-30-20-13-28"
        }
    },
    "baseball": {
        "False": {
            "1":"ada:ft-data-and-ai-systems-tu-darmstadt:baseball-false-1-2023-03-28-22-19-35"
            },
        "True": {
            "1":"ada:ft-data-and-ai-systems-tu-darmstadt:baseball-true-1-2023-03-29-00-11-28"
            }
    }
}

# load model
if modifications == None:
    model = model_dic[sport_domain][str(shuffle_cols)][str(random_state)]
else:
    model = model_dic[f"{sport_domain}_{modifications}"][str(shuffle_cols)][str(random_state)]
print(f"Loaded model: {model}")  

# test model
total_true_labels = []
total_predicted_labels = []
for i in tqdm(range(0,len(test_dataset))):    
    response = openai.Completion.create(
        model= model,
        temperature = 0.8,
        max_tokens=600,
        prompt= test_dataset.iloc[i]["prompt"]
    )
    number_of_predictions = len(test_dataset.iloc[i]["completion"].split("\n"))
    true_labels = test_dataset.iloc[i]["completion"].split("\n")
    true_labels[0] = true_labels[0].replace(" ", "")
    
    predicted_labels = response["choices"][0]["text"].split("\n")[:number_of_predictions]
    predicted_labels[0] = predicted_labels[0].replace(" ", "")
    
    if number_of_predictions != len(predicted_labels):
        print(f"Number of labels should be {number_of_predictions} but is {len(predicted_labels)}")
        print(i)
        break
        
    total_true_labels.extend(true_labels)
    total_predicted_labels.extend(predicted_labels)
    
class_report = classification_report(
    total_true_labels, total_predicted_labels, output_dict=True)

with open(f"./results/classification_report_{sport_domain}_{shuffle_cols}_{random_state}_{modifications}.json", "w") as f:
    json.dump(class_report, f)
    
with open(f"./results/predictions_{sport_domain}_{shuffle_cols}_{random_state}_{modifications}.json", "w") as f:
    json.dump({"y_pred":total_predicted_labels, "y_true":total_true_labels}, f)