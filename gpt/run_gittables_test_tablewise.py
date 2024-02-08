## Test tablewise model
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
modifications = None

# load test data
if modifications == None:
    test_dataset = pd.read_json(f"./training_data/GitTables_test_{shuffle_cols}_{random_state}_0.7.jsonl", lines=True)
else:
    test_dataset = pd.read_json(f"./training_data/GitTables_test_{shuffle_cols}_{random_state}_0.7_{modifications}.jsonl", lines=True)

model_dic = {
    "tablewise": {
        "True":{
            "1": "ada:ft-data-and-ai-systems-tu-darmstadt:gittables-true-1-0-7-2023-11-08-11-10-13"
        }
    },
    "columnwise":{
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
missed_predictions = 0
for i in tqdm(range(0,len(test_dataset))):
    try:
        # if i > 0:
        #     break    
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
    except:
        missed_predictions += 1
        print(missed_predictions)
    
class_report = classification_report(
    total_true_labels, total_predicted_labels, output_dict=True)

with open(f"./results/GitTables_classification_report_{shuffle_cols}_{random_state}_0.7.json", "w") as f:
    json.dump(class_report, f)
    
with open(f"./results/GitTables_predictions_{shuffle_cols}_{random_state}_0.7.json", "w") as f:
    json.dump({"y_pred":total_predicted_labels, "y_true":total_true_labels}, f)