# pythagoras
This is the code repo of the research paper "Pythagoras: Semantic Type Detection of Numerical Data in Enterprise Data Lakes".

## Settings
In the `.env` file you have to set the given environment variables first. 
- MAIN_DIR => The path of the directory of the code-repo of Pythagoras
- SportsTables => The path to the SportsTables corpus
- GitTables => The path to the GitTables corpus
- PYTHON => Your command to execute Python scripts e.g. `python3`
- OPENAI_API_KEY => If you like to run the experiment with the GPT model, you have to enter your OpenAI API-key here   

## Run Experiments
To run a complete experiment as described in the paper you can execute 
`scripts\run_complete_exp.py`

There are various variables in the script that define which experiment is to be executed.
Descriptions of the variables are in the Python script. Nevertheless, you will find descriptions of how to set these variables in the following.
- `data_corpus` => defines on which corpus you will run the experiment (SportsTables, GitTables)
- `sport_domains_m` => special for SportsTables. You can choose which sports will be included in the experiment with this variable (e.g. ["'baseball', 'basketball', 'football', 'hockey', 'soccer'"]). Unfortunately, there are different setting for training and validation runs. Be aware of the comments in the script.
- `model_architecture` => defines which model architecture of the neural network should be used for the experiment. We used "CA_GCN_Conv1_enriched" as the final model for the result in the paper.
- `table_graph_representation` => defines which graph representation of tables should be used for the experiment. We used "enriched" as the final table representation for the results in the paper.

Be aware that inside the "for" loop of the script, there are different Python run commands for running training, validation ....
Comment in the runs that you want to execute!




