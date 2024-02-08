import os
from os.path import join
import sys
sys.path.append("..")

from glob import glob
from dotenv import load_dotenv
load_dotenv(override=True)
import pandas as pd
from sklearn.model_selection import train_test_split
import json


if __name__ == "__main__":

    sport_domains = ["baseball", "basketball", "football", "hockey", "soccer"]
    random_state = 1

    for random_state in [1,2,3,4,5]:
        for sport_domain_idx, sport_domain in enumerate(sport_domains):
            table_list = glob(join(os.environ["SportsTables"], sport_domain, "*csv"))
            table_list = [table_id.split("/")[-1] for table_id in table_list]
            train, test = train_test_split(table_list, test_size=0.2, random_state=random_state)
            train, valid = train_test_split(train, test_size=0.2, random_state=random_state)
            print(sport_domain, len(table_list), len(train), len(valid), len(test))
            with open(join(os.environ["SportsTables"], sport_domain, f"train_valid_test_split_{random_state}.json"), "w") as f:
                json.dump({
                    "train": train,
                    "valid": valid,
                    "test": test
                }, f)