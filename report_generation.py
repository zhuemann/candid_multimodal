import torch
import torch.nn as nn
import os
import numpy as np
import gc
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, RobertaModel, BertModel


def report_generation(config):


    IMG_SIZE = config["IMG_SIZE"]
    BATCH_SIZE = config["batch_size"]
    LR = 1e-5  # 8e-5
    N_EPOCHS = config["epochs"]
    N_CLASS = config["n_classes"]
    dir_base = config["dir_base"]

    dataframe_location = os.path.join(dir_base,
                                      'Zach_Analysis/candid_data/pneumothorax_large_df.xlsx')  # pneumothorax_df chest_tube_df rib_fracture
    # dataframe_location = os.path.join(dir_base, 'Zach_Analysis/candid_data/pneumothorax_df_testset.xlsx')
    # gets the candid labels and saves it off to the location
    # df = get_candid_labels(dir_base=dir_base)
    # df.to_excel(dataframe_location, index=False)

    # reads in the dataframe as it doesn't really change to save time
    df = pd.read_excel(dataframe_location, engine='openpyxl')
    print(df)

    df.set_index("image_id", inplace=True)
    print(df)

    # creates the path to the roberta model used from the bradshaw drive and loads the tokenizer and roberta model
    roberta_path = os.path.join(dir_base, 'Zach_Analysis/roberta_large/')

    tokenizer = AutoTokenizer.from_pretrained(roberta_path)
    # roberta_model = BertModel.from_pretrained(roberta_path)