from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from PIL import Image
import cv2
import os
import io
import pydicom as pdcm
import matplotlib.pyplot as plt
from os.path import exists

class TextImageDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, truncation=True, dir_base='/home/zmh001/r-fcb-isilon/research/Bradshaw/', mode="train", transforms = None): # data_path = os.path.join(dir_base,'Lymphoma_UW_Retrospective/Data/mips/')
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.label
        self.row_ids = self.data.index
        self.max_len = max_len

        self.df_data = dataframe.values
        self.transforms = transforms
        self.mode = mode
        self.data_path = os.path.join(dir_base, "public_datasets/candid_ptx/dataset1/dataset/")

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):

        # text extraction
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        # images data extraction
        img_name = self.row_ids[index]
        img_name = str(img_name) #+ "_mip.png"
        #if exists(os.path.join(self.data_path, 'Group_1_2_3_curated', img_name)):
        #    data_dir = "Group_1_2_3_curated"
        #if exists(os.path.join(self.data_path, 'Group_4_5_curated', img_name)):
        #    data_dir = "Group_4_5_curated"
        data_dir = "public_datasets/candid_ptx/dataset1/dataset/"
        img_path = os.path.join(self.data_path, img_name)

        #print(img_path)
        #DCM_Img = pdcm.read_file(img_path)
        #test = plt.imshow(DCM_Img.pixel_array, cmap=plt.cm.bone)
        #plt.show()


        try:
            DCM_Img = pdcm.read_file(img_path)
            img_raw = DCM_Img.pixel_array
            #img_raw = io.imread(img_path)
            img_norm = img_raw * (255 / 65535)
            img = Image.fromarray(np.uint8(img_norm)).convert("RGB")

        except:
            print("can't open")
            print(img_path)

        if self.transforms is not None:
            image = self.transforms(img)
            try:
                # image = self.transforms(img)
                image = self.transforms(img)
            except:
                print("can't transform")
                print(img_path)
        else:
            image = img


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float),
            'row_ids': self.row_ids[index],
            'images': image
        }