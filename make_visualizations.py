import os
from sklearn import model_selection
import torchvision.transforms as transforms
from transformers import AutoTokenizer, RobertaModel, BertModel, T5Model, T5Tokenizer
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import pandas as pd

from tqdm import tqdm
from collections import OrderedDict


import numpy as np
import gc
import segmentation_models_pytorch as smp
import albumentations as albu

from models.Gloria import GLoRIA
import cv2


from torch.optim.lr_scheduler import MultiStepLR

#from albumentations.pytorch.transforms import ToTensorV2
#from pytorch_metric_learning import losses
#import torch.nn.functional as F
from models.Visualization_model import Attention_ConTEXTual_Seg_Model
from models.ResNetUNet import ResNetUNet
from models.ResAttUnet import ResAttNetUNet


#from PIL import Image


#from sklearn import metrics
#from sklearn.metrics import accuracy_score, hamming_loss

from dataloader_image_text import TextImageDataset
#from vit_base import ViTBase16
#from utility import compute_metrics
from utility import dice_coeff
#from vgg16 import VGG16
#import matplotlib.pyplot as plt

import ssl
ssl.SSLContext.verify_mode = ssl.VerifyMode.CERT_OPTIONAL
import nltk

def make_images_on_dgx(config, batch_size=8, epoch=1, dir_base = "/home/zmh001/r-fcb-isilon/research/Bradshaw/", n_classes = 2):
    nltk.download('punkt')
    # model specific global variables
    IMG_SIZE = config["IMG_SIZE"] #256 #1024 #512 #384
    #BATCH_SIZE = batch_size
    LR = 5e-5 #8e-5  # 1e-4 was for efficient #1e-06 #2e-6 1e-6 for transformer 1e-4 for efficientnet
    N_EPOCHS = epoch
    N_CLASS = n_classes

    dir_base = config["dir_base"]
    seed = config["seed"]
    BATCH_SIZE = config["batch_size"]
    BATCH_SIZE = 1
    N_EPOCHS = config["epochs"]
    seed = 98
    dataframe_location = os.path.join(dir_base, 'Zach_Analysis/candid_data/pneumothorax_with_multisegmentation_positive_text_df.xlsx')
    #dataframe_location = os.path.join(dir_base, 'Zach_Analysis/candid_data/pneumothorax_with_text_df.xlsx') #pneumothorax_df chest_tube_df rib_fracture
    #dataframe_location = os.path.join(dir_base, 'Zach_Analysis/candid_data/pneumothorax_large_df.xlsx')
    # gets the candid labels and saves it off to the location
    #df = get_candid_labels(dir_base=dir_base)
    #df = get_all_text_image_pairs(dir_base=dir_base)
    #print(df)
    #df.to_excel(dataframe_location, index=False)

    # reads in the dataframe as it doesn't really change to save time

    df = pd.read_excel(dataframe_location, engine='openpyxl')
    #print(df)
    df.set_index("image_id", inplace=True)



    # creates the path to the roberta model used from the bradshaw drive and loads the tokenizer and roberta model
    #language_path = os.path.join(dir_base, 'Zach_Analysis/roberta_large/')
    #language_path = os.path.join(dir_base, 'Zach_Analysis/roberta/')

    #language_path = os.path.join(dir_base, 'Zach_Analysis/models/bio_clinical_bert/')
    #language_path = os.path.join(dir_base, 'Zach_Analysis/models/bert/')
    #language_path = os.path.join(dir_base, 'Zach_Analysis/models/candid_mlm/bert_mlm/')
    #language_path = os.path.join(dir_base, 'Zach_Analysis/models/candid_mlm/bio_clinical_bert_candid/')
    #language_path = os.path.join(dir_base, 'Zach_Analysis/models/candid_mlm/roberta_candid_v2/')

    latient_layer = 768
    #tokenizer = AutoTokenizer.from_pretrained(language_path)
    #language_model = BertModel.from_pretrained(language_path, output_hidden_states=True)
    #language_model = BERTClass(language_model, n_class=N_CLASS, n_nodes=latient_layer)
    #language_model = BertModel.from_pretrained(language_path, output_hidden_states=True
    #language_model = RobertaModel.from_pretrained(language_path, output_hidden_states=False)


    # use t5 as text encoder
    t5_path = os.path.join(dir_base, 'Zach_Analysis/models/t5_large/')
    tokenizer = T5Tokenizer.from_pretrained(t5_path)
    language_model = T5Model.from_pretrained(t5_path)


    #load in a language model used in the contrastive learning
    pretrained_model = False
    if pretrained_model:
        roberta_path_contrastive_pretraining = os.path.join(dir_base,
                                       'Zach_Analysis/models/candid_pretrained_models/roberta/full_gloria')

        gloria_model = GLoRIA(config=config, tokenizer=tokenizer, language_model=language_model)
        gloria_model.load_state_dict(torch.load(roberta_path_contrastive_pretraining))
        state_dict = gloria_model.text_encoder.state_dict()
        # create new OrderedDict that does not contain `model.`
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[6:]  # remove `model.`
            new_state_dict[name] = v

        language_model.load_state_dict(new_state_dict)

    #language_model.load_state_dict(new_state_dict)


    # takes just the last 512 tokens if there are more than 512 tokens in the text
    # df = truncate_left_text_dataset(df, tokenizer)

    # Splits the data into 80% train and 20% valid and test sets
    train_df, test_valid_df = model_selection.train_test_split(
        df, train_size=config["train_samples"], random_state=seed, shuffle=True #stratify=df.label.values
    )
    # Splits the test and valid sets in half so they are both 10% of total data
    test_df, valid_df = model_selection.train_test_split(
        test_valid_df, test_size=config["test_samples"], random_state=seed, shuffle=True #stratify=test_valid_df.label.values
    )

    valid_dataframe_location = os.path.join(dir_base, 'Zach_Analysis/result_logs/candid_result/image_text_segmentation_for_paper/with_augmentation/' +
    't5_vis_attention_positive_cases_with_bilinear_interp_v11/seed' + str(seed) +'/pneumothorax_df_validseed' + str(seed) + '.xlsx')
    valid_df.to_excel(valid_dataframe_location, index=True)

    train_dataframe_location = os.path.join(dir_base,'Zach_Analysis/result_logs/candid_result/image_text_segmentation_for_paper/with_augmentation/' +
                        't5_vis_attention_positive_cases_with_bilinear_interp_v11/seed' + str(
                        seed) + '/pneumothorax_df_trainseed' + str(seed) + '.xlsx')
    train_df.to_excel(train_dataframe_location, index=True)

    test_frame_locaction = os.path.join(dir_base,
    "Zach_Analysis/result_logs/candid_result/image_text_segmentation_for_paper/with_augmentation/" +
    "t5_vis_attention_positive_cases_with_bilinear_interp_v11/seed98/pneumothorax_testset_df_seed98.xlsx")
    #test_frame_locaction = os.path.join(dir_base,
    #"Zach_Analysis/result_logs/candid_result/image_text_segmentation_for_paper/with_augmentation/" +
    #"multisegmentation_model_train_v13/seed98/examples_to_visualize.xlsx")
    #test_frame_locaction = os.path.join(dir_base,
    #"Zach_Analysis/result_logs/candid_result/image_text_segmentation_for_paper/with_augmentation/" +
    #"multisegmentation_model_train_v13/seed98/size_change_to_visualize_v3.xlsx")

    test_df = pd.read_excel(test_frame_locaction, engine='openpyxl')
    test_df.set_index("image_id", inplace=True)

    albu_augs = albu.Compose([
    ])

    transforms_valid = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.PILToTensor(),
        ]
    )

    transforms_resize = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.PILToTensor()])
    output_resize = transforms.Compose([transforms.Resize((1024, 1024))])

    test_set =     TextImageDataset(test_df,  tokenizer, 512, transforms = transforms_valid, resize=transforms_resize, dir_base = dir_base, img_size=IMG_SIZE)

    train_params = {'batch_size': BATCH_SIZE,
                'shuffle': True,
                'num_workers': 1
                }
    test_params = {'batch_size': BATCH_SIZE,
                   'shuffle': False,
                   'num_workers': 0
                   }

    test_loader = DataLoader(test_set, **test_params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_obj = Attention_ConTEXTual_Seg_Model(lang_model=language_model, n_channels=3, n_classes=1, bilinear=True)

    test_obj.to(device)

    print("visualization run")
    del train_df
    valid_log = []
    test_obj.eval()
    row_ids = []

    #saved_path = os.path.join(dir_base,'Zach_Analysis/models/candid_finetuned_segmentation/forked_2/segmentation_forked_candid_vis_and_word_attention_seed117')
    #saved_path = os.path.join(dir_base,
    #                          'Zach_Analysis/models/candid_finetuned_segmentation/forked_1/segmentation_forked_candid_negatives_seed98')

    saved_path = os.path.join(dir_base,
    "Zach_Analysis/result_logs/candid_result/image_text_segmentation_for_paper/with_augmentation/" +
    "multisegmentation_model_train_v13/seed98/best_segmentation_model_seed98")

    test_obj.load_state_dict(torch.load(saved_path))

    with torch.no_grad():
        test_obj.train()
        test_dice = []
        gc.collect()
        for i, data in tqdm(enumerate(test_loader, 0)):
            print(i)
            #if i == 2:
            #    break
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            #targets = torch.squeeze(targets)
            images = data['images'].to(device, dtype=torch.float)

            #outputs = model_obj(images)
            outputs = test_obj(images, ids, mask, token_type_ids, targets)

            outputs = output_resize(torch.squeeze(outputs, dim=1))
            targets = output_resize(targets)

            sigmoid = torch.sigmoid(outputs)
            outputs = torch.round(sigmoid)
            row_ids.extend(data['row_ids'])

            for i in range(0, outputs.shape[0]):
                dice = dice_coeff(outputs[i], targets[i])
                dice = dice.item()
                if torch.max(outputs[i]) == 0 and torch.max(targets[i]) == 0:
                    dice = 1

                target = targets.cpu().detach().numpy()
                target = target[0, :, :]
                max = np.amax(target)
                target = (target * 255) / max
                fullpath = os.path.join(dir_base, 'Zach_Analysis/dgx_images/negative_cases/targets/' + str(i) + '.png')
                cv2.imwrite(fullpath, target)

                output = outputs.cpu().detach().numpy()
                output = output[0, :, :]
                max = np.amax(output)
                output = (output * 255) / max
                fullpath = os.path.join(dir_base, 'Zach_Analysis/dgx_images/negative_cases/outputs/' + str(i) + '.png')
                cv2.imwrite(fullpath, output)

                #image = images.cpu().detach().numpy()
                image = images[0, 0, :, :]
                image = image.cpu().detach().numpy()
                #images = images[0, :, :]
                fullpath = os.path.join(dir_base, 'Zach_Analysis/dgx_images/negative_cases/images/' + str(i) + '.png')
                cv2.imwrite(fullpath, image)


                test_dice.append(dice)
        print(test_dice)
        print(len(test_dice))
        avg_test_dice = np.average(test_dice)
        print(f"Epoch {str(epoch)}, Average Test Dice Score = {avg_test_dice}")

        return avg_test_dice, valid_log





