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

from models.T5_classifier import T5Classifier

from models.Gloria import GLoRIA


from torch.optim.lr_scheduler import MultiStepLR

#from albumentations.pytorch.transforms import ToTensorV2
#from pytorch_metric_learning import losses
#import torch.nn.functional as F
from models.ConTEXTual_Seg_attention_model import Attention_ConTEXTual_Seg_Model
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


def train_text_classification_then_image_segmentation(config, batch_size=8, epoch=1, dir_base = "/home/zmh001/r-fcb-isilon/research/Bradshaw/", n_classes = 2):

    # model specific global variables
    IMG_SIZE = config["IMG_SIZE"] #256 #1024 #512 #384
    #BATCH_SIZE = batch_size
    LR = 5e-5 #8e-5  # 1e-4 was for efficient #1e-06 #2e-6 1e-6 for transformer 1e-4 for efficientnet
    N_EPOCHS = epoch
    N_CLASS = n_classes

    dir_base = config["dir_base"]
    seed = config["seed"]
    BATCH_SIZE = config["batch_size"]
    N_EPOCHS = config["epochs"]
    #LR = config["LR"]

    dataframe_location = os.path.join(dir_base, "Zach_Analysis/candid_data/pneumothorax_with_multisegmentation_text_negatives_balanced_df.xlsx")
    #dataframe_location = os.path.join(dir_base, 'Zach_Analysis/candid_data/pneumothorax_with_multisegmentation_text_df_test1.xlsx')
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

    # use t5 as text encoder
    t5_path = os.path.join(dir_base, 'Zach_Analysis/models/t5_large/')
    tokenizer = T5Tokenizer.from_pretrained(t5_path)
    language_model = T5Model.from_pretrained(t5_path)

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

    #test_dataframe_location = os.path.join(dir_base, 'Zach_Analysis/candid_data/pneumothorax_df_testset.xlsx')
    #test_df.to_excel(test_dataframe_location, index=True)

    # report invariant augmentaitons
    using_t5 = True
    if using_t5:
        albu_augs = albu.Compose([
            albu.OneOf([
                albu.RandomContrast(),
                albu.RandomGamma(),
                albu.RandomBrightness(),
            ], p=.3),
            albu.OneOf([
                albu.GridDistortion(),
                albu.OpticalDistortion(distort_limit=2, shift_limit=0.5),
                albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
            ], p=.3)
        ])

    # emprically the good augmentations, taken from kaggle winner
    vision_only = False
    if vision_only:
        albu_augs = albu.Compose([
            albu.HorizontalFlip(),
            albu.OneOf([
                albu.RandomContrast(),
                albu.RandomGamma(),
                albu.RandomBrightness(),
            ], p=.3),
            albu.OneOf([
                albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                albu.GridDistortion(),
                albu.OpticalDistortion(distort_limit=2, shift_limit=0.5),
            ], p=.3),
            albu.ShiftScaleRotate(),
        ])

    # used for empty augmentation tests
    # if not vision_only and not using_t5:
    # albu_augs = albu.Compose([
    #
    #        ])

    transforms_valid = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.PILToTensor(),
            # transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5)),
            # transforms.Normalize((0.5,), (0.5,))
            # transforms.Grayscale(num_output_channels=1),
            # transforms.Normalize([0.5], [0.5])
        ]
    )

    transforms_resize = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.PILToTensor()])
    output_resize = transforms.Compose([transforms.Resize((1024, 1024))])


    print("train_df")
    print(train_df)
    print("valid df")
    print(valid_df)
    training_set = TextImageDataset(train_df, tokenizer, 512, mode="train", transforms = albu_augs, resize=transforms_resize, dir_base = dir_base, img_size=IMG_SIZE)
    valid_set =    TextImageDataset(valid_df, tokenizer, 512,               transforms = transforms_valid, resize=transforms_resize, dir_base = dir_base, img_size=IMG_SIZE)
    test_set =     TextImageDataset(test_df,  tokenizer, 512,               transforms = transforms_valid, resize=transforms_resize, dir_base = dir_base, img_size=IMG_SIZE)

    train_params = {'batch_size': BATCH_SIZE,
                'shuffle': True,
                'num_workers': 2
                }
    test_params = {'batch_size': BATCH_SIZE,
                   'shuffle': True,
                   'num_workers': 4
                   }

    training_loader = DataLoader(training_set, **train_params)
    valid_loader = DataLoader(valid_set, **test_params)
    test_loader = DataLoader(test_set, **test_params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_model = False
    if load_model:
        # model is orginally from here which was saved and reloaded to get around SSL
        model_obj = smp.Unet(encoder_name="vgg19", encoder_weights="imagenet", in_channels=3, classes=1)
        save_path = os.path.join(dir_base, 'Zach_Analysis/models/smp_models/default_from_smp/vgg19')
        torch.save(model_obj.state_dict(), save_path)
    else:
        model_obj = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=1, classes=1) #timm-efficientnet-b8 resnet34 decoder_channels=[512, 256, 128, 64, 32]
        save_path = os.path.join(dir_base, 'Zach_Analysis/models/smp_models/default_from_smp/resnet50')
        model_obj.load_state_dict(torch.load(save_path))

    #test_obj = Attention_ConTEXTual_Seg_Model(lang_model=language_model, n_channels=3, n_classes=1, bilinear=False)

    #test_obj = ResAttNetUNet(lang_model=language_model, n_class=1, dir_base=dir_base)

    for param in language_model.parameters():
        param.requires_grad = False

    test_obj = T5Classifier(language_model, n_class=2)
    test_obj.to(device)

    criterion = nn.BCEWithLogitsLoss()

    # defines which optimizer is being used
    optimizer = torch.optim.AdamW(params=test_obj.parameters(), lr=LR)

    #optimizer = torch.optim.Adam(params=test_obj.parameters(), lr=LR) # was used for all the baselines
    #optimizer_vis = torch.optim.Adam(params = vision_model.parameters(), lr=LR, weight_decay=1e-6)
    #optimizer_lang = torch.optim.Adam(params=language_model.parameters(), lr=LR, weight_decay=1e-6)
    #optimizer = torch.optim.Adam(params= list(vision_model.parameters()) + list(language_model.parameters()), lr=LR, weight_decay=1e-6)
    #scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    #scheduler = MultiStepLR(optimizer, milestones=[5, 10, 25, 37, 50, 75], gamma=0.50)

    print("about to start training loop")
    lowest_loss = 100
    best_acc = 0
    del train_df
    valid_log = []
    avg_loss_list = []
    for epoch in range(1, N_EPOCHS + 1):
        #vision_model.train()
        #language_model.train()
        #model_obj.train()
        test_obj.train()
        training_dice = []
        gc.collect()
        torch.cuda.empty_cache()

        loss_list = []

        for _, data in tqdm(enumerate(training_loader, 0)):

            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            targets = torch.squeeze(targets)
            images = data['images'].to(device, dtype=torch.float)


            #outputs = test_obj(images, ids, mask, token_type_ids)
            outputs = test_obj(ids, mask, token_type_ids)

            for target in targets:
                if torch.max(target) == 0:
                    target = 0
                if torch.max(target) == 1:
                    target = 1

            print(targets)





            """
            outputs = output_resize(torch.squeeze(outputs, dim=1))
            targets = output_resize(targets)
            
            
            
            optimizer.zero_grad()

            loss = criterion(outputs, targets)

            if _ % 400 == 0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # put output between 0 and 1 and rounds to nearest integer ie 0 or 1 labels
            sigmoid = torch.sigmoid(outputs)
            outputs = torch.round(sigmoid)

            # calculates the dice coefficent for each image and adds it to the list
            for i in range(0, outputs.shape[0]):
                dice = dice_coeff(outputs[i], targets[i])
                dice = dice.item()
                # gives a dice score of 1 if correctly predicts negative
                if torch.max(outputs[i]) == 0 and torch.max(targets[i]) == 0:
                    dice = 1

                training_dice.append(dice)

        avg_training_dice = np.average(training_dice)
        print(f"Epoch {str(epoch)}, Average Training Dice Score = {avg_training_dice}")

        # each epoch, look at validation data
        with torch.no_grad():

            #model_obj.eval()
            test_obj.eval()
            valid_dice = []
            gc.collect()
            for _, data in tqdm(enumerate(valid_loader, 0)):
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.float)
                targets = torch.squeeze(targets)
                images = data['images'].to(device, dtype=torch.float)

                #outputs = model_obj(images)
                outputs = test_obj(images, ids, mask, token_type_ids)


                outputs = output_resize(torch.squeeze(outputs, dim=1))
                targets = output_resize(targets)

                # put output between 0 and 1 and rounds to nearest integer ie 0 or 1 labels
                sigmoid = torch.sigmoid(outputs)
                outputs = torch.round(sigmoid)

                # calculates the dice coefficent for each image and adds it to the list
                for i in range(0, outputs.shape[0]):
                    dice = dice_coeff(outputs[i], targets[i])
                    dice = dice.item()
                    if torch.max(outputs[i]) == 0 and torch.max(targets[i]) == 0:
                        dice = 1
                    valid_dice.append(dice)

            #scheduler.step()
            avg_valid_dice = np.average(valid_dice)
            print(f"Epoch {str(epoch)}, Average Valid Dice Score = {avg_valid_dice}")
            valid_log.append(avg_valid_dice)

            if avg_valid_dice >= best_acc:
                best_acc = avg_valid_dice
                # save_path = os.path.join(dir_base, 'Zach_Analysis/models/vit/best_multimodal_modal_forked_candid')
                save_path = os.path.join(dir_base, 'Zach_Analysis/models/candid_finetuned_segmentation/forked_1/segmentation_forked_candid')
                #save_path = os.path.join(dir_base, 'Zach_Analysis/models/candid_finetuned_segmentation/forked_2/segmentation_forked_candid2')
                #save_path = os.path.join(dir_base,'Zach_Analysis/models/candid_finetuned_segmentation/forked_3/segmentation_forked_candid')

                #save_path = os.path.join(dir_base,
                #                         'Zach_Analysis/models/candid_finetuned_segmentation/weak_supervision_models/imagenet_labeling_functions/segmentation_candid' + str(
                #                             seed))
                # torch.save(model_obj.state_dict(), '/home/zmh001/r-fcb-isilon/research/Bradshaw/Zach_Analysis/models/vit/best_multimodal_modal')
                torch.save(test_obj.state_dict(), save_path)

    test_obj.eval()
    row_ids = []
    # saved_path = os.path.join(dir_base, 'Zach_Analysis/models/vit/best_multimodal_modal_forked_candid')
    saved_path = os.path.join(dir_base,'Zach_Analysis/models/candid_finetuned_segmentation/forked_1/segmentation_forked_candid')
    #saved_path = os.path.join(dir_base,'Zach_Analysis/models/candid_finetuned_segmentation/forked_2/segmentation_forked_candid2')
    #saved_path = os.path.join(dir_base,'Zach_Analysis/models/candid_finetuned_segmentation/forked_3/segmentation_forked_candid')

    #saved_path = os.path.join(dir_base,
    #                          'Zach_Analysis/models/candid_finetuned_segmentation/weak_supervision_models/imagenet_labeling_functions/segmentation_candid' + str(
    #                              seed))
    # model_obj.load_state_dict(torch.load('/home/zmh001/r-fcb-isilon/research/Bradshaw/Zach_Analysis/models/vit/best_multimodal_modal'))
    test_obj.load_state_dict(torch.load(saved_path))

    with torch.no_grad():
        test_dice = []
        gc.collect()
        for _, data in tqdm(enumerate(test_loader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            targets = torch.squeeze(targets)
            images = data['images'].to(device, dtype=torch.float)

            #outputs = model_obj(images)
            outputs = test_obj(images, ids, mask, token_type_ids)

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
                test_dice.append(dice)

        avg_test_dice = np.average(test_dice)
        print(f"Epoch {str(epoch)}, Average Test Dice Score = {avg_test_dice}")

        return avg_test_dice, valid_log
        """





