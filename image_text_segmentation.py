import os
from sklearn import model_selection
import torchvision.transforms as transforms
from transformers import AutoTokenizer, RobertaModel, BertModel
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import pandas as pd
from bert_base import BERTClass

from tqdm import tqdm

import numpy as np
import gc
import segmentation_models_pytorch as smp
import albumentations as albu
#from albumentations.pytorch.transforms import ToTensorV2
#from pytorch_metric_learning import losses
#import torch.nn.functional as F
from torch.optim import lr_scheduler
from Gloria import GLoRIA
from ConTEXTual_Segmentation_model import ConTEXTual_seg_model


#from PIL import Image


#from sklearn import metrics
#from sklearn.metrics import accuracy_score, hamming_loss

from candid_datasetup import get_candid_labels, get_all_text_image_pairs
from dataloader_image_text import TextImageDataset
#from vit_base import ViTBase16
#from utility import compute_metrics
from utility import hamming_score, dice_coeff
#from vgg16 import VGG16
from ResNet import resnet_34, resnet_50
#import matplotlib.pyplot as plt
from loss_functions import ContrastiveLoss, global_loss, get_global_similarities

from text_encoder import BertEncoder
from vision_encoder import ImageEncoder

import ssl
ssl.SSLContext.verify_mode = ssl.VerifyMode.CERT_OPTIONAL


def train_image_text_segmentation(config, batch_size=8, epoch=1, dir_base = "/home/zmh001/r-fcb-isilon/research/Bradshaw/", n_classes = 2):

    # model specific global variables
    IMG_SIZE = 256 #256 #1024 #512 #384
    BATCH_SIZE = batch_size
    LR = 5e-5 #8e-5  # 1e-4 was for efficient #1e-06 #2e-6 1e-6 for transformer 1e-4 for efficientnet
    N_EPOCHS = epoch
    N_CLASS = n_classes

    dir_base = config["dir_base"]
    seed = config["seed"]
    batch_size = config["batch_size"]
    N_EPOCHS = config["epochs"]

    #dataframe_location = os.path.join(dir_base, 'Zach_Analysis/candid_data/images_with_text_df.xlsx') #pneumothorax_df chest_tube_df rib_fracture
    dataframe_location = os.path.join(dir_base, 'Zach_Analysis/candid_data/pneumothorax_large_df.xlsx')
    # gets the candid labels and saves it off to the location
    #df = get_candid_labels(dir_base=dir_base)
    #df = get_all_text_image_pairs(dir_base=dir_base)
    #print(df)
    #df.to_excel(dataframe_location, index=False)

    # reads in the dataframe as it doesn't really change to save time

    train_df = pd.read_excel(dataframe_location, engine='openpyxl')
    #print(df)
    train_df.set_index("image_id", inplace=True)



    # creates the path to the roberta model used from the bradshaw drive and loads the tokenizer and roberta model
    language_path = os.path.join(dir_base, 'Zach_Analysis/roberta_large/')
    #language_path = os.path.join(dir_base, 'Zach_Analysis/models/bio_clinical_bert/')
    #language_path = os.path.join(dir_base, 'Zach_Analysis/models/bert/')
    #language_path = os.path.join(dir_base, 'Zach_Analysis/models/candid_mlm/bert_mlm/')
    #language_path = os.path.join(dir_base, 'Zach_Analysis/models/candid_mlm/bio_clinical_bert_candid/')
    #language_path = os.path.join(dir_base, 'Zach_Analysis/models/candid_mlm/roberta_candid_v2/')

    latient_layer = 768
    tokenizer = AutoTokenizer.from_pretrained(language_path)
    #language_model = BertModel.from_pretrained(language_path, output_hidden_states=True)
    language_model = RobertaModel.from_pretrained(language_path, output_hidden_states=True)
    #language_model = BERTClass(language_model, n_class=N_CLASS, n_nodes=latient_layer)
    # roberta_model = BertModel.from_pretrained(roberta_path)

    # takes just the last 512 tokens if there are more than 512 tokens in the text
    # df = truncate_left_text_dataset(df, tokenizer)

    # Splits the data into 80% train and 20% valid and test sets
    #train_df, test_valid_df = model_selection.train_test_split(
    #    df, test_size=0.0, random_state=seed, shuffle=True #stratify=df.label.values
    #)
    # Splits the test and valid sets in half so they are both 10% of total data
    #test_df, valid_df = model_selection.train_test_split(
    #    test_valid_df, test_size=0.5, random_state=seed, shuffle=True #stratify=test_valid_df.label.values
    #)

    #test_dataframe_location = os.path.join(dir_base, 'Zach_Analysis/candid_data/pneumothorax_df_testset.xlsx')
    #test_df.to_excel(test_dataframe_location, index=True)



#    albu_augs = albu.Compose([
#        #ToTensorV2(),
#        albu.HorizontalFlip(),
#        albu.OneOf([
#            albu.RandomContrast(),
#            albu.RandomGamma(),
#            albu.RandomBrightness(),
#        ], p=.3),  #p=0.3),
#        albu.OneOf([
#            #albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
#            #albu.GridDistortion(),
#            albu.OpticalDistortion(distort_limit=2, shift_limit=0.5),
#        ], p=.3),#p=0.3),
#        albu.ShiftScaleRotate(),
#        #albu.Resize(img_size, img_size, always_apply=True),
#    ])
    albu_augs = albu.Compose([
        #ToTensorV2(),
        albu.HorizontalFlip(),
        albu.RandomCrop(height=224, width=224),
        albu.ColorJitter(),
        #albu.RandomAfine

        #albu.ShiftScaleRotate(),
    ])

    transforms_resize = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.PILToTensor()])
    output_resize = transforms.Compose([transforms.Resize((1024, 1024))])


    print("train_df")
    print(train_df)
    training_set = TextImageDataset(train_df, tokenizer, 512, mode="train", transforms = albu_augs, resize=transforms_resize, dir_base = dir_base, img_size=IMG_SIZE)
    #valid_set =    TextImageDataset(valid_df, tokenizer, 512,               transforms = transforms_valid, resize=transforms_resize, dir_base = dir_base, img_size=IMG_SIZE)
    #test_set =     TextImageDataset(test_df,  tokenizer, 512,               transforms = transforms_valid, resize=transforms_resize, dir_base = dir_base, img_size=IMG_SIZE)

    print(training_set)

    train_params = {'batch_size': BATCH_SIZE,
                'shuffle': True,
                'num_workers': 1
                }


    training_loader = DataLoader(training_set, **train_params)
    #valid_loader = DataLoader(valid_set, **test_params)
    #test_loader = DataLoader(test_set, **test_params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #model_obj = ViTBase16(n_classes=N_CLASS, pretrained=True, dir_base=dir_base)
    #model_obj = VGG16(n_classes=N_CLASS, pretrained=True, dir_base=dir_base)

    load_model = False
    if load_model:
        # model is orginally from here which was saved and reloaded to get around SSL
        model_obj = smp.Unet(encoder_name="vgg19", encoder_weights="imagenet", in_channels=3, classes=1)
        save_path = os.path.join(dir_base, 'Zach_Analysis/models/smp_models/default_from_smp/vgg19')
        torch.save(model_obj.state_dict(), save_path)
    else:
        model_obj = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=1, classes=1) #timm-efficientnet-b8 resnet34 decoder_channels=[512, 256, 128, 64, 32]
        #save_path = os.path.join(dir_base, 'Zach_Analysis/models/smp_models/default_from_smp_three_channel/resnet50')
        #model_obj.load_state_dict(torch.load(save_path))

    #text_encoder = BertEncoder(tokenizer=tokenizer, language_model=language_model)
    #img_encoder = ImageEncoder()

    #save_path = os.path.join(dir_base, 'Zach_Analysis/models/resnet34/default_from_smp/resnet152')
    #torch.save(model_obj.state_dict(), save_path)
    #vision_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)

    #vision_model, feature_dim, nums = resnet_50(pretrained=True, dir_base = dir_base)
    #gloria_model = GLoRIA(cfg = None, tokenizer=tokenizer, language_model=language_model)

    run_from_checkpoint = False
    if run_from_checkpoint:
        checkpoint_path = os.path.join(dir_base, 'Zach_Analysis/models/candid_pretrained_models/bert/full_gloria_checkpoint_40ep')
        #gloria_model.load_state_dict(torch.load(checkpoint_path))


    #gloria_model.to(device)

    language_model.to(device)
    model_obj.to(device)

    test_obj = ConTEXTual_seg_model(lang_model=language_model, n_channels=1, n_classes=2, bilinear=False)
    test_obj.to(device)

    #print(model)

    #vision_model.to(device)
    #language_model.to(device)
    #print(model_obj.parameters())
    #for param in model_obj.parameters():
    #    print(param)
    #for param in vision_model.parameters():
    #    param.requires_grad = True
    #for param in language_model.parameters():
    #    param.requires_grad = True
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    criterion = nn.BCEWithLogitsLoss()
    # criterion = ContrastiveLoss(temperature=CFG.temperature).to
    # criterion = ContrastiveLoss(temperature=.1).to(device)
    # criterion = global_loss()

    # defines which optimizer is being used
    optimizer = torch.optim.Adam(params=test_obj.parameters(), lr=LR)
    #optimizer_vis = torch.optim.Adam(params = vision_model.parameters(), lr=LR, weight_decay=1e-6)
    #optimizer_lang = torch.optim.Adam(params=language_model.parameters(), lr=LR, weight_decay=1e-6)
    #optimizer = torch.optim.Adam(params= list(vision_model.parameters()) + list(language_model.parameters()), lr=LR, weight_decay=1e-6)
    #scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    print("about to start training loop")
    lowest_loss = 100

    del train_df

    avg_loss_list = []
    for epoch in range(1, N_EPOCHS + 1):
        #vision_model.train()
        #language_model.train()
        model_obj.train()
        training_dice = []
        gc.collect()
        torch.cuda.empty_cache()

        loss_list = []

        for _, data in tqdm(enumerate(training_loader, 0)):

            x = {}
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            targets = torch.squeeze(targets)
            images = data['images'].to(device, dtype=torch.float)

            x["imgs"] = images
            x["caption_ids"] = ids
            x["attention_mask"] = mask
            x["token_type_ids"] = token_type_ids

            #test = img_encoder(images) #[batchsize, 2048]
            #print("image encoder output")
            #print(test.size())

            outputs = test_obj(images, ids, mask, token_type_ids)


            #text_emb_l, text_emb_g, sents = text_encoder(ids, mask, token_type_ids)
            #print(text_emb_g.size()) #[batchsize, 768]
            #print(text_emb_l.size()) #[batchsize, 768, 512]

            lang_output = language_model(ids, mask, token_type_ids)
            #print(type(lang_output))
            #print(lang_output[0].size())
            #print(lang_output[1].size())
            #print(torch.unsqueeze(lang_output[1], 2).size())
            #lang_rep = torch.unsqueeze(torch.unsqueeze(lang_output[1], 2), 3)
            #lang_rep = lang_rep.repeat(1,2,8,8)
            #print(lang_rep.size())

            test1 = model_obj.encoder(images)
            #print(test1)
            #print("encoder")
            #print(type(test1))

            #text_emb_l = torch.reshape(text_emb_l, (16, 768, 32, 16))
            test1[5] = lang_rep

            #for i in range(0, len(test1)):
            #    print(i)
            #    print(test1[i].size())

            #print("decoder")
            #test2 = model_obj.decoder(*test1)
            #print(type(test2))

            #for i in range(0, len(test2)):
            #    print(i)
            #    print(test2[i].size())
            #print("mask")
            #outputs = model_obj.segmentation_head(test2)




            #outputs = model_obj(images)
            # print(type(outputs))
            outputs = output_resize(torch.squeeze(outputs, dim=1))
            targets = output_resize(targets)

            optimizer.zero_grad()
            # loss = loss_fn(outputs[:, 0], targets)
            loss = criterion(outputs, targets)
            # print(loss)
            if _ % 20 == 0:
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
                training_dice.append(dice)

        avg_training_dice = np.average(training_dice)
        print(f"Epoch {str(epoch)}, Average Training Dice Score = {avg_training_dice}")





