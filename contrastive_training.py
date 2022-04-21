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

import ssl
ssl.SSLContext.verify_mode = ssl.VerifyMode.CERT_OPTIONAL


def contrastive_pretraining(seed, batch_size=8, epoch=1, dir_base = "/home/zmh001/r-fcb-isilon/research/Bradshaw/", n_classes = 2):

    # model specific global variables
    IMG_SIZE = 256 #256 #1024 #512 #384
    BATCH_SIZE = batch_size
    LR = 5e-5 #8e-5  # 1e-4 was for efficient #1e-06 #2e-6 1e-6 for transformer 1e-4 for efficientnet
    N_EPOCHS = epoch
    N_CLASS = n_classes
    seed = seed

    dataframe_location = os.path.join(dir_base, 'Zach_Analysis/candid_data/images_with_text_df.xlsx') #pneumothorax_df chest_tube_df rib_fracture
    # gets the candid labels and saves it off to the location
    #df = get_candid_labels(dir_base=dir_base)
    #df = get_all_text_image_pairs(dir_base=dir_base)
    #print(df)
    #df.to_excel(dataframe_location, index=False)

    # reads in the dataframe as it doesn't really change to save time
    train_df = pd.read_excel(dataframe_location, engine='openpyxl')
    #print(df)
    train_df.set_index("image_id", inplace=True)
    #print(df)


    # creates the path to the roberta model used from the bradshaw drive and loads the tokenizer and roberta model
    # roberta_path = os.path.join(dir_base, 'Zach_Analysis/roberta_large/')
    language_path = os.path.join(dir_base, 'Zach_Analysis/models/bio_clinical_bert/')
    # language_path = os.path.join(dir_base, 'Zach_Analysis/models/bert/')
    #language_path = os.path.join(dir_base, 'Zach_Analysis/models/candid_mlm/bio_clinical_bert_candid/')

    latient_layer = 768
    tokenizer = AutoTokenizer.from_pretrained(language_path)
    language_model = BertModel.from_pretrained(language_path, output_hidden_states=True)
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


    # create image augmentations
    transforms_train = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            #transforms.RandomHorizontalFlip(p=0.3),
            #transforms.RandomVerticalFlip(p=0.3),
            #transforms.RandomAffine(degrees = 10, translate =(.1,.1), scale = None, shear = None),
            #transforms.RandomResizedCrop(IMG_SIZE),
            transforms.PILToTensor(),
            #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            #transforms.Grayscale(num_output_channels=1),
            #transforms.Normalize([0.5], [0.5])
        ]
    )

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

    transforms_valid = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.PILToTensor(),
            #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            #transforms.Normalize((0.5,), (0.5,))
            #transforms.Grayscale(num_output_channels=1),
            #transforms.Normalize([0.5], [0.5])
        ]
    )
    transforms_resize = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.PILToTensor()])
    #output_resize = transforms.Compose([transforms.Resize((1024, 1024))])


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

    test_params = {'batch_size': BATCH_SIZE,
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
        model_obj = smp.Unet(encoder_name="vgg19", encoder_weights="imagenet", in_channels=1, classes=1)
        save_path = os.path.join(dir_base, 'Zach_Analysis/models/smp_models/default_from_smp/vgg19')
        torch.save(model_obj.state_dict(), save_path)
    else:
        model_obj = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=1, classes=1) #timm-efficientnet-b8 resnet34
        save_path = os.path.join(dir_base, 'Zach_Analysis/models/smp_models/default_from_smp_three_channel/resnet50')
        #model_obj.load_state_dict(torch.load(save_path))

    #save_path = os.path.join(dir_base, 'Zach_Analysis/models/resnet34/default_from_smp/resnet152')
    #torch.save(model_obj.state_dict(), save_path)
    #vision_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)

    #vision_model, feature_dim, nums = resnet_50(pretrained=True, dir_base = dir_base)
    gloria_model = GLoRIA(cfg = None, tokenizer=tokenizer, language_model=language_model)

    run_from_checkpoint = False
    if run_from_checkpoint:
        checkpoint_path = os.path.join(dir_base, 'Zach_Analysis/models/candid_pretrained_models/bert/full_gloria_checkpoint_40ep')
        gloria_model.load_state_dict(torch.load(checkpoint_path))


    gloria_model.to(device)

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
    # criterion = nn.BCEWithLogitsLoss()
    # criterion = ContrastiveLoss(temperature=CFG.temperature).to
    # criterion = ContrastiveLoss(temperature=.1).to(device)
    # criterion = global_loss()

    # defines which optimizer is being used
    optimizer = torch.optim.Adam(params=gloria_model.parameters(), lr=LR)
    #optimizer_vis = torch.optim.Adam(params = vision_model.parameters(), lr=LR, weight_decay=1e-6)
    #optimizer_lang = torch.optim.Adam(params=language_model.parameters(), lr=LR, weight_decay=1e-6)
    #optimizer = torch.optim.Adam(params= list(vision_model.parameters()) + list(language_model.parameters()), lr=LR, weight_decay=1e-6)
    #scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    print("about to start training loop")
    lowest_loss = 100

    avg_loss_list = []
    for epoch in range(1, N_EPOCHS + 1):
        #vision_model.train()
        #language_model.train()
        gloria_model.train()
        gc.collect()

        loss_list = []

        for _, data in tqdm(enumerate(training_loader, 0)):

            x = {}
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            #targets = data['targets'].to(device, dtype=torch.float)
            #print("test")
            images = data['images'].to(device, dtype=torch.float)

            x["imgs"] = images
            x["caption_ids"] = ids
            x["attention_mask"] = mask
            x["token_type_ids"] = token_type_ids

            img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents = gloria_model(x)
            loss, attn_maps = gloria_model.calc_loss(img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents)


            #loss = criterion(pooler_outputs, vision_outputs)
            #loss_lang, loss_vision = get_global_similarities(vision_outputs, pooler_outputs)
            #loss = ContrastiveLoss(pooler_outputs, vision_outputs)
            #print(pooler_outputs.shape)
            #print(vision_outputs.shape)
            #print(loss)
            #loss = loss(pooler_outputs, vision_outputs)
            #loss_lang, loss_vision = global_loss(img_emb_g,text_emb_g , temp3 = 10)

            #loss_diff = abs(loss_lang.item() - loss_vision.item())
            if _ % 100 == 0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')

                #print(f'Epoch: {epoch}, Contrastive Loss:  {loss}')
                #out_img = plt.imshow(outputs[0].squeeze().cpu().detach().numpy(), cmap=plt.cm.bone)
                #plt.show()
                #tar_img = plt.imshow(targets[0].squeeze().cpu().detach().numpy(), cmap=plt.cm.bone)
                #plt.show()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())

        epoch_avg_loss = np.mean(np.asarray(loss_list))
        avg_loss_list.append(epoch_avg_loss)
        if epoch_avg_loss < lowest_loss:
            lowest_loss = epoch_avg_loss
        print(f"Epoch {str(epoch)} average loss: {epoch_avg_loss}")

        if epoch % 10 == 0:
            save_path = os.path.join(dir_base, 'Zach_Analysis/models/candid_pretrained_models/bio_clincial_bert_v2/candid_checkpoint')
            torch.save(gloria_model.img_encoder.state_dict(), save_path)
            save_path = os.path.join(dir_base, 'Zach_Analysis/models/candid_pretrained_models/bio_clincial_bert_v2/full_gloria_checkpoint')
            torch.save(gloria_model.state_dict(), save_path)


    save_path = os.path.join(dir_base, 'Zach_Analysis/models/candid_pretrained_models/bio_clincial_bert_v2/candid_best_contrastive')
    torch.save(gloria_model.img_encoder.state_dict(), save_path)
    save_path = os.path.join(dir_base, 'Zach_Analysis/models/candid_pretrained_models/bio_clincial_bert_v2/full_gloria')
    torch.save(gloria_model.state_dict(), save_path)



    return gloria_model, lowest_loss, avg_loss_list



