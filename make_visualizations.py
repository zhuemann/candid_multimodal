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


def make_images_on_dgx(config, batch_size=8, epoch=1, dir_base = "/home/zmh001/r-fcb-isilon/research/Bradshaw/", n_classes = 2):

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

    dataframe_location = os.path.join(dir_base, 'Zach_Analysis/candid_data/pneumothorax_with_multisegmentation_text_negatives_balanced_df.xlsx')
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

    #test_dataframe_location = os.path.join(dir_base, 'Zach_Analysis/candid_data/pneumothorax_df_testset.xlsx')
    #test_df.to_excel(test_dataframe_location, index=True)

    test_frame_locaction = os.path.join(dir_base,
    "Zach_Analysis/result_logs/candid_result/image_text_segmentation_for_paper/with_augmentation/" +
    "t5_attention_unet_positive_cases_vision_aug_and_text_shuffle_synonom_replacement_v5/seed295/pneumothorax_testset_df_seed295")
    test_df = pd.read_excel(test_frame_locaction, engine='openpyxl')

    albu_augs = albu.Compose([
        # ToTensorV2(),
        #albu.HorizontalFlip(),
        #albu.OneOf([
        #    albu.RandomContrast(),
        #    albu.RandomGamma(),
        #    albu.RandomBrightness(),
        #], p=.3),  # p=0.3),
        #albu.OneOf([
        #    albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        #    albu.GridDistortion(),
        #    albu.OpticalDistortion(distort_limit=2, shift_limit=0.5),
        #], p=.3),  # turned off all three to stabilize training
        #albu.ShiftScaleRotate(),
        # albu.Resize(img_size, img_size, always_apply=True),
    ])



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
    #albu_augs = albu.Compose([
        #ToTensorV2(),
    #    albu.HorizontalFlip(),
    #    albu.RandomCrop(height=224, width=224),
    #    albu.ColorJitter(),
        #albu.RandomAfine

        #albu.ShiftScaleRotate(),
    #])

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
                'num_workers': 1
                }
    test_params = {'batch_size': BATCH_SIZE,
                   'shuffle': True,
                   'num_workers': 4
                   }

    training_loader = DataLoader(training_set, **train_params)
    valid_loader = DataLoader(valid_set, **test_params)
    test_loader = DataLoader(test_set, **test_params)

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
        save_path = os.path.join(dir_base, 'Zach_Analysis/models/smp_models/default_from_smp/resnet50')
        model_obj.load_state_dict(torch.load(save_path))

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

    #language_model.to(device)
    #model_obj.to(device)

    #test_obj = ConTEXTual_seg_model(lang_model=language_model, n_channels=1, n_classes=1, bilinear=False)
    test_obj = Attention_ConTEXTual_Seg_Model(lang_model=language_model, n_channels=3, n_classes=1, bilinear=False)
    #test_obj = ResNetUNet(n_class=1, dir_base=dir_base) #lang_model=language_model

    #test_obj = ResAttNetUNet(lang_model=language_model, n_class=1, dir_base=dir_base)

    for param in language_model.parameters():
        param.requires_grad = False


    #test_obj = Attention_ConTEXTual_Seg_Model(lang_model=language_model, n_channels=3, n_classes=1, bilinear=False)

    test_obj.to(device)


    #    param.requires_grad = True
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    criterion = nn.BCEWithLogitsLoss()
    # criterion = ContrastiveLoss(temperature=CFG.temperature).to
    # criterion = ContrastiveLoss(temperature=.1).to(device)
    # criterion = global_loss()

    # defines which optimizer is being used
    optimizer = torch.optim.AdamW(params=test_obj.parameters(), lr=LR)


    print("visualization run")
    lowest_loss = 100
    best_acc = 0
    del train_df
    valid_log = []
    avg_loss_list = []


    test_obj.eval()
    row_ids = []

    #saved_path = os.path.join(dir_base,'Zach_Analysis/models/candid_finetuned_segmentation/forked_2/segmentation_forked_candid_vis_and_word_attention_seed117')
    #saved_path = os.path.join(dir_base,
    #                          'Zach_Analysis/models/candid_finetuned_segmentation/forked_1/segmentation_forked_candid_negatives_seed98')

    saved_path = os.path.join(dir_base,
    "Zach_Analysis/result_logs/candid_result/image_text_segmentation_for_paper/with_augmentation/" +
    "t5_attention_unet_positive_cases_vision_aug_and_text_shuffle_synonom_replacement_v5/seed295/best_segmentation_model_seed295")

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





