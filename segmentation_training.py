import os
from sklearn import model_selection
import torchvision.transforms as transforms
from transformers import AutoTokenizer, RobertaModel, BertModel
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import pandas as pd

from tqdm import tqdm

import numpy as np
import gc
import segmentation_models_pytorch as smp
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2


from PIL import Image


#from sklearn import metrics
from sklearn.metrics import accuracy_score, hamming_loss

from candid_datasetup import get_candid_labels
from dataloader_image_text import TextImageDataset
from vit_base import ViTBase16
from utility import compute_metrics
from utility import hamming_score, dice_coeff
from create_unet import load_img_segmentation_model
from vgg16 import VGG16
import matplotlib.pyplot as plt


def segmentation_training(seed, batch_size=8, epoch=1, dir_base = "/home/zmh001/r-fcb-isilon/research/Bradshaw/", n_classes = 2, pretrained_model = None):

    print("will have training and stuff here")
    # model specific global variables
    IMG_SIZE = 256 #256 #512 #384
    BATCH_SIZE = batch_size
    LR = 1e-5 #8e-5  # 1e-4 was for efficient #1e-06 #2e-6 1e-6 for transformer 1e-4 for efficientnet
    N_EPOCHS = epoch
    N_CLASS = n_classes
    seed = seed

    dataframe_location = os.path.join(dir_base, 'Zach_Analysis/candid_data/pneumothorax_large_df.xlsx') #pneumothorax_df chest_tube_df rib_fracture
    # gets the candid labels and saves it off to the location
    #df = get_candid_labels(dir_base=dir_base)
    #df.to_excel(dataframe_location, index=False)

    # reads in the dataframe as it doesn't really change to save time
    df = pd.read_excel(dataframe_location, engine='openpyxl')
    print(df)
    df.set_index("image_id", inplace=True)
    print(df)


    # creates the path to the roberta model used from the bradshaw drive and loads the tokenizer and roberta model
    roberta_path = os.path.join(dir_base, 'Zach_Analysis/roberta_large/')

    tokenizer = AutoTokenizer.from_pretrained(roberta_path)
    # roberta_model = BertModel.from_pretrained(roberta_path)

    # takes just the last 512 tokens if there are more than 512 tokens in the text
    # df = truncate_left_text_dataset(df, tokenizer)

    # Splits the data into 80% train and 20% valid and test sets
    train_df, test_valid_df = model_selection.train_test_split(
        df, test_size=0.2, random_state=seed, shuffle=True #stratify=df.label.values
    )
    # Splits the test and valid sets in half so they are both 10% of total data
    test_df, valid_df = model_selection.train_test_split(
        test_valid_df, test_size=0.5, random_state=seed, shuffle=True #stratify=test_valid_df.label.values
    )

    test_dataframe_location = os.path.join(dir_base, 'Zach_Analysis/candid_data/pneumothorax_df_testset.xlsx')
    test_df.to_excel(test_dataframe_location, index=True)


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

    albu_augs = albu.Compose([
        #ToTensorV2(),
        albu.HorizontalFlip(),
        albu.OneOf([
            albu.RandomContrast(),
            albu.RandomGamma(),
            albu.RandomBrightness(),
        ], p=.3),  #p=0.3),
        albu.OneOf([
            albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            albu.GridDistortion(),
            albu.OpticalDistortion(distort_limit=2, shift_limit=0.5),
        ], p=.3),#p=0.3),
        albu.ShiftScaleRotate(),
        #albu.Resize(img_size, img_size, always_apply=True),
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
    output_resize = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE))])


    print("train_df")
    print(train_df)
    training_set = TextImageDataset(train_df, tokenizer, 512, mode="train", transforms = albu_augs, resize=transforms_resize, dir_base = dir_base, img_size=IMG_SIZE)
    valid_set =    TextImageDataset(valid_df, tokenizer, 512,               transforms = transforms_valid, resize=transforms_resize, dir_base = dir_base, img_size=IMG_SIZE)
    test_set =     TextImageDataset(test_df,  tokenizer, 512,               transforms = transforms_valid, resize=transforms_resize, dir_base = dir_base, img_size=IMG_SIZE)

    print(training_set)

    train_params = {'batch_size': BATCH_SIZE,
                'shuffle': True,
                'num_workers': 4
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
        model_obj = smp.Unet(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=1)
        save_path = os.path.join(dir_base, 'Zach_Analysis/models/smp_models/default_from_smp_three_channel/resnet50')
        torch.save(model_obj.state_dict(), save_path)
    else:
        model_obj = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1) #timm-efficientnet-b8 resnet34
        save_path = os.path.join(dir_base, 'Zach_Analysis/models/smp_models/default_from_smp_three_channel/resnet50')
        model_obj.load_state_dict(torch.load(save_path))


    use_pretrained_encoder = False
    if use_pretrained_encoder:
        model_obj = load_img_segmentation_model(dir_base = dir_base, pretrained_model=True)

    #save_path = os.path.join(dir_base, 'Zach_Analysis/models/resnet34/default_from_smp/resnet152')
    #torch.save(model_obj.state_dict(), save_path)

    model_obj.to(device)
    #print(model_obj.parameters())
    #for param in model_obj.parameters():
    #    print(param)

    #criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    criterion = nn.BCEWithLogitsLoss()

    # defines which optimizer is being used
    optimizer = torch.optim.Adam(params=model_obj.parameters(), lr=LR)
    print("about to start training loop")
    best_acc = -1
    valid_log = []
    for epoch in range(1, N_EPOCHS + 1):
        model_obj.train()
        gc.collect()
        training_dice = []

        for _, data in tqdm(enumerate(training_loader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            targets = torch.squeeze(targets)
            images = data['images'].to(device, dtype=torch.float)

            #print(images.shape)
            #outputs = model_obj(ids, mask, token_type_ids, images)
            outputs = model_obj(images)
            #print(type(outputs))
            outputs = output_resize(torch.squeeze(outputs, dim=1))

            optimizer.zero_grad()
            # loss = loss_fn(outputs[:, 0], targets)
            loss = criterion(outputs, targets)
            # print(loss)
            if _ % 20 == 0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')

                #f, ax = plt.subplots(1, 3)
                #ax[0].imshow(np.uint8(torch.permute(images[0], (1,2,0)).squeeze().cpu().detach().numpy()))
                #ax[1].imshow(outputs[0].squeeze().cpu().detach().numpy(), cmap=plt.cm.bone)
                #ax[2].imshow(targets[0].squeeze().cpu().detach().numpy(), cmap=plt.cm.bone)
                #ax[2].imshow(np.uint8(torch.permute(images[0], (1,2,0)).squeeze().cpu().detach().numpy()), cmap=plt.cm.bone, alpha=.5)

                #out_img = plt.imshow(outputs[0].squeeze().cpu().detach().numpy(), cmap=plt.cm.bone)
                #plt.show()
                #tar_img = plt.imshow(targets[0].squeeze().cpu().detach().numpy(), cmap=plt.cm.bone)
                #plt.show()

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

        # each epoch, look at validation data

        with torch.no_grad():
            valid_dice = []
            gc.collect()
            for _, data in tqdm(enumerate(valid_loader, 0)):
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.float)
                images = data['images'].to(device, dtype=torch.float)

                #outputs = model_obj(ids, mask, token_type_ids, images)
                outputs = model_obj(images)
                outputs = output_resize(torch.squeeze(outputs, dim=1))

                # put output between 0 and 1 and rounds to nearest integer ie 0 or 1 labels
                sigmoid = torch.sigmoid(outputs)
                outputs = torch.round(sigmoid)

                # calculates the dice coefficent for each image and adds it to the list
                for i in range(0, outputs.shape[0]):
                    dice = dice_coeff(outputs[i], targets[i])
                    dice = dice.item()
                    valid_dice.append(dice)

            avg_valid_dice = np.average(valid_dice)
            print(f"Epoch {str(epoch)}, Average Valid Dice Score = {avg_valid_dice}")
            valid_log.append(avg_valid_dice)

            if avg_valid_dice >= best_acc:
                best_acc = avg_valid_dice
                save_path = os.path.join(dir_base, 'Zach_Analysis/models/vit/best_multimodal_modal_forked_candid')
                # torch.save(model_obj.state_dict(), '/home/zmh001/r-fcb-isilon/research/Bradshaw/Zach_Analysis/models/vit/best_multimodal_modal')
                torch.save(model_obj.state_dict(), save_path)

    model_obj.eval()

    row_ids = []
    saved_path = os.path.join(dir_base, 'Zach_Analysis/models/vit/best_multimodal_modal_forked_candid')
    # model_obj.load_state_dict(torch.load('/home/zmh001/r-fcb-isilon/research/Bradshaw/Zach_Analysis/models/vit/best_multimodal_modal'))
    model_obj.load_state_dict(torch.load(saved_path))

    with torch.no_grad():
        test_dice = []
        gc.collect()
        for _, data in tqdm(enumerate(test_loader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            images = data['images'].to(device, dtype=torch.float)

            # outputs = model_obj(ids, mask, token_type_ids, images)
            outputs = model_obj(images)
            outputs = output_resize(torch.squeeze(outputs, dim=1))

            sigmoid = torch.sigmoid(outputs)
            outputs = torch.round(sigmoid)

            row_ids.extend(data['row_ids'])

            for i in range(0, outputs.shape[0]):
                dice = dice_coeff(outputs[i], targets[i])
                dice = dice.item()
                test_dice.append(dice)

        avg_test_dice = np.average(test_dice)
        print(f"Epoch {str(epoch)}, Average Test Dice Score = {avg_test_dice}")


        return avg_test_dice, valid_log


