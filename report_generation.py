import torch
import torch.nn as nn
import os
import numpy as np
import gc
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, RobertaModel, BertModel
from sklearn import model_selection
import torchvision.transforms as transforms
from torchvision import models as models_2d
from torch.utils.data import DataLoader
from collections import OrderedDict


from dataloader_image_text import TextImageDataset
from vision_encoder import ImageEncoder


def report_generation(config):


    IMG_SIZE = config["IMG_SIZE"]
    BATCH_SIZE = config["batch_size"]
    LR = 1e-5  # 8e-5
    N_EPOCHS = config["epochs"]
    N_CLASS = config["n_classes"]
    dir_base = config["dir_base"]
    seed = config["seed"]

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

    train_df, test_valid_df = model_selection.train_test_split(
        df, train_size=.7, random_state=seed, shuffle=True  # stratify=df.label.values
    )
    # Splits the test and valid sets in half so they are both 10% of total data
    test_df, valid_df = model_selection.train_test_split(
        test_valid_df, test_size=.5, random_state=seed, shuffle=True  # stratify=test_valid_df.label.values
    )

    transforms_valid = transforms.Compose(
        [transforms.Resize((IMG_SIZE, IMG_SIZE)),transforms.PILToTensor()]
    )
    transforms_resize = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.PILToTensor()])
    output_resize = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE))])

    print("train_df")
    print(train_df)
    training_set = TextImageDataset(train_df, tokenizer, 512, transforms=transforms_valid,
                                    resize=transforms_resize, dir_base=dir_base, img_size=IMG_SIZE)
    valid_set = TextImageDataset(valid_df, tokenizer, 512, transforms=transforms_valid, resize=transforms_resize,
                                 dir_base=dir_base, img_size=IMG_SIZE)
    test_set = TextImageDataset(test_df, tokenizer, 512, transforms=transforms_valid, resize=transforms_resize,
                                dir_base=dir_base, img_size=IMG_SIZE)

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


    # Should be the path to the candid checkpoint you got from the UW box
    pretrained_imgencoder_path = os.path.join(dir_base,'Zach_Analysis/models/candid_pretrained_models/bio_clincial_bert/candid_checkpoint_50ep')

    state_dict = torch.load(pretrained_imgencoder_path)

    img_encoder = ImageEncoder()
    print(type(img_encoder))
    img_encoder.load_state_dict(state_dict)


    # seg_model.encoder.load_state_dict(ckpt)

    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[6:]  # remove `model.`
        new_state_dict[name] = v

    # delete extra layers
    new_state_dict["fc.weight"]= new_state_dict["_embedder.weight"]
    #new_state_dict["fc.bias"] = new_state_dict["global_embedder.bias"]
    del new_state_dict["embedder.weight"]
    del new_state_dict["_embedder.weight"]
    del new_state_dict["_embedder.bias"]

    #sets up the image encoder and loads in the pretrained weights
    #vis_model = models_2d.resnet50(pretrained=False)
    #vis_model.load_state_dict(new_state_dict)
    #vis_model.load_state_dict(state_dict)
    img_encoder.to(device)

    """
    Need to create decoder model
    """
    #decoder_model = None

    #decoder_model.to(device)



    # not sure what your loss function should be but that should go here
    criterion = nn.BCEWithLogitsLoss()

    # just used the same optimizer, you want to decoder to be in the params argument here to tell it those are
    # the things to update with the loss
    optimizer = torch.optim.Adam(params=img_encoder.parameters(), lr=LR)

    for epoch in range(1, N_EPOCHS + 1):
        #decoder_model.train()
        gc.collect()

        for _, data in tqdm(enumerate(training_loader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            targets = torch.squeeze(targets)
            images = data['images'].to(device, dtype=torch.float)

            # outputs = model_obj(ids, mask, token_type_ids, images)
            vis_embedding = img_encoder(images)
            print(type(vis_embedding))
            print(vis_embedding.shape)
            outputs = output_resize(torch.squeeze(outputs, dim=1))

            optimizer.zero_grad()
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


