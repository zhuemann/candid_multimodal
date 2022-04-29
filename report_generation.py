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

    # just settings up some paramters from the config dictionary defined in main
    IMG_SIZE = config["IMG_SIZE"]
    BATCH_SIZE = config["batch_size"]
    LR = 1e-5  # 8e-5
    N_EPOCHS = config["epochs"]
    N_CLASS = config["n_classes"]
    dir_base = config["dir_base"]
    seed = config["seed"]

    # this should point to the dataframe which I uploaded to the uw box
    dataframe_location = os.path.join(dir_base,
                                      'Zach_Analysis/candid_data/pneumothorax_with_text_df.xlsx')

    # reads in the dataframe as it doesn't really change to save time
    df = pd.read_excel(dataframe_location, engine='openpyxl')
    df.set_index("image_id", inplace=True)
    print(df)

    # creates the path to the roberta model used from the bradshaw drive and loads the tokenizer and roberta model
    roberta_path = os.path.join(dir_base, 'Zach_Analysis/roberta_large/')
    tokenizer = AutoTokenizer.from_pretrained(roberta_path)
    # roberta_model = BertModel.from_pretrained(roberta_path)

    # splits up the data in train, test and valid sets
    train_df, test_valid_df = model_selection.train_test_split(
        df, train_size=.7, random_state=seed, shuffle=True  # stratify=df.label.values
    )
    # Splits the test and valid sets in half so they are both 10% of total data
    test_df, valid_df = model_selection.train_test_split(
        test_valid_df, test_size=.5, random_state=seed, shuffle=True  # stratify=test_valid_df.label.values
    )

    # sets up some augmentations for shaping the images and changing the types
    transforms_valid = transforms.Compose(
        [transforms.Resize((IMG_SIZE, IMG_SIZE)),transforms.PILToTensor()]
    )
    transforms_resize = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.PILToTensor()])

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

    # sets up the dataloaders which take in the different splits of data and the params setting batch size and such
    training_loader = DataLoader(training_set, **train_params)
    valid_loader = DataLoader(valid_set, **test_params)
    test_loader = DataLoader(test_set, **test_params)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Should be the path to the candid checkpoint you got from the UW box
    pretrained_imgencoder_path = os.path.join(dir_base,'Zach_Analysis/models/candid_pretrained_models/bio_clincial_bert/candid_checkpoint_50ep')

    # loads in the image encoder we trained before, just make sure to change that path to the model path
    state_dict = torch.load(pretrained_imgencoder_path)
    img_encoder = ImageEncoder()
    img_encoder.load_state_dict(state_dict)

    '''
    This may be needed to convert some layer names such that they can be loaded but it might be resolved above and
    not needed, don't worry about this, it will likely be deleted later
    
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
    '''

    img_encoder.to(device)

    """
    Need to create decoder model
    """
    #decoder_model = None
    # once you create the decoder make sure you send it to device
    #decoder_model.to(device)



    # not sure what your loss function should be but that should go here
    criterion = nn.BCEWithLogitsLoss()

    # just used the same optimizer, you want to decoder to be in the params argument here to tell it those are
    # the things to update with the loss right now I gave it the img_encoder just to initialize some optimizer
    optimizer = torch.optim.Adam(params=img_encoder.parameters(), lr=LR)

    for epoch in range(1, N_EPOCHS + 1):
        #decoder_model.train()
        gc.collect()

        # loops through the training loader geting a batch for images and the text
        for _, data in tqdm(enumerate(training_loader, 0)):
            # these are all standard as input into bert models and correspond to the text
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)

            # this target is the segmentation mask, i should be used for anything in this task
            targets = data['targets'].to(device, dtype=torch.float)
            targets = torch.squeeze(targets)

            # this is the image which will get fed into the img_encoder to get our vision embedding
            images = data['images'].to(device, dtype=torch.float)

            # this is an example for a normal bert style model
            #outputs = bert_type_model(ids, mask, token_type_ids)
            vis_embedding = img_encoder(images)
            print(type(vis_embedding))
            print(vis_embedding.shape)

            """
            Need to put the decoder here such that it takes in the vis_embeddings and the ids, mask, token_type_ids
            and outputs those ids again as a target
            """


            optimizer.zero_grad()


            # I think this target will need to be the input text and the outputs here would be whatever the model outputs
            loss = criterion(outputs, targets)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()





