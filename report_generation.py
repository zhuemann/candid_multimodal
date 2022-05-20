import gc
import os

import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn import model_selection
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer

from dataloader_image_text import TextImageDataset
from models.vision_encoder import ImageEncoder


# import IPython


def report_generation(config):
    IMG_SIZE = config["IMG_SIZE"]
    BATCH_SIZE = 1  # config["batch_size"]
    LR = 1e-5  # 8e-5
    N_EPOCHS = 1  # config["epochs"]
    N_CLASS = config["n_classes"]
    dir_base = config["dir_base"]
    seed = config["seed"]
    dataframe_location = os.path.join(dir_base, 'CS769_Gloria_models/pneumothorax_with_text_df.xlsx')

    # reads in the static dataframe to save time
    df = pd.read_excel(dataframe_location, engine='openpyxl')
    df.set_index("image_id", inplace=True)
    print(df)

    # creates the path to the roberta model used from the bradshaw drive and loads the tokenizer and roberta model
    roberta_path = os.path.join(dir_base, 'bert-base-uncased/')  # path to huggingface tokenizer
    tokenizer = AutoTokenizer.from_pretrained(roberta_path)
    # roberta_model = BertModel.from_pretrained(roberta_path)

    # splits up the data in train, test and valid sets (70%, 15%, 15%)
    train_df, test_valid_df = model_selection.train_test_split(
        df, train_size=.7, random_state=seed, shuffle=True  # stratify=df.label.values
    )
    test_df, valid_df = model_selection.train_test_split(
        test_valid_df, test_size=.5, random_state=seed, shuffle=True  # stratify=test_valid_df.label.values
    )

    # sets up augmentations for shaping the images and changing the types
    transforms_valid = transforms.Compose(
        [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.PILToTensor()]
    )
    transforms_resize = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.PILToTensor()])
    # print("train_df", train_df)
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

    device = torch.device("cpu")  # "cuda" if torch.cuda.is_available() else "cpu"
    # Should be the path to the candid checkpoint
    pretrained_imgencoder_path = os.path.join(dir_base, 'CS769_Gloria_models/candid_checkpoint_50ep')
    # loads in the image encoder we trained before, just make sure to change that path to the model path
    state_dict = torch.load(pretrained_imgencoder_path,
                            map_location=torch.device('cpu'))  # remove 2nd argument when running on gpu
    img_encoder = ImageEncoder()
    img_encoder.load_state_dict(state_dict)
    # txt_encoder = BertEncoder()
    img_encoder.to(device)
    # img_feat_g, img_emb_l = img_encoder(imgs, get_local=True)
    # img_emb_g, img_emb_l = img_encoder.generate_embeddings(
    #     img_feat_g, img_emb_l
    # )
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = model.to(device)
    # tokenizer = tokenizer.to(device)  # Error: tokenizer has no to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)

    for epoch in range(1, N_EPOCHS + 1):
        gc.collect()
        # loops through the training loader getting a batch for images and the text
        for _, data in tqdm(enumerate(training_loader, 0)):
            # these are all standard as input into bert models and correspond to the text
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)

            # this is the image which will get fed into the img_encoder to get our vision embedding
            images = data['images'].to(device, dtype=torch.float)
            # outputs = bert_type_model(ids, mask, token_type_ids) # Example from a bert style model
            vis_embedding_pt = torch.tensor(img_encoder(images)).to(device)
            vis_embedding_pt = torch.multiply(vis_embedding_pt, 100000).int()
            vis_embedding_pt = torch.narrow(vis_embedding_pt, dim=1, start=1, length=1024)
            # print(vis_embedding_pt)
            # print("orig here", vis_embedding_pt.shape)
            # IPython.embed()
            outputs_2 = model.generate(inputs=vis_embedding_pt)
            text_op_2 = tokenizer.decode(outputs_2[0], skip_special_tokens=True)
            print("generator op 2", outputs_2)
            print("txt op 2", text_op_2)
            optimizer.zero_grad()
            loss = criterion(text_op_2, ids)
            # print(loss)
            loss.backward()
            optimizer.step()
