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

#from sklearn import metrics
from sklearn.metrics import accuracy_score, hamming_loss

from candid_dataloader import get_candid_labels
from dataloader_image_text import TextImageDataset
from vit_base import ViTBase16
from utility import compute_metrics
from utility import hamming_score


def training_loop(seed, batch_size=8, epoch=1, dir_base = "/home/zmh001/r-fcb-isilon/research/Bradshaw/", n_classes = 2):

    print("will have training and stuff here")
    # model specific global variables
    IMG_SIZE = 384
    BATCH_SIZE = batch_size
    LR = 8e-5  # 1e-4 was for efficient #1e-06 #2e-6 1e-6 for transformer 1e-4 for efficientnet
    N_EPOCHS = epoch
    N_CLASS = n_classes
    seed = seed

    dataframe_location = os.path.join(dir_base, 'Zach_Analysis/candid_data/saved_dataframe.xlsx')
    # gets the candid labels and saves it off to the location
    #df = get_candid_labels(dir_base=dir_base)
    #df.to_excel(dataframe_location, index=False)

    # reads in the dataframe as it doesn't really change to save time
    df = pd.read_excel(dataframe_location)
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
        df, test_size=0.2, random_state=seed, stratify=df.label.values
    )
    # Splits the test and valid sets in half so they are both 10% of total data
    test_df, valid_df = model_selection.train_test_split(
        test_valid_df, test_size=0.5, random_state=seed, stratify=test_valid_df.label.values
    )

    # create image augmentations
    transforms_train = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomAffine(degrees = 10, translate =(.1,.1), scale = None, shear = None),
            #transforms.RandomResizedCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    transforms_valid = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    print("train_df")
    print(train_df)
    training_set = TextImageDataset(train_df, tokenizer, 512, mode="train", transforms = transforms_train, dir_base = dir_base)
    valid_set = TextImageDataset(valid_df, tokenizer, 512, transforms = transforms_valid, dir_base = dir_base)
    test_set = TextImageDataset(test_df, tokenizer, 512, transforms = transforms_valid, dir_base = dir_base)

    print(training_set)

    criterion = nn.CrossEntropyLoss()

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

    model_obj = ViTBase16(n_classes=N_CLASS, pretrained=True, dir_base=dir_base)

    model_obj.to(device)

    # defines which optimizer is being used
    optimizer = torch.optim.Adam(params=model_obj.parameters(), lr=LR)

    best_acc = -1
    for epoch in range(1, N_EPOCHS + 1):
        model_obj.train()
        gc.collect()
        fin_targets = []
        fin_outputs = []
        confusion_matrix = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]

        if epoch > 25:
            for param in model_obj.parameters():
                param.requires_grad = True
            for learning_rate in optimizer.param_groups:
                learning_rate['lr'] = 5e-6  # 1e-6 for roberta

        for _, data in tqdm(enumerate(training_loader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)
            images = data['images'].to(device)

            #outputs = model_obj(ids, mask, token_type_ids, images)
            outputs = model_obj(images)

            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
            # fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
            # targets = torch.nn.functional.one_hot(input = targets.long(), num_classes = n_classes)

            optimizer.zero_grad()
            # loss = loss_fn(outputs[:, 0], targets)
            loss = criterion(outputs, targets)
            if _ % 50 == 0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for i in range(0, outputs.shape[0]):
                actual = targets[i].detach().cpu().data.numpy()
                predicted = outputs.argmax(dim=1)[i].detach().cpu().data.numpy()
                confusion_matrix[predicted][actual] += 1

        # get the final score
        # if N_CLASS > 2:
        final_outputs = np.copy(fin_outputs)
        # final_outputs = np.round(final_outputs, decimals=0)
        # final_outputs = (final_outputs == final_outputs.max(axis=1)[:,None]).astype(int)
        final_outputs = np.argmax(final_outputs, axis=1)
        # else:
        #    final_outputs = np.array(fin_outputs) > 0.5

        # print(final_outputs.tolist())
        # print(fin_targets)
        accuracy = accuracy_score(np.array(fin_targets), np.array(final_outputs))
        print(f"Train Accuracy = {accuracy}")
        print(confusion_matrix)

        # each epoch, look at validation data
        model_obj.eval()
        fin_targets = []
        fin_outputs = []
        confusion_matrix = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]

        with torch.no_grad():
            gc.collect()
            for _, data in tqdm(enumerate(valid_loader, 0)):
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.long)
                images = data['images'].to(device)

                #outputs = model_obj(ids, mask, token_type_ids, images)
                outputs = model_obj(images)

                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())  # for two class
                # fin_outputs.extend(outputs.cpu().detach().numpy().tolist())

                for i in range(0, outputs.shape[0]):
                    actual = targets[i].detach().cpu().data.numpy()
                    predicted = outputs.argmax(dim=1)[i].detach().cpu().data.numpy()
                    confusion_matrix[predicted][actual] += 1

            # get the final score
            # if N_CLASS > 2:
            final_outputs = np.copy(fin_outputs)
            # final_outputs = np.round(final_outputs, decimals=0)
            final_outputs = np.argmax(final_outputs, axis=1)
            # final_outputs = (final_outputs == final_outputs.max(axis=1)[:,None]).astype(int)
            # else:
            #    final_outputs = np.array(fin_outputs) > 0.5

            # final_outputs = np.array(fin_outputs) > 0.5
            # final_outputs = np.copy(fin_outputs)
            # final_outputs = (final_outputs == final_outputs.max(axis=1)[:,None]).astype(int)
            val_hamming_loss = hamming_loss(fin_targets, final_outputs)
            val_hamming_score = hamming_score(np.array(fin_targets), np.array(final_outputs))

            accuracy = accuracy_score(np.array(fin_targets), np.array(final_outputs))
            print(f"valid Hamming Score = {val_hamming_score}\nValid Accuracy = {accuracy}")

            print(f"Epoch {str(epoch)}, Validation Hamming Score = {val_hamming_score}")
            print(f"Epoch {str(epoch)}, Validation Hamming Loss = {val_hamming_loss}")
            print(confusion_matrix)
            if accuracy >= best_acc:
                best_acc = accuracy
                save_path = os.path.join(dir_base, 'Zach_Analysis/models/vit/best_multimodal_modal_forked_candid')
                # torch.save(model_obj.state_dict(), '/home/zmh001/r-fcb-isilon/research/Bradshaw/Zach_Analysis/models/vit/best_multimodal_modal')
                torch.save(model_obj.state_dict(), save_path)

    model_obj.eval()
    fin_targets = []
    fin_outputs = []
    row_ids = []
    confusion_matrix = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    saved_path = os.path.join(dir_base, 'Zach_Analysis/models/vit/best_multimodal_modal_forked_candid')
    # model_obj.load_state_dict(torch.load('/home/zmh001/r-fcb-isilon/research/Bradshaw/Zach_Analysis/models/vit/best_multimodal_modal'))
    model_obj.load_state_dict(torch.load(saved_path))

    with torch.no_grad():
        for _, data in tqdm(enumerate(test_loader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)
            images = data['images'].to(device)

            outputs = model_obj(ids, mask, token_type_ids, images)
            row_ids.extend(data['row_ids'])
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())  # for two class
            # fin_outputs.extend(outputs.cpu().detach().numpy().tolist())

            for i in range(0, outputs.shape[0]):
                actual = targets[i].detach().cpu().data.numpy()
                predicted = outputs.argmax(dim=1)[i].detach().cpu().data.numpy()
                confusion_matrix[predicted][actual] += 1

        # get the final score
        # if N_CLASS > 2:
        final_outputs = np.copy(fin_outputs)
        final_outputs = np.argmax(final_outputs, axis=1)
        # final_outputs = np.round(final_outputs, decimals=0)
        # final_outputs = (final_outputs == final_outputs.max(axis=1)[:,None]).astype(int)
        # else:
        #    final_outputs = np.array(fin_outputs) > 0.5

        test_hamming_score = hamming_score(np.array(fin_targets), np.array(final_outputs))
        accuracy = accuracy_score(np.array(fin_targets), np.array(final_outputs))
        print(f"Test Hamming Score = {test_hamming_score}\nTest Accuracy = {accuracy}")
        print(confusion_matrix)

        return accuracy, confusion_matrix


