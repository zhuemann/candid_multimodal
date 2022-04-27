

from segmentation_training import segmentation_training
import pandas as pd
import os
from utility import rle_decode, mask2rle, rle_decode_modified
import matplotlib.pyplot as plt
import numpy as np
#from test_model import load_best_model
import pydicom as pdcm
from contrastive_training import contrastive_pretraining
#from create_unet import load_img_segmentation_model
from siim_dataloader import siim_datasetup



if __name__ == '__main__':

    #Sets which directory to use
    local = False
    if local == True:
        directory_base = "Z:/"
        #directory_base = "/home/zmh001/r-fcb-isilon/research/Bradshaw/"
    else:
        directory_base = "/UserData/"


    #siim_datasetup(dir_base = directory_base)


    pretraining = False
    if pretraining:
        pretrained_model, lowest_loss, loss_list  = contrastive_pretraining(seed = 7, batch_size = 1, dir_base= directory_base, epoch = 50, n_classes = 2)

        folder_name = "bio_clincial_bert"
        filepath = os.path.join(directory_base, '/UserData/Zach_Analysis/result_logs/candid_result/' + str(folder_name) + '/base_bio_clincial_bert/contrastive_ep_loss' + '.xlsx')
        df = pd.DataFrame(loss_list)
        df.to_excel(filepath, index=False)



    #model_obj = load_img_segmentation_model()

    #load_best_model(dir_base= directory_base)

    seeds = [117, 295, 98, 456, 915, 1367, 712]
    #seeds = [98, 117, 295, 456, 915]

    #seeds = [915]
    accuracy_list = []

    # loops through the segmentation training multiple times with different seeds
    for seed in seeds:

        acc, valid_log = segmentation_training(seed = seed, batch_size = 6, dir_base= directory_base, epoch = 150, n_classes = 2, pretrained_model = None)

        accuracy_list.append(acc)
        print(valid_log)
        matrix = acc
        df = pd.DataFrame(valid_log)
        df["test_accuracy"] = acc
        #file_name = 'pretraining_vision_run_v3'
        #file_name = 'image_net_weights_v2'
        #file_name = 'gloria_vision_run_v3'
        #folder_name = "imagenet_models"
        #folder_name = "bio_clincial_bert_v1_ep100"
        folder_name = "gloria_full_data_150ep"
        ## save to xlsx file
        filepath = os.path.join(directory_base,
                                '/UserData/Zach_Analysis/result_logs/candid_result/bio_clincial_bert_seg/' + str(folder_name) +'/valid_run_seed' + str(
                                    seed) + '.xlsx')

        df.to_excel(filepath, index=False)

    print(accuracy_list)
    #print(lowest_loss)

