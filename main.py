

from segmentation_training import segmentation_training
import pandas as pd
import os
from utility import rle_decode, mask2rle, rle_decode_modified
import matplotlib.pyplot as plt
import numpy as np
#from test_model import load_best_model
from contrastive_training import contrastive_pretraining
from create_unet import load_img_segmentation_model



if __name__ == '__main__':

    #Sets which directory to use
    local = False
    if local == True:
        directory_base = "Z:/"
        #directory_base = "/home/zmh001/r-fcb-isilon/research/Bradshaw/"
    else:
        directory_base = "/UserData/"

    pretrained_model = contrastive_pretraining(seed = 7, batch_size = 16, dir_base= directory_base, epoch = 30, n_classes = 2)
    #pretrained_model = None
    #model_obj = load_img_segmentation_model()

    #load_best_model(dir_base= directory_base)

    seeds = [117, 295, 98, 456, 915, 1367, 712]
    accuracy_list = []

    # loops through the segmentation training multiple times with different seeds
    for seed in seeds:

        acc = segmentation_training(seed = seed, batch_size = 16, dir_base= directory_base, epoch = 35, n_classes = 2, pretrained_model = pretrained_model)
        acc = 1
        accuracy_list.append(acc)

        matrix = acc
        #df = pd.DataFrame(matrix)
        file_name = 'first_vision_run'
        ## save to xlsx file
        filepath = os.path.join(directory_base,
                                '/UserData/Zach_Analysis/result_logs/candid_result/tests/' + str(file_name) +'/confusion_matrix_seed' + str(
                                    seed) + '.xlsx')

        #df.to_excel(filepath, index=False)

    print(accuracy_list)

