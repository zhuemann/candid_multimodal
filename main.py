import os

import pandas as pd

from candid_mlm import candid_fine_tuning_candid
from contrastive_training import contrastive_pretraining
from report_generation import report_generation
from segmentation_training import segmentation_training
# from create_unet import load_img_segmentation_model
from siim_dataloader import siim_datasetup

if __name__ == '__main__':
    local = False
    if local:
        zach_directory_base = "Z:/"  # rename to dir_base for Zach
        # directory_base = "/Users/kritigoyal/Documents/CS_769_NLP/"
    else:
        zach_directory_base = "/UserData/"
        directory_base = "/Users/kritigoyal/Documents/CS_769_NLP/"

    config = {"seed": 1, "batch_size": 8, "dir_base": directory_base, "epochs": 150, "n_classes": 2, "LR": 1e-5,
              "IMG_SIZE": 256, "train_samples": 120, "test_samples": 120, "data_path": "D:/candid_ptx/"}

    train_report_generation = True  # flip this to True to do report generation
    if train_report_generation:
        report_generation(config)

    siim_datasetup(dir_base=directory_base)
    mlm_pretraining = False
    if mlm_pretraining:
        candid_fine_tuning_candid(dir_base=directory_base)

    pretraining = False
    if pretraining:
        pretrained_model, lowest_loss, loss_list = contrastive_pretraining(seed=7, batch_size=16,
                                                                           dir_base=directory_base, epoch=50,
                                                                           n_classes=2)
        folder_name = "bert_loss"
        filepath = os.path.join(directory_base, '/UserData/Zach_Analysis/result_logs/candid_result/' + str(
            folder_name) + '/base_bert_mlm/contrastive_ep_loss' + '.xlsx')
        df = pd.DataFrame(loss_list)
        df.to_excel(filepath, index=False)

    # model_obj = load_img_segmentation_model()
    # load_best_model(dir_base= directory_base)
    # seeds = [117, 295, 98, 456, 915, 1367, 712]
    # seeds = [98, 117, 295, 456, 915]
    # seeds = [42, 117, 295]
    seeds = [295]
    # seeds = [915]
    accuracy_list = []

    # loops through the segmentation training multiple times with different seeds
    for seed in seeds:
        acc, valid_log = segmentation_training(seed=seed, batch_size=16, dir_base=directory_base, epoch=150,
                                               n_classes=2, pretrained_model=None)
        accuracy_list.append(acc)
        print(valid_log)
        matrix = acc
        df = pd.DataFrame(valid_log)
        df["test_accuracy"] = acc
        # file_name = 'pretraining_vision_run_v3'
        # file_name = 'image_net_weights_v2'
        # file_name = 'gloria_vision_run_v3'
        # folder_name = "imagenet_models"
        # folder_name = "bio_clincial_bert_v1_ep100"
        # folder_name = "roberta_100_images"
        # folder_name = "gloria_100_images"
        folder_name = "imagenet_labeling_functions"
        # save to xlsx file
        filepath = os.path.join(directory_base,
                                '/UserData/Zach_Analysis/result_logs/candid_result/weak_supervision/' + str(
                                    folder_name) + '/valid_run_seed' + str(
                                    seed) + '.xlsx')
        df.to_excel(filepath, index=False)

    print(accuracy_list)
    # print(lowest_loss)
