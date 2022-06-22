import argparse
import os

import pandas as pd

from candid_mlm import candid_fine_tuning_candid
from contrastive_training import contrastive_pretraining
from report_generation import report_generation
from segmentation_training import segmentation_training
from image_text_segmentation import train_image_text_segmentation
#from create_unet import load_img_segmentation_model
#from test_model import load_best_model
from siim_dataloader import siim_datasetup

def create_parser():
    parser = argparse.ArgumentParser(description="The main file to run multimodal setup. Consists of pre-training joint representation, masked language modeling and report generation.")
    parser.add_argument('--local', '-l', type=bool, help="Should the program run locally", default=False)
    parser.add_argument('--report_gen', '-r', type=bool, help="Should we train report generation?", default=False)
    parser.add_argument('--mlm_pretraining', '-m', type=bool, help="Should we perform MLM pretraining?", default=False)
    parser.add_argument('--contrastive_training', '-p', type=bool, help="Should we perform multimodal pretraining?", default=False)
    arg = parser.parse_args()
    return arg


if __name__ == '__main__':
    args = create_parser()
    #local = args.local

    local = False

    if local:
        directory_base = "Z:/"
    else:
        directory_base = "/UserData/"

    config = {"seed": 1, "batch_size": 8, "dir_base": directory_base, "epochs": 20, "n_classes": 2, "LR": 1e-5,
              "IMG_SIZE": 256, "train_samples": .8, "test_samples": .5, "data_path": "D:/candid_ptx/", "report_gen":False, "mlm_pretraining":False, "contrastive_training": True}

    train_report_generation = args.report_gen  # flip this to True to do report generation
    if train_report_generation:
        report_generation(config)


    """
    acc, valid_log = train_image_text_segmentation(config)

    df = pd.DataFrame(valid_log)
    df["test_accuracy"] = acc
    folder_name = "text_and_image_sink_down"
    # save to xlsx file
    filepath = os.path.join(directory_base,'/UserData/Zach_Analysis/result_logs/candid_result/text_segmentation/' + str( folder_name) + '/valid_150ep' +"seed"+str(seed) +'.xlsx')
    df.to_excel(filepath, index=False)
    """

    #siim_datasetup(dir_base=directory_base)
    mlm_pretraining = args.mlm_pretraining
    if mlm_pretraining:
        candid_fine_tuning_candid(dir_base=directory_base)


    contrastive_training = args.contrastive_training
    if contrastive_training:
        print("contastive training")
        #pretrained_model, lowest_loss, loss_list = contrastive_pretraining(seed=7, batch_size=16,
        #                                                                   dir_base=directory_base, epoch=50,
        #n_classes=2)
        pretrained_model, lowest_loss, loss_list = contrastive_pretraining(config)

        folder_name = "bert_loss"
        filepath = os.path.join(directory_base, '/UserData/Zach_Analysis/result_logs/candid_result/' + str(
            folder_name) + '/base_bert_mlm/contrastive_ep_loss' + '.xlsx')
        df = pd.DataFrame(loss_list)
        df.to_excel(filepath, index=False)

    # model_obj = load_img_segmentation_model()
    # load_best_model(dir_base= directory_base)
    # seeds = [117, 295, 98, 456, 915, 1367, 712]
    seeds = [98, 117, 295, 456, 915]
    # seeds = [42, 117, 295]
    # seeds = [295]
    # seeds = [915]
    accuracy_list = []

    for seed in seeds:

        config["seed"] = seed

        acc, valid_log = train_image_text_segmentation(config)

        df = pd.DataFrame(valid_log)
        df["test_accuracy"] = acc
        folder_name = "multimodal_v15_resnet_baseline_with_att"
        # save to xlsx file
        filepath = os.path.join(directory_base,
                                '/UserData/Zach_Analysis/result_logs/candid_result/text_segmentation/' + str(
                                    folder_name) + '/valid_150ep_' + "seed" + str(seed) + '.xlsx')
        df.to_excel(filepath, index=False)

    """
    # loops through the segmentation training multiple times with different seeds
    for seed in seeds:
        acc, valid_log = segmentation_training(seed=seed, batch_size=16, dir_base=directory_base, epoch=50,
                                               n_classes=2, pretrained_model=None)
        accuracy_list.append(acc)
        print(valid_log)
        matrix = acc
        df = pd.DataFrame(valid_log)
        df["test_accuracy"] = acc
        folder_name = "imagenet_labeling_functions"
        # save to xlsx file
        filepath = os.path.join(directory_base,
                                '/UserData/Zach_Analysis/result_logs/candid_result/weak_supervision/' + str(
                                    folder_name) + '/valid_run_seed' + str(
                                    seed) + '.xlsx')
        #df.to_excel(filepath, index=False)

    print(accuracy_list)
    # print(lowest_loss)
    """
