import argparse
import os

import pandas as pd
import argparse

from candid_mlm import candid_fine_tuning_candid
from contrastive_training import contrastive_pretraining
from report_generation import report_generation
from segmentation_training import segmentation_training
from image_text_segmentation import train_image_text_segmentation
#from create_unet import load_img_segmentation_model
#from test_model import load_best_model
#from siim_dataloader import siim_datasetup
from make_visualizations import make_images_on_dgx
from candid_datasetup import get_pneumothorax_image

from make_plots import make_plots
#from test_model import load_best_model
from two_step_segmentation import train_text_classification_then_image_segmentation

def create_parser():
    parser = argparse.ArgumentParser(description="The main file to run multimodal setup. Consists of pre-training joint representation, masked language modeling and report generation.")
    parser.add_argument('--local', '-l', type=bool, help="Should the program run locally", default=False)
    parser.add_argument('--report_gen', '-r', type=bool, help="Should we train report generation?", default=False)
    parser.add_argument('--mlm_pretraining', '-m', type=bool, help="Should we perform MLM pretraining?", default=False)
    parser.add_argument('--contrastive_training', '-p', type=bool, help="Should we perform multimodal pretraining?", default=False)
    arg = parser.parse_args()
    return arg

def get_parser():
    parser = argparse.ArgumentParser(description='LAVT training and testing')
    parser.add_argument('--amsgrad', action='store_true',
                        help='if true, set amsgrad to True in an Adam or AdamW optimizer.')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--bert_tokenizer', default='bert-base-uncased', help='BERT tokenizer')
    parser.add_argument('--ck_bert', default='bert-base-uncased', help='pre-trained BERT weights')
    parser.add_argument('--dataset', default='refcoco', help='refcoco, refcoco+, or refcocog')
    parser.add_argument('--ddp_trained_weights', action='store_true',
                        help='Only needs specified when testing,'
                             'whether the weights to be loaded are from a DDP-trained model')
    parser.add_argument('--device', default='cuda:0', help='device')  # only used when testing on a single machine
    parser.add_argument('--epochs', default=40, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--fusion_drop', default=0.0, type=float, help='dropout rate for PWAMs')
    parser.add_argument('--img_size', default=480, type=int, help='input image size')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--lr', default=0.00005, type=float, help='the initial learning rate')
    parser.add_argument('--mha', default='', help='If specified, should be in the format of a-b-c-d, e.g., 4-4-4-4,'
                                                  'where a, b, c, and d refer to the numbers of heads in stage-1,'
                                                  'stage-2, stage-3, and stage-4 PWAMs')
    parser.add_argument('--model', default='lavt_one', help='model: lavt, lavt_one')
    parser.add_argument('--model_id', default='lavt', help='name to identify the model')
    parser.add_argument('--output-dir', default='./checkpoints/', help='path where to save checkpoint weights')
    parser.add_argument('--pin_mem', action='store_true',
                        help='If true, pin memory when using the data loader.')
    parser.add_argument('--pretrained_swin_weights', default='',
                        help='path to pre-trained Swin backbone weights')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--refer_data_root', default='./refer/data/', help='REFER dataset root directory')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--split', default='test', help='only used when testing')
    parser.add_argument('--splitBy', default='unc', help='change to umd or google when the dataset is G-Ref (RefCOCOg)')
    parser.add_argument('--swin_type', default='base',
                        help='tiny, small, base, or large variants of the Swin Transformer')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float, metavar='W', help='weight decay',
                        dest='weight_decay')
    parser.add_argument('--window12', action='store_true',
                        help='only needs specified when testing,'
                             'when training, window size is inferred from pre-trained weights file name'
                             '(containing \'window12\'). Initialize Swin with window size 12 instead of the default 7.')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers')

    return parser


if __name__ == '__main__':
    #make_plots()
    #print(fail)
    args = create_parser()
    parser = get_parser()
    args_dict = parser.parse_args()
    #local = args.local
    print("newest version")
    local = False
    if local:
        directory_base = "Z:/"
    else:
        directory_base = "/UserData/"

    config = {"seed": 1, "batch_size": 8, "dir_base": directory_base, "epochs": 150, "n_classes": 2, "LR": 1e-5,
              "IMG_SIZE": 256, "train_samples": .8, "test_samples": .5, "data_path": "D:/candid_ptx/", "report_gen":False, "mlm_pretraining":False, "contrastive_training": True}

    config = {"seed": 1, "batch_size": 2, "dir_base": directory_base, "epochs": 50, "n_classes": 2, "LR": 1e-3,
              "IMG_SIZE": 1024, "train_samples":.8, "test_samples": .5, "data_path": "D:/candid_ptx/",
              "report_gen": False, "mlm_pretraining": False, "contrastive_training": False, "save_location": ""} #batch size was 8 with image size 256 .8 can use .004 train_samples = .8
    #image size was 1024 batch size 2
    #args = {}
    #train_report_generation = args.report_gen  # flip this to True to do report generation
    #if train_report_generation:
    #    report_generation(config)
    #train_text_classification_then_image_segmentation(config)
    #load_best_model(directory_base)
    #config["seed"] = 915
    #make_images_on_dgx(config)

    #dataframe_location = os.path.join(directory_base, 'Zach_Analysis/candid_data/pneumothorax_with_multisegmentation_text_negatives_balanced_df.xlsx')

    #df = get_pneumothorax_image(dir_base=directory_base)
    #make_plots()
    #print(fail)
    config["seed"] = 915
    make_images_on_dgx(config)

    print(fail)

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
    #mlm_pretraining = args.mlm_pretraining
    #if mlm_pretraining:
    #    candid_fine_tuning_candid(dir_base=directory_base)

    """
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
    """
    # model_obj = load_img_segmentation_model()
    # load_best_model(dir_base= directory_base)
    # seeds = [117, 295, 98, 456, 915, 1367, 712]
    #seeds = [915]
    #seeds = [98, 98, 98, 98, 98, 98, 98, 98, 98, 98]
    #seeds = [456, 915]
    # seeds = [295]
    #seeds = [98, 117, 295, 456, 915]
    #seeds = [98, 117, 295, 456]
    seeds = [915]
    #seeds = [456]

    accuracy_list = []

    for seed in seeds:

        #folder_name = "with_augmentation/attention_unet_frozen_positive_cases_all_aug_100flip/seed" + str(seed) + "/"
        #folder_name = "higher_res_for_paper/t5_language_att_with_setence_shuffle_larger_img_v27/seed" + str(seed) + "/"
        folder_name = "higher_res_for_paper/contextual_net_less_augmentations_v51/seed" + str(seed) + "/"

        #folder_name = "higher_res_for_paper/baseline_unet_no_aug_larger_img_v28/seed" + str(seed) + "/"

        #folder_name = "with_augmentation/baseline_vision_attention_unet_only_positive_cases/seed" + str(seed) + "/"
        #folder_name = "two_step_seg/dev_test"

        #folder_name = "no_augmentation/attention_unet_frozen_t5_negative_cases/seed" + str(seed) + "/"
        save_string = "/UserData/Zach_Analysis/result_logs/candid_result/image_text_segmentation_for_paper/" + folder_name
        save_location = os.path.join(directory_base, save_string)
        #save_location = ""

        config["seed"] = seed
        config["save_location"] = save_location
        # make_images_on_dgx(config)

        acc, valid_log = train_image_text_segmentation(config, args=args_dict)
        df = pd.DataFrame(valid_log)
        df["test_accuracy"] = acc

        filepath = os.path.join(config["save_location"], "valid_150ep_seed" + str(seed) + '.xlsx')
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
