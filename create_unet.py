import torch
import segmentation_models_pytorch as smp
import os
from collections import OrderedDict

def load_img_segmentation_model(dir_base = "/UserData/", pretrained_model = None):

    #pretrained_path = os.path.join(dir_base, 'Zach_Analysis/models/candid_pretrained_models/bio_clincial_bert/candid_best_contrastive')
    #pretrained_path = os.path.join(dir_base, 'Zach_Analysis/models/candid_pretrained_models/bio_clincial_bert/candid_checkpoint_50ep')
    pretrained_path = os.path.join(dir_base,'Zach_Analysis/models/candid_pretrained_models/roberta/candid_best_contrastive')
    #pretrained_path = os.path.join(dir_base,'Zach_Analysis/models/candid_pretrained_models/bert/candid_best_contrastive')

    #pretrained_path = os.path.join(dir_base, 'Zach_Analysis/models/candid_pretrained_models/bert/candid_best_contrastive')
    base_model = "resnet50"

    # load base model
    segmentation_model = smp.Unet(base_model, encoder_weights=None, activation=None)

    if pretrained_model == True:
        state_dict = torch.load(pretrained_path)
        #seg_model.encoder.load_state_dict(ckpt)

        # create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[6:]  # remove `model.`
            new_state_dict[name] = v

        # delete extra layers
        del new_state_dict["_embedder.weight"]
        del new_state_dict["_embedder.bias"]
        del new_state_dict["embedder.weight"]

        # load in the parameters
        segmentation_model.encoder.load_state_dict(new_state_dict)

    else:
        pretrained_path = os.path.join(dir_base, 'Zach_Analysis/models/candid_pretrained_models/bio_clincial_bert/chexpert_resnet50.ckpt')
        state_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))#map_location='cuda:0')
        new_state_dict = {}
        for k, v in state_dict["state_dict"].items():
            if k.startswith("gloria.img_encoder.model"):
                k = ".".join(k.split(".")[3:])
                new_state_dict[k] = v
            new_state_dict["fc.bias"] = None
            new_state_dict["fc.weight"] = None
        segmentation_model.encoder.load_state_dict(new_state_dict)

    return segmentation_model