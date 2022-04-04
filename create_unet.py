import torch
import segmentation_models_pytorch as smp
import os
from collections import OrderedDict

def load_img_segmentation_model(dir_base = "/UserData/", pretrained_model = None,
    name: str = "gloria_resnet50",
):
    """Load a GLoRIA pretrained classification model
    Parameters
    ----------
    name : str
        A model name listed by `gloria.available_models()`, or the path to a model checkpoint containing the state_dict
    device : Union[str, torch.device]
        The device to put the loaded model
    Returns
    -------
    img_model : torch.nn.Module
        The GLoRIA pretrained image classification model
    """
    #dir_base = "/UserData/"
    #dir_base = "Z:/"
    ckpt_path = os.path.join(dir_base, 'Zach_Analysis/models/candid_pretrained_models/candid_best_contrastive')
    base_model = "resnet50"
    # warnings
    #if name in _SEGMENTATION_MODELS:
    #    ckpt_path = os.path.join(dir_base, 'Zach_Analysis/models/vit/candid_best_contrastive')
    #    base_model = name.split("_")[-1]
    #elif os.path.isfile(name):
    #    ckpt_path = name
    #    base_model = "resnet50"  #
    #else:
    #    raise RuntimeError(
    #        f"Model {name} not found; available models = {available_segmentation_models()}"
    #    )

    # load base model
    seg_model = smp.Unet(base_model, encoder_weights=None, activation=None)

    # update weight
    #ckpt = torch.load(ckpt_path)

    state_dict = torch.load(ckpt_path)

    if pretrained_model == True:
        #seg_model.encoder.load_state_dict(ckpt)

        # create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[6:]  # remove `model.`
            new_state_dict[name] = v
        # load params
        new_state_dict = new_state_dict.pop("_embedder.weight", None)
        new_state_dict = new_state_dict.pop("_embedder.bias", None)
        new_state_dict = new_state_dict.pop("embedder.weight", None)
        #new_state_dict["_embedder.bias"] = None
        #new_state_dict["embedder.weight"] = None

        seg_model.encoder.load_state_dict(new_state_dict)


    #print(ckpt["OrderedDict"].items())
    else:
        seg_model.encoder.load_state_dict(ckpt)


    #ckpt_dict = {}
    #for k, v in ckpt["state_dict"].items():
    #    if k.startswith("gloria.img_encoder.model"):
    #        k = ".".join(k.split(".")[3:])
    #        ckpt_dict[k] = v
    #    ckpt_dict["fc.bias"] = None
    #    ckpt_dict["fc.weight"] = None
    #seg_model.encoder.load_state_dict(ckpt_dict)

    return seg_model