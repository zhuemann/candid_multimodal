import torch
import segmentation_models_pytorch as smp
import os

def load_img_segmentation_model(
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
    dir_base = "/UserData/"
    ckpt_path = os.path.join(dir_base, 'Zach_Analysis/models/vit/candid_best_contrastive')
    base_model = "resnet34"
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
    ckpt = torch.load(ckpt_path)

    print(ckpt)
    print(ckpt["state_dict"].items())



    ckpt_dict = {}
    for k, v in ckpt["state_dict"].items():
        if k.startswith("gloria.img_encoder.model"):
            k = ".".join(k.split(".")[3:])
            ckpt_dict[k] = v
        ckpt_dict["fc.bias"] = None
        ckpt_dict["fc.weight"] = None
    seg_model.encoder.load_state_dict(ckpt_dict)

    return seg_model