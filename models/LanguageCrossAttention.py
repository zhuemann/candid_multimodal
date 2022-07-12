import torch
import torch.nn as nn

import os
import cv2

class LangCrossAtt(nn.Module):
    "add documentaiton"


    def __init__(self, emb_dim):
        super(LangCrossAtt, self).__init__()

        self.multihead_attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=1) #vdim=vdimension

    def forward(self, lang_rep, vision_rep):

        # gets all of the dimensions to be used in the attention
        input_batch = vision_rep.size()[0]
        input_channel =  vision_rep.size()[1]
        input_width = vision_rep.size()[2]
        input_height =  vision_rep.size()[3]

        # puts the vision representation into the right shape for attention mechanism
        vision_rep = torch.swapaxes(vision_rep, 0, 1)
        vision_rep_flat = torch.flatten(vision_rep, start_dim=2)
        vision_rep = torch.swapaxes(vision_rep_flat, 2, 0)


        lang_rep = torch.unsqueeze(lang_rep, 1)
        # puts the language rep into the right shape for attention
        lang_rep = torch.swapaxes(lang_rep, 0, 1)
        #lang_rep = torch.swapaxes(lang_rep, 1, 2)


        # does cross attention between vision and language
        att_matrix, attn_output_weights = self.multihead_attn(query=vision_rep, key=lang_rep, value=lang_rep)

        # visualize attention maps
        img = attn_output_weights[0].squeeze().cpu().detach().numpy()
        print(img.shape())
        dir_base = "/UserData/"
        fullpath = os.path.join(dir_base, 'Zach_Analysis/dgx_images/test_img' + str(j) + '.png')
        cv2.imwrite(fullpath, img)

        # gets the attention weights and repeats them to have the same size as the total channels
        attn_output_weights = torch.swapaxes(attn_output_weights, 0, 1)
        #attn_output_weights = torch.swapaxes(attn_output_weights, 0, 2)
        attn_output_weights = attn_output_weights.repeat(1, 1, input_channel)

        # multiplies the attention to focus the vision rep based on the lang rep
        vision_rep = vision_rep * attn_output_weights
        vision_rep = vision_rep.contiguous()

        # rearanges the output matrix to be the dimensions of the input
        out = vision_rep.view(input_width, input_height, input_batch, input_channel)
        out = torch.swapaxes(out, 0, 2)
        out = torch.swapaxes(out, 1, 3)
        return out