import torch
import torch.nn as nn
import numpy as np

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

        print(f"input batch {input_batch}")
        print(f"input channel {input_channel}")
        print(f"input width {input_width}")
        print(f"input height {input_height}")


        # puts the vision representation into the right shape for attention mechanism
        vision_rep = torch.swapaxes(vision_rep, 0, 1)
        vision_rep_flat = torch.flatten(vision_rep, start_dim=2)
        vision_rep = torch.swapaxes(vision_rep_flat, 2, 0)


        lang_rep = torch.unsqueeze(lang_rep, 1)
        # puts the language rep into the right shape for attention
        lang_rep = torch.swapaxes(lang_rep, 0, 1)
        #lang_rep = torch.swapaxes(lang_rep, 1, 2)

        #print(f"vision_rep used as key and value:{vision_rep.size()}")
        #print(f"lang_rep used as key and query: {lang_rep.size()}")
        # does cross attention between vision and language
        att_matrix, attn_output_weights = self.multihead_attn(query=vision_rep, key=lang_rep, value=lang_rep)


        print(f"attention matrix: {att_matrix.size()}")
        #print(f"attention_output_weight {attn_output_weights.size()}")
        print(f"vision rep: {vision_rep.size()}")

        #print(f"attension matrix max: {torch.max(att_matrix)}")
        #print(f"attension matrix min: {torch.min(att_matrix)}")

        #print(f"max: {torch.max(attn_output_weights)}")
        #print(f"min: {torch.min(attn_output_weights)}")

        #print("attn_output weights")
        #print(attn_output_weights.size())
        #print("vis_rep")
        #print(vision_rep.size())

        # visualize attention maps
        img = att_matrix.cpu().detach().numpy()

        img = img[:,0,0]
        #img2 = img[:,0,1]

        print(f"all the elements for one batch {np.shape(img)}")

        img = np.reshape(img, (input_width, input_height))
        #img = np.reshape(img, (input_channel, 1))

        max = np.amax(img)
        min = np.amin(img)
        print(f"max: {max}")
        print(f"min: {min}")
        print(np.shape(img))
        img = (img * 255) / max
        dir_base = "/UserData/"
        fullpath = os.path.join(dir_base, 'Zach_Analysis/dgx_images/attention_visualize/test_img' + '.png')
        cv2.imwrite(fullpath, img)

        img = att_matrix.cpu().detach().numpy()
        for i in range(0,input_channel):
            img_ch = img[:,0,i]
            img_ch = np.reshape(img_ch, (input_width, input_height))
            max = np.amax(img_ch)
            img_ch = (img_ch * 255) / max
            fullpath = os.path.join(dir_base, 'Zach_Analysis/dgx_images/attention_visualize/test_img_ch:'+str(i) + '.png')
            cv2.imwrite(fullpath, img_ch)

        vision_rep = vision_rep * att_matrix
        vision_rep = vision_rep.contiguous()

        # rearanges the output matrix to be the dimensions of the input
        out = vision_rep.view(input_width, input_height, input_batch, input_channel)
        out = torch.swapaxes(out, 0, 2)
        out = torch.swapaxes(out, 1, 3)

        print(f"out size: {out.size()}")
        return out







"""
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

        print(f"input batch {input_batch}")
        print(f"input channel {input_channel}")
        print(f"input width {input_width}")
        print(f"input height {input_height}")


        # puts the vision representation into the right shape for attention mechanism
        vision_rep = torch.swapaxes(vision_rep, 0, 1)
        vision_rep_flat = torch.flatten(vision_rep, start_dim=2)
        vision_rep = torch.swapaxes(vision_rep_flat, 2, 0)


        lang_rep = torch.unsqueeze(lang_rep, 1)
        # puts the language rep into the right shape for attention
        lang_rep = torch.swapaxes(lang_rep, 0, 1)
        #lang_rep = torch.swapaxes(lang_rep, 1, 2)

        print(f"vision_rep used as query:{vision_rep.size()}")
        print(f"lang_rep used as key and value: {lang_rep.size()}")
        # does cross attention between vision and language
        att_matrix, attn_output_weights = self.multihead_attn(query=vision_rep, key=lang_rep, value=lang_rep)


        print(f"attention matrix: {att_matrix.size()}")
        #print(f"attention_output_weight {attn_output_weights.size()}")
        print(f"vision rep: {vision_rep.size()}")

        #print(f"attension matrix max: {torch.max(att_matrix)}")
        #print(f"attension matrix min: {torch.min(att_matrix)}")

        #print(f"max: {torch.max(attn_output_weights)}")
        #print(f"min: {torch.min(attn_output_weights)}")

        #print("attn_output weights")
        #print(attn_output_weights.size())
        #print("vis_rep")
        #print(vision_rep.size())

        # visualize attention maps
        img = att_matrix.cpu().detach().numpy()

        img = img[0,0,:]

        print(f"all the elements for one batch {np.shape(img)}")

        #img = np.reshape(img, (input_width, input_height))
        img = np.reshape(img, (input_channel, 1))

        max = np.amax(img)
        min = np.amin(img)
        print(f"max: {max}")
        print(f"min: {min}")
        print(np.shape(img))
        img = (img * 255) / max
        dir_base = "/UserData/"
        fullpath = os.path.join(dir_base, 'Zach_Analysis/dgx_images/attention_visualize/test_img' + '.png')
        cv2.imwrite(fullpath, img)

        vision_rep = vision_rep * att_matrix
        vision_rep = vision_rep.contiguous()

        # rearanges the output matrix to be the dimensions of the input
        out = vision_rep.view(input_width, input_height, input_batch, input_channel)
        out = torch.swapaxes(out, 0, 2)
        out = torch.swapaxes(out, 1, 3)

        print(f"out size: {out.size()}")
        return out
        """

"""
        # gets the attention weights and repeats them to have the same size as the total channels
        attn_output_weights = torch.swapaxes(attn_output_weights, 0, 1)
        attn_output_weights = attn_output_weights.repeat(1, 1, input_channel)

        # multiplies the attention to focus the vision rep based on the lang rep
        vision_rep = vision_rep * attn_output_weights
        vision_rep = vision_rep.contiguous()

        # rearanges the output matrix to be the dimensions of the input
        out = vision_rep.view(input_width, input_height, input_batch, input_channel)
        out = torch.swapaxes(out, 0, 2)
        out = torch.swapaxes(out, 1, 3)
        return out
        """