
import numpy as np

import os
import cv2
import torch
import copy

def visualization_attention(img, vision_rep_before, vision_rep, lang_rep, att_matrix, target_batch, model_output):

    print("hi")

    print(f"lang_rep size: {lang_rep.size()}")
    print(f"vision_rep size: {vision_rep.size()}")
    print(f"att_matix size: {att_matrix.size()}")
    print(f"img size: {img.size()}")
    print(f"targets size: {target_batch.size()}")
    print(f"vis rep_before size: {vision_rep_before.size()}")

    input_channel = vision_rep.size()[1]
    input_width = vision_rep.size()[2]
    input_height = vision_rep.size()[3]

    # print(f"attension matrix max: {torch.max(att_matrix)}")
    # print(f"attension matrix min: {torch.min(att_matrix)}")

    # print(f"max: {torch.max(attn_output_weights)}")
    # print(f"min: {torch.min(attn_output_weights)}")

    # print("attn_output weights")
    # print(attn_output_weights.size())
    # print("vis_rep")
    # print(vision_rep.size())

    # visualize attention maps
    # img = att_matrix.cpu().detach().numpy()

    # img = img[0,0,:]
    # img2 = img[:,0,1]

    # print(f"all the elements for one batch {np.shape(img)}")

    # img = np.reshape(img, (input_width, input_height))
    # img = np.reshape(img, (input_channel, 1))

    # max = np.amax(img)
    # min = np.amin(img)
    # print(f"max: {max}")
    # print(f"min: {min}")
    # print(np.shape(img))
    # img = (img * 255) / max
    dir_base = "/UserData/"
    # fullpath = os.path.join(dir_base, 'Zach_Analysis/dgx_images/attention_visualize/test_img' + '.png')
    # cv2.imwrite(fullpath, img)

    fullpath = os.path.join(dir_base,
                            'Zach_Analysis/dgx_images/visualizations_for_paper/input' + '.png')
    img = img.cpu().detach().numpy()
    img = img[0, 0, :, :]
    img = (img * 255) / np.amax(img)
    print(f"img shape: {np.shape(img)}")
    cv2.imwrite(fullpath, img)

    target_batch = target_batch.cpu().detach().numpy()
    target_batch = (target_batch * 255) / np.amax(target_batch)
    print(f"target_batch: {np.shape(target_batch)}")
    fullpath = os.path.join(dir_base,
                            'Zach_Analysis/dgx_images/visualizations_for_paper/target' + '.png')
    cv2.imwrite(fullpath, target_batch[0, 0, :, :])

    lang_rep = lang_rep.cpu().detach().numpy()
    lang_rep = (lang_rep * 255) / np.amax(lang_rep)
    fullpath = os.path.join(dir_base,
                            'Zach_Analysis/dgx_images/visualizations_for_paper/lang_rep' + '.png')
    cv2.imwrite(fullpath, lang_rep[0, :, :])

    sigmoid = torch.sigmoid(model_output)
    model_output = torch.round(sigmoid)
    model_output = model_output.cpu().detach().numpy()
    model_output = (model_output * 255) / np.amax(model_output)
    print(f"model_output: {np.shape(model_output)}")
    fullpath = os.path.join(dir_base,
                            'Zach_Analysis/dgx_images/visualizations_for_paper/model_output' + '.png')
    cv2.imwrite(fullpath, model_output[0, 0, :, :])

    # visualizes the attention matrices
    att_img = att_matrix.cpu().detach().numpy()
    vis_mat = vision_rep.cpu().detach().numpy()
    vis_before_mat = vision_rep_before.cpu().detach().numpy()
    for i in range(0,input_channel):
        img_ch = att_img[:,0,i]
        img_ch = np.reshape(img_ch, (input_width, input_height))
        max = np.amax(img_ch)
        min = np.amin(img_ch)
        #print(f"max: {max}")
        #print(f"min: {min}")
        colorize_img(img_ch)
        img_ch_scale = img_ch+abs(min)
        img_ch = (img_ch_scale*255)/np.amax(img_ch_scale)
        fullpath = os.path.join(dir_base, 'Zach_Analysis/dgx_images/visualizations_for_paper/attention_ch/attention_ch'+str(i) + '.png')
        cv2.imwrite(fullpath, img_ch)

        vis_ch = vis_mat[0,i,:,:]
        #vis_ch = (vis_ch*255)/np.amax(vis_ch)
        vis_ch_scale = vis_ch + abs(np.amin(vis_ch))
        vis_ch = (vis_ch_scale * 255) / np.amax(vis_ch_scale)
        fullpath = os.path.join(dir_base, 'Zach_Analysis/dgx_images/visualizations_for_paper/vis_ch/vis_ch' + str(i) + '.png')
        cv2.imwrite(fullpath, vis_ch)

        vis_before = vis_before_mat[0,i,:,:]
        #vis_ch_before = (vis_before * 255) / np.amax(vis_before)
        vis_before_scale = vis_before + abs(np.amin(vis_before))
        vis_ch_before = (vis_before_scale * 255) / np.amax(vis_before_scale)
        fullpath = os.path.join(dir_base,
                                'Zach_Analysis/dgx_images/visualizations_for_paper/vis_ch_before/vis_ch_before' + str(i) + '.png')
        cv2.imwrite(fullpath, vis_ch_before)



def colorize_img(img):

    colorized_img_pos = copy.deepcopy(img)
    colorized_img_neg = copy.deepcopy(img)

    colorized_img_pos[colorized_img_pos < 0] = 0
    colorized_img_neg[colorized_img_neg > 0] = 0
    colorized_img_neg = colorized_img_neg * -1

    colorized_img = np.concatenate((colorized_img_pos,  colorized_img_neg), axis = 1)
    print(f"coloriaed_img size: {colorized_img.shape}")

    return colorized_img