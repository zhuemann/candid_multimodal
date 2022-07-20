
import numpy as np

import os
import cv2

def visualization_attention(img, vision_rep, lang_rep, att_matrix, target_batch):

    print("hi")

    print(f"lang_rep size: {lang_rep.size()}")
    print(f"vision_rep size: {vision_rep.size()}")
    print(f"att_matix size: {att_matrix.size()}")
    print(f"img size: {img.size()}")
    print(f"targets size: {target_batch.size()}")

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

    # visualizes the attention matrices
    att_img = att_matrix.cpu().detach().numpy()
    for i in range(0,input_channel):
        img_ch = att_img[:,0,i]
        img_ch = np.reshape(img_ch, (input_width, input_height))
        max = np.amax(img_ch)
        min = np.amin(img_ch)
        #print(f"max: {max}")
        #print(f"min: {min}")
        img_ch = (img_ch * 255) / max
        fullpath = os.path.join(dir_base, 'Zach_Analysis/dgx_images/GE meeting/word_attention/img_ch'+str(i) + '.png')
        cv2.imwrite(fullpath, img_ch)

    fullpath = os.path.join(dir_base,
                            'Zach_Analysis/dgx_images/GE meeting/input' + '.png')
    img = img.cpu().detach().numpy()
    img = img[0,0,:,:]
    print(f"img shape: {np.shape(img)}")
    cv2.imwrite(fullpath, img)

    target_batch = target_batch.cpu().detach().numpy()
    print(f"target_batch: {np.shape(target_batch)}")
    fullpath = os.path.join(dir_base,
                            'Zach_Analysis/dgx_images/GE meeting/target' + '.png')
    cv2.imwrite(fullpath, target_batch[0,:,:])

    lang_rep = lang_rep.cpu().detach().numpy()
    fullpath = os.path.join(dir_base,
                            'Zach_Analysis/dgx_images/GE meeting/lang_rep' + '.png')
    cv2.imwrite(fullpath, lang_rep[0,:,:])