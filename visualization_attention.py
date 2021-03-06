
import numpy as np

import os
import cv2

def visualization_attention(img, vision_rep, lang_rep, att_matrix, target_batch, model_output):

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
    vis_mat = vision_rep.cpu().detach().numpy()
    for i in range(0,input_channel):
        img_ch = att_img[:,0,i]
        img_ch = np.reshape(img_ch, (input_width, input_height))
        max = np.amax(img_ch)
        min = np.amin(img_ch)
        #print(f"max: {max}")
        #print(f"min: {min}")
        img_ch = (img_ch * 255) / max
        fullpath = os.path.join(dir_base, 'Zach_Analysis/dgx_images/GE meeting/word_attention/attention_ch'+str(i) + '.png')
        cv2.imwrite(fullpath, img_ch)

        vis_ch = vis_mat[0,i,:,:]
        vis_ch = (vis_ch*255)/np.amax(vis_ch)
        fullpath = os.path.join(dir_base, 'Zach_Analysis/dgx_images/GE meeting/vis_ch/vis_ch' + str(i) + '.png')
        cv2.imwrite(fullpath, vis_ch)



    fullpath = os.path.join(dir_base,
                            'Zach_Analysis/dgx_images/GE meeting/input' + '.png')
    img = img.cpu().detach().numpy()
    img = img[0,0,:,:]
    img = (img*255)/np.amax(img)
    print(f"img shape: {np.shape(img)}")
    cv2.imwrite(fullpath, img)

    target_batch = target_batch.cpu().detach().numpy()
    target_batch = (target_batch*255)/np.amax(target_batch)
    print(f"target_batch: {np.shape(target_batch)}")
    fullpath = os.path.join(dir_base,
                            'Zach_Analysis/dgx_images/GE meeting/target' + '.png')
    cv2.imwrite(fullpath, target_batch[0,:,:])

    lang_rep = lang_rep.cpu().detach().numpy()
    lang_rep = (lang_rep*255)/np.amax(lang_rep)
    fullpath = os.path.join(dir_base,
                            'Zach_Analysis/dgx_images/GE meeting/lang_rep' + '.png')
    cv2.imwrite(fullpath, lang_rep[0,:,:])

    model_output = model_output.cpu().detach().numpy()
    model_output = (model_output * 255) / np.amax(model_output)
    print(f"model_output: {np.shape(model_output)}")
    fullpath = os.path.join(dir_base,
                            'Zach_Analysis/dgx_images/GE meeting/model_output' + '.png')
    cv2.imwrite(fullpath, model_output[0,0, :, :])

