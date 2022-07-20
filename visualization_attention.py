
def visualization_attention(img, decode1, lang_rep1, att_matrix, target_batch):

    print("hi")

    print(f"lang_rep size: {lang_rep1.size()}")
    print(f"decode1 size: {decode1.size()}")
    print(f"att_matix size: {att_matrix.size()}")
    print(f"img size: {img.size()}")
    print(f"targets size: {target_batch.size()}")



    # print(f"attention matrix: {att_matrix.size()}")
    # print(f"attention_output_weight {attn_output_weights.size()}")
    # print(f"vision rep: {vision_rep.size()}")

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
    # dir_base = "/UserData/"
    # fullpath = os.path.join(dir_base, 'Zach_Analysis/dgx_images/attention_visualize/test_img' + '.png')
    # cv2.imwrite(fullpath, img)

    # visualizes the attention matrices
    # img = att_matrix.cpu().detach().numpy()
    # for i in range(0,input_channel):
    #    img_ch = img[:,0,i]
    #    img_ch = np.reshape(img_ch, (input_width, input_height))
    #    max = np.amax(img_ch)
    # min = np.amin(img_ch)
    # print(f"max: {max}")
    # print(f"min: {min}")
    #    img_ch = (img_ch * 255) / max
    #    fullpath = os.path.join(dir_base, 'Zach_Analysis/dgx_images/attention_visualize/word_attention/test_img_ch'+str(i) + '.png')
    #    cv2.imwrite(fullpath, img_ch)