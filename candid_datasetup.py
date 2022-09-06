import os
from os import listdir

import pandas as pd
import numpy as np

from utility import rle_decode_modified, mask2rle, rle_decode


def get_candid_labels(dir_base="Z:/"):
    xray_dir = os.path.join(dir_base, "public_datasets/candid_ptx/dataset1/dataset/")
    fracture_dir = os.path.join(dir_base, "public_datasets/candid_ptx/acute_rib_fracture.csv")
    fracture_df = pd.read_csv(fracture_dir)
    print("fracture_df")
    chest_tube_dir = os.path.join(dir_base, "public_datasets/candid_ptx/chest_tube.csv")
    chest_tube_df = pd.read_csv(chest_tube_dir)
    pneumothorax_dir = os.path.join(dir_base, "public_datasets/candid_ptx/Pneumothorax_reports.csv")
    pneumothorax_df = pd.read_csv(pneumothorax_dir)

    # gets all the file names in and puts them in a list
    xray_files = listdir(xray_dir)
    print("xray_files")

    data_with_labels = pd.DataFrame(columns=['id', 'image_id', 'text', 'label'])
    i = 0
    missing_reports = 0
    num_fractures = 0
    num_tubes = 0
    num_pneumothorax = 0
    total = 0
    for image in xray_files:

        if pneumothorax_df['SOPInstanceUID'].str.contains(image).any():
            # continue
            text = get_text(pneumothorax_df, image)
            mask = pneumothorax_df.loc[pneumothorax_df['SOPInstanceUID'] == image]
            mask_str = mask.iloc[0]['EncodedPixels']
            if mask_str == "-1":
                continue
            else:
                data_with_labels.loc[i] = [image, image, text, mask_str]
                num_pneumothorax += 1
                i = i + 1
        # checks to see if an xray image is in the fracture list
        elif fracture_df['anon_SOPUID'].str.contains(image).any():
            continue
            mask = fracture_df.loc[fracture_df['anon_SOPUID'] == image]
            mask_str = mask.iloc[0]['mask_rle']
            if mask_str == "-1":
                continue
            else:
                data_with_labels.loc[i] = [image, image, "", mask_str]
                num_fractures += 1
                i = i + 1
        elif chest_tube_df['anon_SOPUID'].str.contains(image).any():
            continue
            mask = chest_tube_df.loc[chest_tube_df['anon_SOPUID'] == image]
            mask_str = mask.iloc[0]['mask_rle']
            if mask_str == "-1":
                continue
            else:
                data_with_labels.loc[i] = [image, image, "", mask_str]
                num_tubes += 1
                i = i + 1
        else:
            total += 1

    data_with_labels.set_index("id", inplace=True)
    return data_with_labels

# get images of pneumothorax which includes cases of double pneumothorax
def get_pneumothorax_image(dir_base="Z:/"):
    xray_dir = os.path.join(dir_base, "public_datasets/candid_ptx/dataset1/dataset/")
    fracture_dir = os.path.join(dir_base, "public_datasets/candid_ptx/acute_rib_fracture.csv")
    fracture_df = pd.read_csv(fracture_dir)
    print("pnuemo images")
    chest_tube_dir = os.path.join(dir_base, "public_datasets/candid_ptx/chest_tube.csv")
    chest_tube_df = pd.read_csv(chest_tube_dir)
    pneumothorax_dir = os.path.join(dir_base, "public_datasets/candid_ptx/Pneumothorax_reports.csv")
    pneumothorax_df = pd.read_csv(pneumothorax_dir)

    counter_list = []
    same_index = 0
    duplicate_df = pd.DataFrame()
    for i in range(0,len(pneumothorax_df) - 1):
        #print(pneumothorax_df.iloc[i]['SOPInstanceUID'])
        # if the row is unique continue looping through
        #print(f"i: {i}")
        #print(f"id: {pneumothorax_df.iloc[i]['SOPInstanceUID']}")

        if pneumothorax_df.iloc[i]['SOPInstanceUID'] == "8.5.323.562604.1.588.4.7008319959.7387629250.60820":
            continue
        if pneumothorax_df.iloc[i]['SOPInstanceUID'] == "7.8.081.133986.9.156.6.2371195176.7322935138.065936":
            continue
        if pneumothorax_df.iloc[i]['SOPInstanceUID'] == "9.8.974.569526.1.617.1.1894012707.3171487907.766466":
            continue
        if pneumothorax_df.iloc[i]['SOPInstanceUID'] == "1.4.435.094506.7.695.5.6026509150.6747477954.924906":
            continue
        if pneumothorax_df.iloc[i]['SOPInstanceUID'] == "6.8.999.403705.5.598.0.5587688844.2801490257.264789":
            continue
        if pneumothorax_df.iloc[i]['SOPInstanceUID'] == "1.4.684.748624.5.222.5.5068201957.1343821255.254562":
            continue
        if pneumothorax_df.iloc[i]['SOPInstanceUID'] == "6.6.642.476817.4.384.9.0073098212.3281820063.964794":
            continue

        if pneumothorax_df.iloc[i]['SOPInstanceUID'] != pneumothorax_df.iloc[i+1]['SOPInstanceUID']:
            rle = pneumothorax_df.iloc[i]['EncodedPixels']
            if rle == '-1':
                continue
            mask = rle_decode_modified(rle, (1024, 1024))
            pneumothorax_df.iloc[i]['EncodedPixels'] = mask2rle(mask)
            #continue
        else:
            # calculate masks for entries that have multiple pnuemothorx per image

            # counts how many pnuemothorax segmentations there are, counter is number of consecutive masks with same
            # image in the dataset because all muliple pnuemothorax are next to each other

            counter = 0
            while pneumothorax_df.iloc[i]['SOPInstanceUID'] == pneumothorax_df.iloc[i + counter + 1]['SOPInstanceUID']:
                counter += 1
                if counter > 1:
                    print(f"counter: {counter}")
                    print(pneumothorax_df.iloc[i])

            # adds all the individual segmentations as masks to an empty target
            composit_target = np.zeros((1024, 1024))
            #print(pneumothorax_df.iloc[i])
            for j in range(0,counter):
                rle = pneumothorax_df.iloc[i+j]['EncodedPixels']
                #if rle == "-1":
                #    continue
                #print(f"rle: {rle}")
                composit_target += rle_decode_modified(rle, (1024, 1024))

            # if the values is above 1 for any reason then set it to 1
            composit_target[composit_target > 1] = 1

            # converts that composite image to rle
            composit_rle = mask2rle(composit_target)

            # saves this rle for every entry so that they are all identical
            for j in range(0, counter):
                pneumothorax_df.iloc[i + j]['EncodedPixels'] = composit_rle
            #print(f"id: {pneumothorax_df.iloc[i]['SOPInstanceUID']}")

            #print(counter)
            counter_list.append(counter)
            same_index += 1

    #print(pneumothorax_df)
    #print(f"same indexes: {same_index}")
    #print(f"sum of counter list: {np.sum(counter_list)}")
    # gets all the file names in and puts them in a list
    xray_files = listdir(xray_dir)
    print(f"length of xrays: {len(xray_files)}")
    print("xray_files")

    data_with_labels = pd.DataFrame(columns=['id', 'image_id', 'text', 'label'])
    i = 0
    num_pneumothorax = 0
    total = 0
    negative_cases = 0
    for image in xray_files:

        if pneumothorax_df['SOPInstanceUID'].str.contains(image).any():
            # continue
            text = get_text(pneumothorax_df, image)
            mask = pneumothorax_df.loc[pneumothorax_df['SOPInstanceUID'] == image]
            mask_str = mask.iloc[0]['EncodedPixels']
            # only positive cases
            if mask_str == "-1":
                negative_cases += 1
                if negative_cases > 3196:
                    continue
            #else:
            #    data_with_labels.loc[i] = [image, image, text, mask_str]
            #    num_pneumothorax += 1
            #    i = i + 1
            data_with_labels.loc[i] = [image, image, text, mask_str]
            num_pneumothorax += 1
            i = i + 1

        else:
            total += 1
    print(f"num_pneumo found: {num_pneumothorax}")
    data_with_labels.set_index("id", inplace=True)
    return data_with_labels


# gets the text from the reports which matches the file_check argument
def get_text(reports, file_check):
    row = reports.loc[reports['SOPInstanceUID'] == file_check]
    text = row['Report']
    text = text.iloc[0]
    return text


def get_all_text_image_pairs(dir_base="/home/zmh001/r-fcb-isilon/research/Bradshaw/"):
    xray_dir = os.path.join(dir_base, "public_datasets/candid_ptx/dataset1/dataset")
    fracture_dir = os.path.join(dir_base, "public_datasets/candid_ptx/acute_rib_fracture.csv")
    fracture_df = pd.read_csv(fracture_dir)
    chest_tube_dir = os.path.join(dir_base, "public_datasets/candid_ptx/chest_tube.csv")
    chest_tube_df = pd.read_csv(chest_tube_dir)
    pneumothorax_dir = os.path.join(dir_base, "public_datasets/candid_ptx/Pneumothorax_reports.csv")
    pneumothorax_df = pd.read_csv(pneumothorax_dir)

    # gets all the file names in and puts them in a list
    xray_files = listdir(xray_dir)

    data_with_labels = pd.DataFrame(columns=['id', 'image_id', 'text', 'label'])
    df_index = 0
    missing_reports = 0
    num_fractures = 0
    num_tubes = 0
    num_pneumothorax = 0
    total = 0
    for image in xray_files:
        if pneumothorax_df['SOPInstanceUID'].str.contains(image).any():
            # continue
            text = get_text(pneumothorax_df, image)
            mask = pneumothorax_df.loc[pneumothorax_df['SOPInstanceUID'] == image]
            mask_str = mask.iloc[0]['EncodedPixels']
            data_with_labels.loc[df_index] = [image, image, text, mask_str]
            num_pneumothorax += 1
            df_index = df_index + 1
        # checks to see if an xray image is in the fracture list
        # elif fracture_df['anon_SOPUID'].str.contains(image).any():
        #    continue
        #    mask = fracture_df.loc[fracture_df['anon_SOPUID'] == image]
        #    mask_str = mask.iloc[0]['mask_rle']
        #    if mask_str == "-1":
        #        continue
        #    else:
        #        data_with_labels.loc[i] = [image, image, "", mask_str]
        #        num_fractures += 1
        #        df_index = df_index + 1
        # elif chest_tube_df['anon_SOPUID'].str.contains(image).any():
        #    continue
        #    mask = chest_tube_df.loc[chest_tube_df['anon_SOPUID'] == image]
        #    mask_str = mask.iloc[0]['mask_rle']
        #    if mask_str == "-1":
        #        continue
        #    else:
        #        data_with_labels.loc[i] = [image, image, "", mask_str]
        #        num_tubes += 1
        #        df_index = df_index + 1
        # else:
        #    total += 1

    data_with_labels.set_index("id", inplace=True)
    return data_with_labels


def main():
    get_candid_labels()


if __name__ == "__main__":
    main()
