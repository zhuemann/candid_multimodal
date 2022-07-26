import os
from os import listdir

import pandas as pd


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

    # gets all the file names in and puts them in a list
    xray_files = listdir(xray_dir)
    print(f"length of xrays: {len(xray_files)}")
    print("xray_files")

    data_with_labels = pd.DataFrame(columns=['id', 'image_id', 'text', 'label'])
    i = 0
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
        else:
            total += 1

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
