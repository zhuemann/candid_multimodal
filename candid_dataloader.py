from os import listdir
from os.path import isfile, join
from os.path import exists
import pandas as pd
import os

def get_candid_labels(dir_base = "/home/zmh001/r-fcb-isilon/research/Bradshaw/"):


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
    i = 0
    missing_reports = 0

    num_fractures = 0
    num_tubes = 0
    num_pneumothorax = 0
    total = 0
    for image in xray_files:
        # checks to see if an xray image is in the fracture list
        if fracture_df['anon_SOPUID'].str.contains(image).any():
            data_with_labels.loc[i] = [image, image, "", 0]
            num_fractures += 1
            i = i + 1
        elif chest_tube_df['anon_SOPUID'].str.contains(image).any():
            data_with_labels.loc[i] = [image, image, "", 1]
            num_tubes += 1
            i = i + 1
        elif pneumothorax_df['SOPInstanceUID'].str.contains(image).any():
            # data_with_labels.loc[i] = [image, image, "", 2]
            num_pneumothorax += 1
        else:
            total += 1
            # i = i - 1
        # i = i + 1




    print(num_fractures)
    print(num_tubes)
    print(num_pneumothorax)
    print(total)
    #print(xray_files)

    data_with_labels.set_index("id", inplace=True)
    return data_with_labels