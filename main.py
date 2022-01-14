

from training_loop import training_loop
import pandas as pd
import os


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    local = False
    if local == True:
        directory_base = "Z:/"
    else:
        directory_base = "/home/zmh001/r-fcb-isilon/research/Bradshaw/"

    DGX = True
    if DGX == True:
        directory_base = "/UserData/"

    seeds = [117, 295, 98, 456, 915, 1367, 712]
    accuracy_list = []

    for seed in seeds:

        acc, matrix = training_loop(seed = seed, batch_size = 2, dir_base= directory_base, epoch = 10, n_classes = 2)
        accuracy_list.append(acc)

        df = pd.DataFrame(matrix)
        file_name = 'first_vision_run'
        ## save to xlsx file
        filepath = os.path.join(directory_base,
                                '/UserData/Zach_Analysis/result_logs/candid_result/tests/' + str(file_name) +'/confusion_matrix_seed' + str(
                                    seed) + '.xlsx')

        df.to_excel(filepath, index=False)

    print(accuracy_list)

