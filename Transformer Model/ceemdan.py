from PyEMD import CEEMDAN
import pandas as pd


def calculate_ceemdan(dataframe, target_col):
    signal = dataframe[target_col].values

    ceemdan = CEEMDAN()

    imfs = ceemdan(signal)  # decomposing
    imfs, res = ceemdan.get_imfs_and_residue()

    result_df = pd.DataFrame({'Residue': res})

    for i, imf in enumerate(imfs):
        result_df[f'IMF_{i + 1}'] = imf

    return result_df
