import inspect
import json
import os
import numpy as np
import pandas as pd
def print_latex_generic(df: pd.DataFrame, float_fmt=2):
    """
    Print a dataframe as LaTeX rows with & separators, generic for any columns.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame.
    float_fmt : int
        Number of decimals for floats.
    """""
    # ---- Print header ----
    header = " & ".join(df.columns) + " \\\\"
    print(header)
    print("\\hline")

    for _, row in df.iterrows():
        row_items = []
        for col in df.columns:
            val = row[col]
            # format floats
            if isinstance(val, (float, np.float64)):
                val_str = f"{val:.{float_fmt}f}"
            else:
                val_str = str(val)
            row_items.append(val_str)
        # join with & and add \\ for LaTeX
        print(" & ".join(row_items) + " \\\\")


def activate_pd_full_print():
    """
    Activates pandas full printing options.
    """""
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)  # prevents line wrapping
    pd.set_option("display.max_colwidth", None)  # shows full cell content


def get_signals_as_dict(signal_path: str, device_name='bitalino', bit_signal_map=None):
    """
    Reads the signals collected from the BITalino or Sympathia devices, using the file path directly.
    
    Parameters
    ----------
    signal_path : str
        Path to signals file.
    device_name : str
        Device name 'bitalino' or 'sympathia'.
    bit_signal_map : dict
        Optional mapping of BITalino channels to signal names, e.g. {"A1": "EDA", "A2": "LUX"}. If None, defaults to this mapping.

    Returns
    -------
    signals : dict
        Resulting dictionary containing the signals.
    """""

    signals = {'sympathia': {}, 'bitalino': {}}

    if device_name == 'bitalino':
        bitalino_data = np.loadtxt(signal_path, skiprows=3)
        bitalino_mdata = bitalino_file_header_as_dict(signal_path)

        if bit_signal_map is None:
            bit_signal_map = {
                "A1": "EDA",
                "A2": "LUX"
            }

        for channel, modal in bit_signal_map.items():
            bitalino_columns = bitalino_mdata['column']
            for i in range(len(bitalino_columns)):
                if channel == bitalino_columns[i]:
                    signals['bitalino'][modal] = bitalino_data[:, i]

    elif device_name == 'sympathia':

        scientisst_data = np.loadtxt(signal_path, skiprows=2)
        scientisst_columns = get_sympathia_header(signal_path)

        for i in range(len(scientisst_columns)):
            if scientisst_columns[i] == 'O2':
                signals['sympathia']['LED'] = scientisst_data[:, i]  # O2 = LED, so we re-name it here
            else:
                signals['sympathia'][scientisst_columns[i]] = scientisst_data[:, i]

    else:
        print("Sympathia signals path not provided.")

    return signals


def get_sympathia_header(data_path: str):
    """Obtains the header of the Sympathia data file, which is in the second line, and returns it as a list of column names.
    
    Parameters
    ----------
    data_path : str
        Path to signals file.
        
    Returns
    -------
    lines : dict
        Resulting dictionary containing the signals.
    """""
    # read first three lines. The second is the mdata.
    with open(data_path, "r") as file:
        lines = [next(file).strip() for i in range(2)][1]

    # remove initial char, and convert single quote to double
    lines = lines.split('#')[1].split('\t')

    return lines


def bitalino_file_header_as_dict(bitalino_data_path: str):
    """Computes patient classification performance in a comprehensive manner (AUC, Accuracy, F1-score).

    Parameters
    ----------
    data_path : str
        Path to data.
    win_len : int
        Window length (in seconds, used in feature extraction).
    win_overlap : float
        Window overlap (decimal, used in feature extraction).
    feature_sels : list
        List containing the feature selection ('eda', 'edr', or 'all' for both).
    norm_method : str
        Normalization method used in feature extraction.

    Returns
    -------
    df_results : pd.DataFrame
        Resulting dataframe.
    """""
    # read first three lines. The second is the mdata.
    with open(bitalino_data_path, "r") as file:
        lines = [next(file).strip() for i in range(3)][1]

    # remove initial char
    lines = lines.split('# ')[1]

    # convert to dictionary
    mdata_dict = json.loads(lines)

    mac_address = list(mdata_dict.keys())[0]

    return mdata_dict[mac_address]


# =========== Unused but useful functions ===========

def save_dict_as_json(saving_filepath: str, data: dict):
    """
    Saves the content of a Dictionary in a Json file.
    
    Parameters
    ----------
    dir_path : str
        Path to data.

    """""
    with open(saving_filepath, 'w') as fp:
        json.dump(data, fp)


def get_script_path():
    """
    Returns the absolute path of the script calling this function.
    
    Returns
    -------
    abs_path : str
        Path of script.
    """""
    # Get the frame of the caller (the script calling this function)
    frame = inspect.stack()[1]
    # Get the filename (script name) from the frame
    script_name = frame[0].f_code.co_filename
    # Return the absolute path of the script
    abs_path = os.path.abspath(script_name)

    return abs_path


def get_parent_folder_path():
    """
    Returns the absolute path of the parent folder of the script in which this function is called.
    
    Returns
    -------
    parent_folder_path : str
        Path of the parent folder.
    """""
    # Get the frame of the caller (the script calling this function)
    frame = inspect.stack()[1]
    # Get the filename (script name) from the frame
    script_name = frame[0].f_code.co_filename

    # Return the absolute path of the script
    current_script = os.path.abspath(script_name)

    parent_folder_path, _ = os.path.split(current_script)

    return parent_folder_path


def list_folders_in_dir(dir_path):
    """Gets a list with folder paths and the folder names.

        Example:
            [[<folder_path_1>, <folder_name_1>],
            ...
            [<folder_path_N>, <folder_name_N>]]
    """""

    directories = []

    for entry in os.listdir(dir_path):
        entry_path = os.path.join(dir_path, entry)
        if os.path.isdir(entry_path):
            directories.append([entry_path, entry])

    sorted_directories = sorted(directories, key=lambda x: x[1])

    return sorted_directories


def list_csv_files_in_dir(dir_path: str):
    """
    Gets a list with CSV file paths and file names.

        Example:
            [[<file_path_1>, <file_name_1>],
            ...
            [<file_path_N>, <file_name_N>]]
            
    Parameters
    ----------
    dir_path : str
        Path to data.

    Returns
    -------
    csv_files : list
        List of .csv files ([file_path, file_name]) in the directory.
    """""

    csv_files = []

    for entry in os.listdir(dir_path):
        entry_path = os.path.join(dir_path, entry)
        if os.path.isfile(entry_path) and entry.lower().endswith('.csv'):
            csv_files.append([entry_path, entry])

    csv_files = sorted(csv_files, key=lambda x: x[1])

    return csv_files
