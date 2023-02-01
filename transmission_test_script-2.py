import ast
import sys
import os
import pandas as pd
import numpy as np

# saving path
root_path = sys.argv[0]
main_dir, _ = os.path.split(root_path)
main_saving_path = os.path.join(main_dir, 'results')

dir_name = 'test-1'
test_path = os.path.join(main_saving_path, dir_name)

df = pd.DataFrame(columns=['DR', 'Fs', 'CHs', 'Diffs'])

result_filename = 'multi_param_test.txt'

for root, _, dirs in os.walk(test_path):

    for file in dirs:

        if file != result_filename and file != '.gitignore':

            file_str_l = file.split("_")

            dr = int(file_str_l[4])
            fs = file_str_l[5]
            sub_str = file_str_l[6]
            ch_str = sub_str.split(".")[0]
            ch = len(ast.literal_eval(ch_str))

            file_path = os.path.join(root, file)

            try:
                data = pd.read_csv(file_path, delimiter='\t', skiprows=1)
                seq_diffs_1 = np.unique(np.diff(np.squeeze(np.array(data[['#NSeq']]))))

                if seq_diffs_1.shape[0] == 2 and np.min(seq_diffs_1) == -4095 and np.max(seq_diffs_1) == 1:
                    # No error in data transmission
                    str_result = 'OK'
                else:
                    # Discrepant Differences (DD)
                    str_result = 'DD'

                df_tmp = pd.DataFrame({'DR': [dr],
                                       'Fs': [fs],
                                       'CHs': [ch],
                                       'Diffs': [str_result]})
                df = df.append(df_tmp, ignore_index=True)

            except:

                with open(file_path) as f:
                    lines = f.readlines()
                    print(lines)
                    detect_str_1 = '    b = data[i]\n'
                    detect_str_2 = 'IndexError: list index out of range\n'

                    if detect_str_1 in lines and detect_str_2 in lines:
                        df_tmp = pd.DataFrame({'DR': [dr],
                                               'Fs': [fs],
                                               'CHs': [ch],
                                               # Index Error (IE)
                                               'Diffs': ["IE"]})
                        df = df.append(df_tmp, ignore_index=True)

# sort rows by data rate
df = df.sort_values(by=['DR'])

# save data frame with test-results
df.to_csv(os.path.join(test_path, result_filename))
