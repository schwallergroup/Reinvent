import os
import subprocess

#configs = ['cl_1.json', 'cl_2.json', 'cl_3.json']

#BASE_PATH = '/home/jeguo/Desktop/OE62/ip_experiments_polished'


#configs = ['cl_lumo_config_1.json', 'cl_lumo_config_2.json', 'cl_lumo_config_3.json']

#BASE_PATH = '/home/jeguo/Desktop/augmented_experience_replay/true_exploration'

for base_path in ['/home/jeguo/Desktop/augmented_experience_replay/tanimoto/rescue',
                  '/home/jeguo/Desktop/augmented_experience_replay/tanimoto/exploit',
                  '/home/jeguo/Desktop/augmented_experience_replay/tanimoto/only_df'
                  ]:
    configs = os.listdir(base_path)

    for config in configs:
        path = os.path.join(base_path, config)
        subprocess.run(['python', 'input.py', path])
