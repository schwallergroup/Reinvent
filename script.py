import os
import subprocess

#configs = ['baseline_rl.json', 'cl_gradual_gradients.json']
configs = ['double_rl_no_filter_2.json', 'double_rl_no_filter_3.json',
           'cl_gradual_gradients_double_rl_no_filter_1.json']

BASE_PATH = '/home/jeguo/Desktop/OE62/ip_experiments_polished'

for conf in configs:
    path = os.path.join(BASE_PATH, conf)
    subprocess.run(['python', 'input.py', path])
