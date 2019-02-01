import os
import pickle
from contextlib import suppress

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

datasets_folder = 'datasets'
log_folder = 'csv_logs'
dump_folder = 'dump_logs'

with suppress(Exception):
    os.mkdir(dump_folder)


for fname in tqdm(os.listdir(log_folder)):

    with open(os.path.join(log_folder, fname)) as f:
        lines = f.readlines()

    data = {}
    for line in lines:
        n_c_clusters = 0
        n_p_clusters = 0
        n_o_clusters = 0
        partial = {'c_clusters': [], 'p_clusters': [], 'o_clusters': []}
        parts = line.split(';')
        instance = int(parts[0].split(' ')[1])
        i = 1
        part = parts[i]
        while part != '\n':
            if 'c_cluster' in part:
                cluster_type = 'c_clusters'
                n_c_clusters += 1
            elif 'p_cluster' in part:
                cluster_type = 'p_clusters'
                n_p_clusters += 1
            elif 'o_cluster' in part:
                cluster_type = 'o_clusters'
                n_o_clusters += 1
            id_ = int(parts[i + 1])
            centroid = np.fromstring(parts[i + 2][1:-1], sep=' ', dtype=float)
            radius = float(parts[i + 3])
            weight = float(parts[i + 4])
            partial[cluster_type].append({'id': id_, 'centroid': centroid,
                                          'radius': radius, 'weight': weight})
            i += 5  # skip centroid radius and weight
            part = parts[i]

        partial['n_c_clusters'] = n_c_clusters
        partial['n_p_clusters'] = n_p_clusters
        partial['n_o_clusters'] = n_o_clusters

        data[instance] = partial

    name = fname.replace('.csv', '.pkl')
    pickle.dump(data, open(os.path.join(dump_folder, name), 'wb'))
