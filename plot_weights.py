import math
import os
import pickle
from collections import defaultdict
from contextlib import suppress

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

datasets_folder = 'datasets'
dump_folder = 'dump_logs'
dump_files = os.listdir(dump_folder)
plot_folder = 'plots_weights'

with suppress(Exception):
    os.mkdir(plot_folder)

for dump_file in tqdm(os.listdir(dump_folder)):
    dump_file = os.path.join(dump_folder, dump_file)
    data = pickle.load(open(dump_file, 'rb'))

    dataset = os.path.splitext(os.path.basename(dump_file))[0].split('_lambda')[0]
    dfname = os.path.join(datasets_folder, dataset) + '.csv'

    df = pd.read_csv(dfname)

    changes = []
    current_c = 0
    for i, c in enumerate(df['class'].values):
        if c != current_c:
            current_c = c
            changes.append((c, i))

    plt.figure(figsize=(8, 6))
    starts = {}
    clusters_weights = defaultdict(list)
    for instance, partial in data.items():
        for c in partial['c_clusters']:
            if c['id'] not in clusters_weights:
                starts[c['id']] = instance
            clusters_weights[c['id']].append(math.log(c['weight']))

        for c in partial['p_clusters']:
            if c['id'] not in clusters_weights:
                starts[c['id']] = instance
            clusters_weights[c['id']].append(math.log(c['weight']))

    cmap = matplotlib.cm.get_cmap('Spectral')
    n_colors = len(clusters_weights)
    step = 1 / n_colors
    colors = [cmap(step * i) for i in range(n_colors)]

    for (id_, weights), color in zip(clusters_weights.items(), colors):
        start = starts[id_]
        x = [start]
        for i in range(len(weights) - 1):
            x.append(50 + x[-1])
        plt.plot(x, weights, c=color, label=f'Cluster id {id_}', linewidth=1.8)

    if changes:
        for infection_type, change in changes:
            if infection_type == 0:
                continue
            if infection_type == 1:
                label = 'DDoS'
                color = 'magenta'
            elif infection_type == 2:
                label = 'PortScan'
                color = 'orange'
            elif infection_type == 3:
                label = 'DDoS + PortScan'
                color = 'cyan'
            plt.plot([change, change], [0, 17], c=color, linestyle='--', label=label, linewidth=1.8)

    # plt.legend(loc='upper center', fontsize=8, ncol=6, bbox_to_anchor=(0, 0.15, 1, 1), fancybox=True)
    plt.legend(loc='center left', fontsize=12, ncol=1, bbox_to_anchor=(0, 0.3, -0.5, 1), fancybox=True)
    plt.ylim(0, 17)
    plt.xticks(fontsize=18, rotation=45)
    plt.yticks(fontsize=18)
    plt.xlabel('Number of instances', fontsize=23)
    plt.ylabel('Weight of MCs (log)', fontsize=23)
    name = dataset[:-4].replace('.', '_') + '_weights' + '.pdf'
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, name))
    plt.close()
