import os
import pickle
from contextlib import suppress

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


datasets_folder = 'datasets'
dump_folder = 'dump_logs'
dump_files = os.listdir(dump_folder)
plot_folder = 'plots'

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

    plt.figure(figsize=(8, 4))

    x = list(data.keys())
    n_c_clusters = []
    n_p_clusters = []
    n_o_clusters = []
    for instance, partial in data.items():
        n_c_clusters.append(partial['n_c_clusters'])
        n_p_clusters.append(partial['n_p_clusters'])
        n_o_clusters.append(partial['n_o_clusters'])

    plt.plot(x, n_c_clusters, c='blue', label='C-MC', alpha=0.4, linewidth=1.8)
    plt.plot(x, n_p_clusters, c='green', label='P-MC', linewidth=1.8)
    plt.plot(x, n_o_clusters, c='red', label='O-MC', alpha=0.4, linewidth=1.8)
    plotted_malicious_to_normal = False
    plotted_normal_to_malicious = False

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

    plt.legend(loc='upper left', fontsize=12, ncol=2, bbox_to_anchor=(0, 0, 1, 1), fancybox=True)
    plt.ylim(0, 10)
    plt.xticks(fontsize=16, rotation=45)
    plt.yticks(fontsize=16)
    plt.xlabel('Number of instances', fontsize=20)
    plt.ylabel('Number of MCs', fontsize=20)
    name = dataset[:-4].replace('.', '_') + '.pdf'
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, name))
    plt.close()
