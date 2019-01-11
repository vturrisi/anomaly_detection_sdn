import numpy as np

from denstream import DenStream, gen_data_plot, plot_clusters


ds = DenStream(n_features=2, lambda_=0.1, beta=0.3,
               epsilon=0.1, mu=0.3, stream_speed=100, min_points=10)


X = np.random.random((10000, 2)) * 3
for i, x in enumerate(X):
    ds.train(x, 1)
    if i % 100 == 0:
        normal_points, outliers, c_clusters, \
            p_clusters, outlier_clusters = gen_data_plot(ds, X[:i])

        plot_clusters(i, normal_points, outliers,
                      c_clusters, p_clusters, outlier_clusters,
                      ds._epsilon)


# print(ds._p_micro_clusters)
# for mc in ds._p_micro_clusters:
#     print(mc.centroid)
#
# print(ds._o_micro_clusters)
# for mc in ds._o_micro_clusters:
#     print(mc.centroid)
#
# print(ds.generate_clusters())
