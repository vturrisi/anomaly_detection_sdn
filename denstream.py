import math
import shutil
import subprocess
from collections import Counter, deque, namedtuple
from contextlib import suppress
from io import StringIO
from math import log2
from multiprocessing import Pool

import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

AlphaPoint = namedtuple('AlphaPoint', ['i', 'alpha', 'X', 'y'])


class NoMicroClusterException(RuntimeError):
    pass


class PointNotAddedException(RuntimeError):
    pass


class DenStream:
    def __init__(self, n_features, lambda_, beta, epsilon, mu,
                 stream_speed, min_points):
        self._n_features = n_features
        self._lambda = lambda_
        self._beta = beta
        self._epsilon = epsilon
        self._mu = mu
        self._stream_speed = stream_speed
        self._min_points = min_points

        self._p_micro_clusters = []
        self._o_micro_clusters = []
        self._mc_id = 0
        self._time = 0
        self._no_processed_points = 0
        self._initiated = False
        self._initiate_buffer = []

    @staticmethod
    def euclidean_distance(X1, X2):
        """
        Compute the Euclidean Distance between two points
        """
        return np.sqrt(np.sum(np.power(X1 - X2, 2)))

    def find_closest_p_mc(self, X):
        """
        Find the closest p_micro_cluster to the point "point"
        according to the Euclidean Distance
        between it and the cluster's centroid
        """
        if len(self._p_micro_clusters) == 0:
            raise NoMicroClusterException

        distances = [(i, self.euclidean_distance(X, cluster.centroid))
                     for i, cluster in enumerate(self._p_micro_clusters)]
        i, dist = min(distances, key=lambda i_dist: i_dist[1])
        p_micro = self._p_micro_clusters[i]
        return i, p_micro, dist

    def find_closest_o_mc(self, X):
        """
        Find the closest o_micro_cluster to the point "point"
        according to the Euclidean Distance
        between it and the cluster's centroid
        """
        if len(self._o_micro_clusters) == 0:
            raise NoMicroClusterException

        distances = [(i, self.euclidean_distance(X, cluster.centroid))
                     for i, cluster in enumerate(self._o_micro_clusters)]
        i, dist = min(distances, key=lambda i_dist: i_dist[1])
        o_micro = self._o_micro_clusters[i]
        return i, o_micro, dist

    def add_point(self, X, y):
        """
        Try to add a point "point" to the existing p_micro_clusters at time "t"
        Otherwise, try to add that point to the existing o_micro_clusters
        If all fails, create a new o_micro_cluster with that new point
        """
        try:
            i, closest_p_mc, _ = self.find_closest_p_mc(X)
            if closest_p_mc.radius_with_new_point(X) <= self._epsilon:
                closest_p_mc.update(X, y, self._time)
            else:
                raise PointNotAddedException

        except (NoMicroClusterException, PointNotAddedException):
            # Try to merge point with closest p_mc
            try:
                i, closest_o_mc, _ = self.find_closest_o_mc(X)
                # Try to merge point with closest o_mc
                if closest_o_mc.radius_with_new_point(X) <= self._epsilon:
                    closest_o_mc.update(X, y, self._time)
                    # Try to promote o_micro_clusters to p_mc
                    if closest_o_mc._weight > self._beta * self._mu:
                        self._o_micro_clusters.pop(i)
                        self._p_micro_clusters.append(closest_o_mc)
                else:
                    raise PointNotAddedException

            except (NoMicroClusterException, PointNotAddedException):
                # create new o_mc containing the new point
                new_o_mc = self.MicroCluster(id_=self._mc_id,
                                             n_features=self._n_features,
                                             time=self._time,
                                             lambda_=self._lambda)
                self._mc_id += 1
                new_o_mc.update(X, y, self._time)
                self._o_micro_clusters.append(new_o_mc)

    def train(self, X, y):
        """
        "Train" Denstream by updating its p_micro_clusters and o_micro_clusters
        with a new point "point"
        """

        self._no_processed_points += 1
        # update time every stream_speed instances
        if self._no_processed_points % self._stream_speed == 0:
            self._time += 1
            # try to decay all p_micro_clusters
            for mc in self._p_micro_clusters:
                mc.decay(cur_time=self._time)

            # try to decay all o_micro_clusters
            for mc in self._o_micro_clusters:
                mc.decay(cur_time=self._time)

        if not self._initiated:
            self._initiate_buffer.append((X, y))
            if len(self._initiate_buffer) == self._min_points:
                self.DBSCAN(self._initiate_buffer)
        else:
            t = self._time
            # Compute Tp
            try:
                part = (self._beta * self._mu) / (self._beta * self._mu - 1)
                Tp = math.ceil(1 / self._lambda * math.log2(part))
            except:
                Tp = 1

            self.add_point(X, y)

            # Test if should remove any p_micro_cluster or o_micro_cluster
            if t % Tp == 0:
                for i, cluster in enumerate(self._p_micro_clusters):
                    if cluster._weight < self._beta * self._mu:
                        self._p_micro_clusters.pop(i)

                for i, cluster in enumerate(self._o_micro_clusters):
                    to = cluster._time
                    e = ((math.pow(2, - self._lambda * (t - to + Tp)) - 1) /
                         (math.pow(2, - self._lambda * Tp) - 1))
                    if cluster._weight < e:
                        self._o_micro_clusters.pop(i)

    def is_normal(self, X):
        """
        Find if point "X" is inside any p_micro_cluster
        """
        for mc in self._p_micro_clusters:
            # if mc.radius_with_new_point(X) <= self._epsilon:
            #     return True
            dist = self.euclidean_distance(X, mc.centroid)
            if dist <= mc.radius:  # dist <= self._epsilon
                return True

        return False

    def DBSCAN(self, buffer):
        """
        Perform DBSCAN to create initial p_micro_clusters
        Works by grouping points with distance <= self._epsilon
        and filtering groups that are not dense enough (n_points >= beta * mu)
        """
        connected_points = []
        # create a queue containing all p_micro_clusters dense enough
        remaining_points = deque(buffer)

        testing_group = -1
        # try to add the remaining clusters to existing groups
        while remaining_points:
            # create a new group
            connected_points.append([remaining_points.popleft()])
            testing_group += 1
            change = True
            while change:
                change = False
                buffer_ = deque()
                # try to add remaining points to the existing group
                # if we add a new cluster to that group,
                # perform the check again
                while remaining_points:
                    remain_X, remain_y = remaining_points.popleft()
                    to_add = False
                    for (X, y) in connected_points[testing_group]:
                        dist = self.euclidean_distance(X, remain_X)
                        if dist <= self._epsilon:
                            to_add = True
                            break
                    if to_add:
                        connected_points[testing_group].append((remain_X, remain_y))
                        change = True
                    else:
                        buffer_.append((X, y))
                remaining_points = buffer_

        # Filter groups not dense enough and create a p_micro_cluster for each
        for group in connected_points:
            weight = len(group)
            if weight >= self._beta * self._mu:
                new_p_mc = self.MicroCluster(id_=self._mc_id,
                                             n_features=self._n_features,
                                             time=0,
                                             lambda_=self._lambda)
                self._mc_id += 1
                for X, y in group:
                    new_p_mc.update(X, y, self._time)
                self._p_micro_clusters.append(new_p_mc)

        self._initiated = True
        del self._initiate_buffer

    def generate_clusters(self):
        """
        Perform DBSCAN to create the final c_micro_clusters
        Works by grouping dense enough p_micro_clusters (weight >= mu)
        with distance <= 2 * self._epsilon
        """
        if len(self._p_micro_clusters) > 1:
            connected_clusters = []
            # create a queue containing all p_micro_clusters dense enough
            # dense_clusters = filter(lambda mc: mc._weight >= self._mu,
            #                         self._p_micro_clusters)
            remaining_clusters = deque((self.Cluster(id_=mc._id,
                                                     centroid=mc.centroid,
                                                     radius=mc.radius,
                                                     weight=mc._weight)
                                        for mc in self._p_micro_clusters))

            testing_group = -1
            # try to add the remaining clusters to existing groups
            while remaining_clusters:
                # create a new group
                connected_clusters.append([remaining_clusters.popleft()])
                testing_group += 1
                change = True
                while change:
                    change = False
                    buffer_ = deque()
                    # try to add remaining clusters to the existing group as it is
                    # if we add a new cluster to that group, perform the check again
                    while remaining_clusters:
                        r_cluster = remaining_clusters.popleft()
                        to_add = False
                        for cluster in connected_clusters[testing_group]:
                            dist = self.euclidean_distance(cluster.centroid,
                                                           r_cluster.centroid)
                            # if dist <= cluster.radius + r_cluster.radius:
                            if dist <= 2 * self._epsilon:
                                to_add = True
                                break
                        if to_add:
                            connected_clusters[testing_group].append(r_cluster)
                            change = True
                        else:
                            buffer_.append(r_cluster)
                    remaining_clusters = buffer_

            dense_groups, sparse_groups = [], []
            for group in connected_clusters:
                if sum([c.weight for c in group]) >= self._mu:
                    dense_groups.append(group)
                else:
                    sparse_groups.append(group)

            if len(dense_groups) == 0:
                dense_groups = [[]]

            if len(sparse_groups) == 0:
                sparse_groups = [[]]

            return dense_groups, sparse_groups

        # only one p_micro_cluster (check if it is dense enough)
        elif len(self._p_micro_clusters) == 1:
            id_ = self._p_micro_clusters[0]._id
            centroid = self._p_micro_clusters[0].centroid
            radius = self._p_micro_clusters[0].radius
            weight = self._p_micro_clusters[0]._weight
            if weight >= self._mu:
                return [[self.Cluster(id_=id_,
                                      centroid=centroid,
                                      radius=radius,
                                      weight=weight)]], [[]]
            else:
                return [[]], [[self.Cluster(id_=id_,
                                            centroid=centroid,
                                            radius=radius,
                                            weight=weight)]]
        return [[]], [[]]

    def generate_p_clusters(self):
        return [self.Cluster(id_=mc._id,
                             centroid=mc.centroid,
                             radius=mc.radius,
                             weight=mc._weight)
                for mc in self._p_micro_clusters]

    def generate_outlier_clusters(self):
        return [self.Cluster(id_=mc._id,
                             centroid=mc.centroid,
                             radius=mc.radius,
                             weight=mc._weight)
                for mc in self._o_micro_clusters]

    class MicroCluster:
        def __init__(self, id_, n_features, time, lambda_):
            self._id = id_
            self._time = time
            self._lambda = lambda_

            self._CF = np.zeros(n_features)
            self._CF2 = np.zeros(n_features)
            self._weight = 0
            self._Y = []

        def __repr__(self):
            args = {'id': self._id, 'centroid': self.centroid,
                    'radius': self.radius, 'class_dist': self.class_dist}

            rpr = ('MicroCluster(id={id},'
                   ' centroid={centroid},'
                   ' radius={radius},'
                   ' class_dist={class_dist})'.format(**args))
            return rpr

        @property
        def centroid(self):
            return self._CF / self._weight

        @property
        def radius(self):
            CF1_squared = (self._CF / self._weight) ** 2
            return np.nan_to_num(np.nanmax(((self._CF2 / self._weight)
                                             - CF1_squared) ** (1 / 2)))

        @property
        def class_dist(self):
            return Counter(self._Y)

        def radius_with_new_point(self, X):
            CF1 = self._CF + X
            CF2 = self._CF2 + X * X
            weight = self._weight + 1
            CF1_squared = (CF1 / weight) ** 2
            return np.nanmax(((CF2 / weight) - CF1_squared) ** (1 / 2))

        def update(self, X, y, time):
            self._CF += X
            self._CF2 += X * X
            self._weight += 1
            self._time = time
            self._Y.append(y)

        def decay(self, cur_time):
            if cur_time > self._time + 1:
                factor = 2 ** (-self._lambda)
                self._CF *= factor
                self._CF2 *= factor
                self._weight *= factor

    class Cluster:
        def __init__(self, id_, centroid, radius, weight):
            self.id = id_
            self.centroid = centroid
            self.radius = radius
            self.weight = weight

        def __str__(self):
            return f'Centroid: {self.centroid} | Radius: {self.radius}'


# auxiliary functions

def gen_data_plot(denstream, points, alpha_range=(0, 1.0)):
    max_points = 1000
    alphas = np.linspace(*alpha_range, num=max_points)
    if len(points) > max_points:
        alphas = ([0.1] * (len(points) - max_points)) + list(alphas)
    normal_points, outliers = [], []
    for i, (alpha, (X, y)) in enumerate(zip(alphas[-len(points):],
                                            points)):
        point = AlphaPoint(i=i, alpha=alpha, X=X, y=y)
        if denstream.is_normal(X):
            normal_points.append(point)
        else:
            outliers.append(point)

    c_clusters, p_clusters = denstream.generate_clusters()
    # p_clusters = denstream.generate_p_clusters()
    outlier_clusters = denstream.generate_outlier_clusters()
    return normal_points, outliers, c_clusters, p_clusters, outlier_clusters


def save_clusters_info(instances_id, denstream, fname):
    c_clusters, p_clusters = denstream.generate_clusters()
    # p_clusters = denstream.generate_p_clusters()
    outlier_clusters = denstream.generate_outlier_clusters()
    with open(fname, 'a') as f:
        f.write(f'instancia {instances_id};')
        for group in c_clusters:
            for i, c in enumerate(group):
                f.write(f'c_cluster(group{i});{c.id};{c.centroid};{c.radius};{c.weight};')

        for group in p_clusters:
            for i, c in enumerate(group):
                f.write(f'p_cluster(group{i});{c.id};{c.centroid};{c.radius};{c.weight};')

        for c in outlier_clusters:
            f.write(f'o_cluster;{c.id};{c.centroid};{c.radius};{c.weight};')

        f.write('\n')


def plot_clusters(fname, normal_points, outliers,
                  c_clusters, p_clusters, outlier_clusters,
                  epsilon,
                  title=None, xfeature_name=None, yfeature_name=None):

    plt.figure(figsize=(9, 9), dpi=300)
    ax = plt.subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if title is None:
        plt.title(fname)
    else:
        plt.title(title)

    if p_clusters[0]:
        for i, group in enumerate(p_clusters):
            for cluster in group:
                if cluster.radius < 0.05:
                    cluster.radius = 0.05
                ax.add_patch(patches.Circle(cluster.centroid,
                             cluster.radius, fill=False,
                             color='blue', ls=':'))
                ax.add_patch(patches.Circle(cluster.centroid,
                             epsilon, fill=False,
                             color='blue', ls='--'))
                ax.annotate(f'{i}', xy=cluster.centroid, color='red')
                # ax.annotate(f'id:{cluster.id}', xy=cluster.centroid, color='red')

    if outlier_clusters:
        for cluster in outlier_clusters:
            if cluster.radius < 0.05:
                cluster.radius = 0.05
            ax.add_patch(patches.Circle(cluster.centroid,
                         cluster.radius, fill=False, color='red', ls='--'))
            ax.add_patch(patches.Circle(cluster.centroid,
                         epsilon, fill=False, color='red', ls='--'))

    if c_clusters[0]:
        for i, group in enumerate(c_clusters):
            for cluster in group:
                if cluster.radius < 0.05:
                    cluster.radius = 0.05
                ax.add_patch(patches.Circle(cluster.centroid,
                             cluster.radius, fill=False, color='black'))
                ax.add_patch(patches.Circle(cluster.centroid,
                             epsilon, fill=False, color='black', ls='--'))
                ax.annotate(f'{i}', xy=cluster.centroid, color='purple', size=12)
                # ax.annotate(f'id:{cluster.id}', xy=cluster.centroid, color='red')

    for p in normal_points:
        if p.y == 0:
            color = 'green'
        else:
            color = 'red'
        ax.scatter(*p.X, alpha=p.alpha, color=color, marker='o', s=11)
        # ax.annotate(p.y, xy=p.X, color=color, size=7)

    for p in outliers:
        if p.y == 0:
            color = 'green'
        else:
            color = 'red'
        ax.scatter(*p.X, alpha=p.alpha, color=color, marker='x', s=11)
        # ax.annotate(p.y, xy=p.X, color=color, size=7)

    if xfeature_name is not None:
        plt.xlabel(xfeature_name)

    if yfeature_name is not None:
        plt.ylabel(yfeature_name)

    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
