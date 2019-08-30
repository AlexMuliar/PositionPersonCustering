import warnings
warnings.filterwarnings("ignore")

from hdbscan import HDBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD

import pandas as pd
import numpy as np

import sys



class PersonClustering(object):
    def __init__(self, position_split_distance_step=0.01, position_split_min_cluster_size=15, position_split_min_samples=15,
                 position_split_alpha=1, position_split_algorithm='best', position_split_cluster_selection_method='eom',
                 noise_filter_compress_n_components=5, noise_filter_min_cluster_size=10, noise_filter_min_samples=10,
                 min_cluster_size_for_cheking_noise=5, noise_filter_distance_step=0.01, max_difference_between_person_samples=0.5,
                 noise_index_max_person_in_cluster=5, time_stretching_degree=36):

        self.position_split_distance_step = position_split_distance_step
        self.position_split_min_cluster_size = position_split_min_cluster_size
        self.position_split_min_samples = position_split_min_samples
        self.position_split_alpha = position_split_alpha
        self.position_split_algorithm = position_split_algorithm
        self.position_split_cluster_selection_method = position_split_cluster_selection_method
        self.noise_filter_compress_n_components = noise_filter_compress_n_components
        self.noise_filter_min_cluster_size = noise_filter_min_cluster_size
        self.noise_filter_min_samples = noise_filter_min_samples
        self.noise_filter_distance_step = noise_filter_distance_step
        self.min_cluster_size_for_cheking_noise = min_cluster_size_for_cheking_noise
        self.max_difference_between_person_samples = max_difference_between_person_samples
        self.noise_index_max_person_in_cluster = noise_index_max_person_in_cluster
        self.time_stretching_degree = time_stretching_degree


    def fit(self, df):
        self.df = self._data_transfrom(df)
        print("Position...")
        self._split_by_position()
        print("Frames...")
        self._split_by_identical_frame_numbers_in_cluster()
        print("Noise...")
        self._noise_filter()
        print("Mean embedings...")
        #self._soft_merge()


    def transform(self, df):
        return self.df


    def fit_transform(self, df):
        self.fit(df)
        return self.transform(self.df)


    def _data_transfrom(self, df):
        df['depth'] = 1 / (df.width * df.height)
        time_stretching_normilizer = (df.frame_no.max() - df.frame_no.min()) / 1200
        for axis in ['x_center', 'y_center','frame_no', 'depth']:
            df[axis] = 1 + (df[axis] - df[axis].mean()) / (df[axis].max() - df[axis].min())
        
        df['frame_no'] = df.frame_no * self.time_stretching_degree * time_stretching_normilizer
        return df


    def __get_raw_for_clustering_by_position(self):
        return [[self.df.x_center[i], self.df.y_center[i], self.df.frame_no[i]] for i in self.df.index]
        
        
    def __search_optimal_distance(self, clf):
        distance = 0
        plot = dict()
        while True:

            unique_cluster_id, counts_cluster_id = np.unique(\
                clf.single_linkage_tree_.get_clusters(distance), return_counts=True)
    
            if not -1 in unique_cluster_id:
                break
            counts_cluster_id = counts_cluster_id.tolist()
            
            nuber_noise_samples = counts_cluster_id[0]
            if counts_cluster_id[1:]:
                nuber_noise_samples += counts_cluster_id[counts_cluster_id.index(max(counts_cluster_id[1:]))]

            plot.update({distance: nuber_noise_samples})

            distance += self.position_split_distance_step

        key_list = list(plot.keys())
        val_list = list(plot.values())
        best_distance = key_list[val_list.index(min(val_list))]
        
        return best_distance


    def _split_by_position(self):
        data = np.array(self.__get_raw_for_clustering_by_position())
        #print(data.shape)
        clf = HDBSCAN(min_cluster_size=self.position_split_min_cluster_size,
                min_samples=self.position_split_min_samples, alpha=self.position_split_alpha,
                cluster_selection_method=self.position_split_cluster_selection_method,
                algorithm=self.position_split_algorithm, allow_single_cluster=False)\
                    .fit(data)

        optimal_distance = self.__search_optimal_distance(clf)
        self.df['cluster'] = clf.single_linkage_tree_.get_clusters(
            optimal_distance, min_cluster_size=self.position_split_min_cluster_size)


    def _split_by_identical_frame_numbers_in_cluster(self):
        white_list = [-1]
        while True:
            labels = self.df.cluster.tolist()
            count_of_not_clear_clusters = 0
            clusters = self.df.cluster.unique()
            for cluster_id in clusters:
                if cluster_id in white_list:
                    continue
                cluster = self.df[self.df.cluster == cluster_id] 
                degree_of_separation = cluster.groupby('frame_no').count()['object_id'].max()
                
                if degree_of_separation == 1:
                    white_list.append(cluster_id)
                    continue
                else:
                    count_of_not_clear_clusters += 1

                agl_clf = AgglomerativeClustering(n_clusters=degree_of_separation)
                #print(np.array(cluster.dress.tolist()).reshape(-1, 512).shape)
                agl_clf.fit(np.array(cluster.dress.tolist()))
                #print(labels.count(cluster_id), self.df.cluster.tolist().count(cluster_id), len(agl_clf.labels_))
                labels = self.__update_labels_by_cluster_index(labels, agl_clf, cluster_id)

                self.df.cluster = labels
            if count_of_not_clear_clusters == 0:
                break


    def _noise_filter(self):
        white_list = [-1]
        #raise(IOError)
        while True:
            is_noise = False
            labels = self.df.cluster.tolist()

            for cluster_id in self.df.cluster.unique():
                distance = 0
                plot = dict()
                if cluster_id in white_list:
                    continue
                cluster = self.df[self.df.cluster == cluster_id]

                if cluster.shape[0] < self.min_cluster_size_for_cheking_noise:
                    continue
                
                #data_raw = np.array(cluster.dress.tolist())
                data_raw = TruncatedSVD(n_components=self.noise_filter_compress_n_components)\
                    .fit_transform(np.array(cluster.dress.tolist()))

                clf = HDBSCAN(min_cluster_size=self.noise_filter_min_cluster_size, 
                            min_samples=self.noise_filter_min_samples).fit(data_raw)


                prev = 0
                while True:
                    distance += self.noise_filter_distance_step

                    mask = clf.single_linkage_tree_.get_clusters(distance, min_cluster_size=self.noise_filter_min_cluster_size)
                    if not mask[mask == -1].shape[0] or distance > 100:
                        break
                    mask = mask[mask != -1]
                    _ , counts_cluster_id = np.unique(mask, return_counts=True)
                    if not counts_cluster_id.shape[0]:
                        continue
                    plot.update({distance: (counts_cluster_id.max() - prev)})
                    prev = counts_cluster_id.max()

                val_list = list(plot.values())
                half_val_list = np.array(val_list[int(len(val_list) * self.max_difference_between_person_samples): ])

                if self.noise_filter_min_cluster_size > self.df.shape[0] *  self.noise_index_max_person_in_cluster:
                    max_noise = 3
                else:
                    max_noise = cluster.shape[0] / self.noise_index_max_person_in_cluster
                separate_degree = half_val_list[half_val_list > max_noise].shape[0]
                
          
                if separate_degree:
                    is_noise = True
                    agl_clf = AgglomerativeClustering(n_clusters=separate_degree + 1)
                    agl_clf.fit(np.array(cluster.dress.tolist()))
                    labels = self.__update_labels_by_cluster_index(labels, agl_clf, cluster_id)
                else:
                    white_list.append(cluster_id)
            self.df.cluster = labels
            if not is_noise:
                break


    def __update_labels_by_cluster_index(self, labels, clf, cluster_index):
        
        #print(labels.count(cluster_index), len(clf.labels_))
        max_index = max(labels)
        mask = (item for item in clf.labels_)
        c = 0
        for label_index in range(len(labels)):
            if labels[label_index] == cluster_index:
                c += 1
                try:
                    new_label = next(mask)
                except:
                    raise(StopIteration)
                    #print(c)
                if new_label:
                    labels[label_index] += new_label + max_index

       # print()
        return labels

    def _soft_merge(self):
        embedings = list()
        print(self.df.shape, self.df.cluster.unique().shape)
        for cluster in self.df.cluster.unique():
            cluster_dress = self.df[self.df.cluster == cluster].dress.values
            embedings.append(np.mean(cluster_dress, axis=0))

        clf = HDBSCAN(min_cluster_size=2).fit(embedings)
        labels = list()
        print(clf)
        for i, j in zip(self.df.groupby('cluster').size(), clf.labels_):
            labels.extend([j] * i)
        l1, l2 = self.df.cluster.tolist(), labels
        d2 = dict(zip(clf.labels_, self.df.cluster.unique()))
        for i in range(self.df.shape[0]):
            if l2[i] == -1:
                l2[i] = l1[i]
            else:
                l2[i] = d2[l2[i]]
        self.df.cluster = l2
        print(self.df.shape, self.df.cluster.unique().shape)
        