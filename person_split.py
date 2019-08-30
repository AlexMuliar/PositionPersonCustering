import pandas as pd
import numpy as np


class PersonClustering(object):
    """
        Class execute 3-step person clustering.

        First stage clustering persons by their position on camera by x-,y-axis and time
        using HDBSCAN and minimaze biggest cluster. 

        Second stage doing division cluster by identical frame_no in it. 
        Using sklearn.cluster.AgglomerativeClustering for division cluster.
        Division degree is max count identical frame_no.
        
        Third stage doing division cluster by biggest cluster intensive growth. 
        If cluster in one moment growth up on some %-value, than in this cluster is noise.
        For analysys use HDBSCAN and sklearn.cluster.AgglomerativeClustering for division cluster.
    """
    def __init__(self, position_split_distance_step=0.01, position_split_min_cluster_size=15, position_split_min_samples=15,
                 position_split_alpha=1.0, position_split_algorithm='best', position_split_cluster_selection_method='eom',
                 noise_filter_compress_n_components=5, noise_filter_min_cluster_size=10, noise_filter_min_samples=10,
                 min_cluster_size_for_cheking_noise=5, noise_filter_distance_step=0.01, max_difference_between_person_samples=0.5,
                 noise_index_max_person_in_cluster=5, time_stretching_degree=36):
        """
        Parametres:

        Fisrt stage: 
            - position_split_distance_step : float, optional (default=0.01)
                A step to analyze HDBSCAN.single_linkage_tree_ to minimize biggest cluster.
                more value increase alhorithm speed, but can dencrease accuracy.
                In both cases the result is not significant.

            - position_split_min_cluster_size : 
                HDBSCAN min_cluster_size : int, optional (default=15)
                The minimum number of samples in a group for that group to be
                considered a cluster; groupings smaller than this size will be left
                as noise.
            https://hdbscan.readthedocs.io/en/latest/parameter_selection.html#selecting-min-cluster-size

            - position_split_min_samples :
                HDBSCAN min_samples : int, optional (default=15)
                The number of samples in a neighborhood for a point
                to be considered as a core point. This includes the point itself.
                defaults to the min_cluster_size.
            https://hdbscan.readthedocs.io/en/latest/parameter_selection.html#selecting-min-samples

            - position_split_alpha : 
                HDBSCAN alpha : float, optional (default=1.0)
                A distance scaling parameter as used in robust single linkage.
            https://hdbscan.readthedocs.io/en/latest/parameter_selection.html#selecting-alpha

            - position_split_algorithm : 
                HDBSCAN algorithm : string, optional (default='best')
                Exactly which algorithm to use; hdbscan has variants specialised
                for different characteristics of the data. By default this is set
                to ``best`` which chooses the "best" algorithm given the nature of
                the data. You can force other options if you believe you know
                better. Options are:
                    * ``best``
                    * ``generic``
                    * ``prims_kdtree``
                    * ``prims_balltree``
                    * ``boruvka_kdtree``
                    * ``boruvka_balltree``

            position_split_cluster_selection_method :
                HDBSCAN cluster_selection_method : string, optional (default='eom')
                The method used to select clusters from the condensed tree. The
                standard approach for HDBSCAN* is to use an Excess of Mass algorithm
                to find the most persistent clusters. Alternatively you can instead
                select the clusters at the leaves of the tree -- this provides the
                most fine grained and homogeneous clusters. Options are:
                    * ``eom``
                    * ``leaf``
            https://hdbscan.readthedocs.io/en/latest/parameter_selection.html#leaf-clustering

        Second stage hasn't parametres.

        Third stage:
            noise_filter_compress_n_components : int, optional (default=5)
                Compress 512-size embedding to N-size by sklearn.decomposition.TruncatedSVD 
                for highlighting differences.
            
            noise_filter_min_cluster_size : int, optional (default=10)
                See position_split_min_cluster_size

            noise_filter_min_samples : int, optional (default=10)
                See position_split_min_samples

            noise_filter_distance_step : float, optional (default=0.01)
                A step to analyze HDBSCAN.single_linkage_tree_ to .
                more value increase alhorithm speed, but can dencrease accuracy.
                The result can be significant.

            min_cluster_size_for_cheking_noise : int, optional (default=5)
                If cluster size smaller than this value, alhorithm don't analysys it cluster.

            max_difference_between_person_samples : float, optional (default=0.5)
                The algorithm analyzes the last %N-part of the tree for the rapid growth of the largest cluster.

            noise_index_max_person_in_cluster : int, optional (default=5)
                Ignore person in cluster if it have lower than N samples.

        Data transform:
            parametres:
                time_stretching_degree : int, optional (default=36)
                    Is degree of stretching along timeline.

            required data format:
                pandas.DataFrame
                ------------------------------------------------------------------
                | object_id | x_center | y_center | frame_no |       dress       |
                ------------------------------------------------------------------
                |   int     |   int    |   int    |   int    | 1-D numpy.ndarray |   
                ------------------------------------------------------------------

        """

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
        "Method execute all clustering stages"
        self.df = self._data_transfrom(df)
        self._split_by_position()
        self._split_by_identical_frame_numbers_in_cluster()
        self._noise_filter() 
        #self._soft_merge() #Low accuracy now


    def transform(self, df):
        return self.df


    def fit_transform(self, df):
        self.fit(df)
        return self.transform(self.df)


    def _data_transfrom(self, df):
        "Tranform data for optimal clustering"
        #df['depth'] = 1 / (df.width * df.height)

        #   The degree of stretching that depends on the length of the video
        time_stretching_normilizer = (df.frame_no.max() - df.frame_no.min()) / 1200

        #   Normilize some parametres in (0,2) diapason 
        for axis in ['x_center', 'y_center','frame_no']: # additional demention can be 'depth'
            df[axis] = 1 + (df[axis] - df[axis].mean()) / (df[axis].max() - df[axis].min())
        
        #   Stretching the timeline by all degrees
        df['frame_no'] = df.frame_no * self.time_stretching_degree * time_stretching_normilizer
        return df


    def __get_raw_for_clustering_by_position(self):
        # Build 2-D list for HDBSCAN-clustering 
        return [[self.df.x_center[i], self.df.y_center[i], self.df.frame_no[i]] for i in self.df.index]
        
        
    def __search_optimal_distance(self, clf):
        "Return distance between samples, when max-cluster size is minimal"
        distance = 0
        plot = dict()
        while True:
            #   get labels of clustering by distance
            unique_cluster_id, counts_cluster_id = np.unique(\
                clf.single_linkage_tree_.get_clusters(distance), return_counts=True)
    
            if not -1 in unique_cluster_id: #   if isn't noise exit from cycle
                break
            counts_cluster_id = counts_cluster_id.tolist()
            
            nuber_noise_samples = counts_cluster_id[0]
            
            #   if clustering return more 1 cluster, add to noise max-size cluster
            if counts_cluster_id[1:]:
                nuber_noise_samples += counts_cluster_id[counts_cluster_id.index(max(counts_cluster_id[1:]))] 

            plot.update({distance: nuber_noise_samples})

            distance += self.position_split_distance_step

        key_list = list(plot.keys())
        val_list = list(plot.values())
        best_distance = key_list[val_list.index(min(val_list))]
        
        return best_distance


    def _split_by_position(self):
        from hdbscan import HDBSCAN

        #   Build HDBSCAN.single_linkage_tree_
        clf = HDBSCAN(min_cluster_size=self.position_split_min_cluster_size,
                min_samples=self.position_split_min_samples, alpha=self.position_split_alpha,
                cluster_selection_method=self.position_split_cluster_selection_method,
                algorithm=self.position_split_algorithm, allow_single_cluster=False)\
                    .fit(self.__get_raw_for_clustering_by_position())

        optimal_distance = self.__search_optimal_distance(clf)

        #   Get labels by optimal distance
        self.df['cluster'] = clf.single_linkage_tree_.get_clusters(
            optimal_distance, min_cluster_size=self.position_split_min_cluster_size)


    def _split_by_identical_frame_numbers_in_cluster(self):
        from sklearn.cluster import AgglomerativeClustering

        #   List of noise and clean cluster
        white_list = [-1]
        
        while True:
            labels = self.df.cluster.tolist()
            count_of_not_clear_clusters = 0
            clusters = self.df.cluster.unique()

            for cluster_id in clusters:
                #   check if cluster is clean
                if cluster_id in white_list: 
                    continue
                cluster = self.df[self.df.cluster == cluster_id] 

                #   maximum identical frame_no in cluster 
                degree_of_separation = cluster.groupby('frame_no').count()['object_id'].max()
                
                #   if all frame_no is unique, than cluster is clean
                if degree_of_separation == 1:
                    white_list.append(cluster_id)
                    continue
                else:
                    count_of_not_clear_clusters += 1

                #   Division cluster 
                agl_clf = AgglomerativeClustering(n_clusters=degree_of_separation)
                agl_clf.fit(np.array(cluster.dress.tolist()))
                labels = self.__update_labels_by_cluster_index(labels, agl_clf, cluster_id)

                self.df.cluster = labels
            if count_of_not_clear_clusters == 0:
                break


    def _noise_filter(self):
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.decomposition import TruncatedSVD
        from hdbscan import HDBSCAN

        #   List of noise and clean cluster
        white_list = [-1]

        while True:
            is_noise = False
            labels = self.df.cluster.tolist()

            for cluster_id in self.df.cluster.unique():
                distance = 0
                plot = dict()
                if cluster_id in white_list:
                    continue
                cluster = self.df[self.df.cluster == cluster_id]

                #   if cluster is small, continue
                if cluster.shape[0] < self.min_cluster_size_for_cheking_noise:
                    continue
                
                #   Compress data for extract differences
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

                    #   Growth analysis of a biggest cluster
                    plot.update({distance: (counts_cluster_id.max() - prev)})
                    prev = counts_cluster_id.max()

                val_list = list(plot.values())

                #   Select %-part of cluster for analys
                half_val_list = np.array(val_list[int(len(val_list) * self.max_difference_between_person_samples): ])

                #   If cluster is too small, avaible size of noise is 3
                if self.noise_filter_min_cluster_size > self.df.shape[0] *  self.noise_index_max_person_in_cluster:
                    max_noise = 3
                else:
                    max_noise = cluster.shape[0] / self.noise_index_max_person_in_cluster
                separate_degree = half_val_list[half_val_list > max_noise].shape[0]
                
                #   If detected rapid rise cluster size
                if separate_degree:
                    is_noise = True
                    agl_clf = AgglomerativeClustering(n_clusters=separate_degree + 1)
                    agl_clf.fit(np.array(cluster.dress.tolist()))
                    labels = self.__update_labels_by_cluster_index(labels, agl_clf, cluster_id)
                else:
                    #   If clean, add to white_list
                    white_list.append(cluster_id)
            self.df.cluster = labels
            if not is_noise:
                break


    def __update_labels_by_cluster_index(self, labels, clf, cluster_index):
        max_index = max(labels)
        #mask = (item for item in clf.labels_)
        index = 0
        for label_index in range(len(labels)):
            if labels[label_index] == cluster_index:
                new_label = clf.labels_[index]
                index = index + 1
                #new_label = next(mask)
                if new_label:
                    labels[label_index] = labels[label_index] + new_label + max_index
        return labels


    def _soft_merge(self):
        from hdbscan import HDBSCAN

        embedings = list()
        for cluster in self.df.cluster.unique():
            cluster_dress = self.df[self.df.cluster == cluster].dress.values
            embedings.append(np.mean(cluster_dress, axis=0))

        clf = HDBSCAN(min_cluster_size=2).fit(embedings)
        labels = list()
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
        