import numpy as np
import pickle
import networkx as nx
import community as co
from collections import OrderedDict

class ActivationNetwork():
    def __init__(self, num_classes=10, num_students=2, top_k_creation=50, percentile_value=99, c_final_conv=256):
        self.interm_out_gr = None
        self.preds = None
        self.labels = None
        self.filter_net_thres = None
        self.filter_network = None
        self.num_students = num_students
        self.percentile_value = percentile_value
        self.num_classes = num_classes
        self.top_k_creation = top_k_creation  # Number of top filters to look at in terms of activation
        self.avg_actv = True
        self.c_final_conv = c_final_conv
        self.filter_net = np.zeros([self.c_final_conv, self.c_final_conv])  # Filter to filter network
        self.f2p_net = np.zeros([self.c_final_conv, num_classes])  # Filter to prediction network
        self.f2c_net = np.zeros([self.c_final_conv, num_classes])  # Filter to class network
        # Filter to prediction network without self.top_k_creation
        self.f2p_net_g2All = np.zeros([self.c_final_conv, num_classes])
        # Filter to class network without self.top_k_creation
        self.f2c_net_g2All = np.zeros([self.c_final_conv, num_classes])
        self.comm_size_thresh = 1

    def partition_activation_network(self, filename, filename_cluster):
        self.load_activation_network(filename)
        self.filter_net /= self.filter_net.max()
        f2fAllLinks = np.asarray(self.filter_net).reshape(-1)
        f2fnonzero = f2fAllLinks[f2fAllLinks != 0]
        percentile_value = self.percentile_value
        self.filter_net_thres = np.percentile(f2fnonzero, percentile_value)
        origF2F = np.zeros_like(self.filter_net)
        # This way, the values are copied into a new variable and the memory isn't shared
        origF2F[:] = self.filter_net[:]
        self.filter_net[self.filter_net < self.filter_net_thres] = 0
        self.filter_network = nx.from_numpy_matrix(self.filter_net)  # This is an undirected graph
        # Put communities into a dictionary: {comm1 :: nodeIDs :: number_of_nodes}
        gamma = 0.7  # Gives 4 clusters total for 0.086 filter net threshold (largest community has 181 filters)
        filter2comm = co.best_partition(self.filter_network, resolution=gamma)
        comm2filter = {}
        comm2num_filters = {}
        for k, v in filter2comm.items():
            comm2filter[v] = comm2filter.get(v, [])
            comm2filter[v].append(k)
            comm2num_filters[v] = len(comm2filter[v])
        commNumList = list(comm2num_filters.values())
        uniqueClusters = np.unique(commNumList)
        filterClusters = {}
        filterClusterSizes = {}
        k = 0
        for i in uniqueClusters:
            if i <= self.comm_size_thresh:
                filtersToCluster = []
                i1 = self.indices(commNumList, lambda x: x <= self.comm_size_thresh)
                for j in i1:
                    filtersToCluster.extend(comm2filter[j])
                filterClusters.update({k: filtersToCluster})
                filterClusterSizes.update({k: len(filtersToCluster)})
            else:
                filtersToCluster = []
                i2 = self.indices(commNumList, lambda x: x == i)
                for j in i2:
                    filtersToCluster.extend(comm2filter[j])
                filterClusters.update({k: filtersToCluster})
                filterClusterSizes.update({k: len(filtersToCluster)})
            k += 1
        # print(f"k = {k}")

        if k-1 < self.num_students:
            print(f"There are not enough partitions to make {self.num_students} students")
            print(f"Max number of students will be {k-1}")
            self.num_students = k-1

        # Initialize the filter dictionaries
        filterClustersFinal = {}
        filterClusterSizesFinal = {}
        for i in range(self.num_students):
            filterClustersFinal[i] = []
            filterClusterSizesFinal[i] = 0

        # Sorting the dict of filters in descending order
        aux = {}
        x = 0
        for i in sorted(filterClusters, key=lambda y: len(filterClusters[y]), reverse=True):
            aux[x] = filterClusters[i]
            x += 1
        filterClusters = aux

        # for i in range(k):
        #     print(f"{i}: {len(filterClusters[i])}")

        # Always ignore the first partition, P0
        cnt = 1

        # Fill in each partition inside a student
        for i in range(self.num_students):
            filterClustersFinal[i] += filterClusters[cnt]
            filterClusterSizesFinal[i] += len(filterClusters[cnt])
            cnt += 1

        # If any other partitions are left, put them in the student with minimum filters
        for i in range(k-cnt):
            idx = min(filterClusterSizesFinal, key=filterClusterSizesFinal.get)
            filterClustersFinal[idx] += filterClusters[cnt + i]
            filterClusterSizesFinal[idx] += len(filterClusters[cnt + i])

        filterClusterSizes = filterClusterSizesFinal
        filterClusters = filterClustersFinal

        print(filterClusterSizes)

        with open(filename_cluster, 'wb') as f:
            print(f"Writing filters to {filename_cluster}")
            pickle.dump([filterClusters, filterClusterSizes], f)

    def create_activation_network(self, batchSize=128):
        g2_out = self.interm_out_gr.data.cpu().numpy()
        yPred = self.preds.data.cpu().numpy()
        final_conv_layer = g2_out
        true_labels = np.asarray(self.labels.data.cpu())
        model_pred = yPred

        for img_idx in range(batchSize):  # range in python 3.x is xrange in python 2.x
            i = 0
            filter_activity = np.zeros(self.c_final_conv)
            for channel_no in range(self.c_final_conv):
                if self.avg_actv:
                    filter_activity[i] = np.matrix.mean(np.asmatrix(final_conv_layer[img_idx, channel_no, :, :]))
                else:
                    filter_activity[i] = np.matrix.max(np.asmatrix(final_conv_layer[img_idx, channel_no, :, :]))
                i += 1

            filter_activity[filter_activity < 0] = 0

            filter_activity /= filter_activity.max()

            sort_idx_desc = np.argsort(filter_activity)[::-1]
            filter_activity_sorted = filter_activity[sort_idx_desc]
            self.top_k_filter_actv = filter_activity_sorted[0:self.top_k_creation]
            self.top_k_filter_idx = sort_idx_desc[0:self.top_k_creation]

            for j in range(self.top_k_creation):
                for k in range(j + 1, self.top_k_creation):
                    # self.filter_net[self.top_k_filter_idx[j],self.top_k_filter_idx[k]] += ( (self.top_k_filter_actv[j]*self.top_k_filter_actv[k]) / (np.abs(self.top_k_filter_actv[j] - self.top_k_filter_actv[k]) + 1) )
                    # Non-coactivation addition:
                    self.filter_net[self.top_k_filter_idx[j], self.top_k_filter_idx[k]] += \
                        (self.top_k_filter_actv[j] * self.top_k_filter_actv[k] *
                         np.abs(self.top_k_filter_actv[j] - self.top_k_filter_actv[k]))
                    self.filter_net[self.top_k_filter_idx[k], self.top_k_filter_idx[j]] = \
                        self.filter_net[self.top_k_filter_idx[j], self.top_k_filter_idx[k]]

            self.f2p_net[self.top_k_filter_idx, model_pred[img_idx]] += self.top_k_filter_actv
            self.f2c_net[self.top_k_filter_idx, true_labels[img_idx]] += self.top_k_filter_actv

            self.f2p_net_g2All[:, model_pred[img_idx]] += filter_activity
            self.f2c_net_g2All[:, true_labels[img_idx]] += filter_activity

    def relu(self, x):
        x[x < 0] = 0
        return x

    def indices(self, a, func):
        return [i for (i, val) in enumerate(a) if func(val)]

    def load_activation_network(self, filename):
        with open(filename, 'rb') as f:
            self.filter_net, self.f2p_net, self.f2c_net, self.f2p_net_gAll, self.f2c_net_gAll = pickle.load(f)

    def dump_activation_network(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump([self.filter_net, self.f2p_net, self.f2c_net, self.f2p_net_g2All, self.f2c_net_g2All], f)
