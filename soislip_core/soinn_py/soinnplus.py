import numpy as np
import sys #TODO remove if not needed anymore
from .color_utils import ColorUtils as cu
from scipy.sparse import csr_matrix, lil_matrix
from scipy.stats import iqr, median_abs_deviation
import matplotlib.pyplot as plt

#TODO check if all functions are functions and indices are indices
#TODO check all dimensions (vectors, matrices) are corerect
#TODO check if all used functions are implemented
#TODO unit tests the SOINN algorithm
#TODO unit tests the individual functions

class SoinnPlus:
    def __init__(self, dim=2):
        '''
        Constructor for the class.

        Parameters:
        ----------
        dim : int, optional (default=2)
            Dimensionality of the input signal.

        '''

        # Parameters
        self.dimension = dim
        self.pull_factor = 100  # Determines how much the winner node pulls its neighbors during update; higher means less pull.
                                # In reference papers this is set to 100

        # SOINN+ Options
        # self.node_flag = node
        # self.edge_flag = edge

        # Data related variable
        self.winner_link_sim_th_M2 = np.zeros((1,2))
        self.winner_link_sim_th_mean = np.zeros((1,2))
        self.nodes = []
        self.track_input = []  #TODO check maybe delete this anyway
        self.track_input_idx = []   #TODO check maybe delete this anyway
        self.winning_times = []
        self.win_nums = []
        self.adjacency_mat = lil_matrix((0,0), dtype=int) # Track edges and edge ages. (0 = no edge, value-1 = age)
        self.links_created = 0
        self.signal_num = 0  
        self.node_count_del = 0
        self.edge_count_del = 0
        
        # Internal variables
        self.avg_unutility_del = 0
        self.avg_idle_del = 0
        self.edge_avg_age_del = 0

        self.enable_tracking = False
        self.param_edge = 2
        self.param_c = 2
        self.param_alpha = 2

        self.delete_edge_handler = self.delete_old_edges_plus
        self.delete_noise_handler = self.delete_noise_nodes_plus
        self.delete_node_period = 1

    def inference(self, stripsig):
        '''
        Inference from the trained network.
        
        Parameters:
        ----------
        stripsig: array-like
            row vector, input signal stripped of the dimesion to predict
        inference_dim: int
            index of the stripped value in the signal
        '''
        
        stripsig = self.check_signal(stripsig, on_inference=True)
        winner, sq_dist = self.find_nearest_nodes(1, stripsig, on_inference=True)

        # Check if the closest node is within similarity distance
        # if not, we cannot make any assumption on the signal as it is new for us
        # otherwise we assume that the cost of the signal is the same as the cost of
        # the closest node
        sim_threshold = self.calculate_similarity_threshold(winner[0], on_inference=True) 
        if sq_dist[0] > sim_threshold:
            return None, 0.0
        else:
            confidence = 1 - np.sqrt(sq_dist[0]) / (np.sqrt(sim_threshold) + 1e-10) # Add small epsilon to avoid division by zero
            return self.nodes[winner[0]][0,0], confidence
    

    def input_signal(self, signal):
        '''
        Input a signal to the SOINN+ algorithm.

        Parameters:
        ----------
        signal : array-like
            row vector, new input signal
        '''
        signal = self.check_signal(signal)
        self.signal_num += 1

        # If in initialization state add node unconditionally
        if len(self.nodes) < 3:
            self.add_node(signal)
            return
        
        # Find the winners and calculate similarity threshold
        winners, dists = self.find_nearest_nodes(2, signal)
        sim_thresholds = self.calculate_similarity_thresholds(winners)

        # Add node if either one of distance greater than corresponding similarity threshold
        if np.any(dists >= sim_thresholds):
            self.add_node(signal)

        else:
            # NODE MERGING
            self.update_winner(winners[0], signal)
            self.update_adjacent_nodes(winners[0], signal)
            
            # NODE LINKING
            # Calculate the trust level of first winner nodes.
            winning_times_nparray = np.array(self.winning_times)
            trustworthiness = (winning_times_nparray[winners] - 1) / (np.max(winning_times_nparray) - 1)
        
            # Condition 1: less than 3 edges in the network
            # Each undirected edge is stored as 2 entries (symmetric), so 3 edges = nnz of 6.
            if self.adjacency_mat.nnz < 6:
                edge_flag = True # create link unconditionally if there is no edge in the network
            
            # Condition 2 & 3:
            else:
                winner_link_sim_th_var = self.param_c*np.sqrt(self.winner_link_sim_th_M2/self.links_created)
                th = (self.winner_link_sim_th_mean + self.param_c*winner_link_sim_th_var).T

                edge_flag = np.any(np.sqrt(sim_thresholds) * (1 - trustworthiness.T) < th)

            # If Condition 1,2 or 3 is true add the edge or update the reset the edge lifetime if it already exists.
            if edge_flag:
                is_new = self.add_edge(winners)
                if is_new:
                    self.links_created += 1
                    pre_mean = self.winner_link_sim_th_mean.copy()
                    # Update the mean and variance of similarity threshold
                    self.winner_link_sim_th_mean += (np.sqrt(sim_thresholds.T) - self.winner_link_sim_th_mean) / self.links_created
                    self.winner_link_sim_th_M2 += (np.sqrt(sim_thresholds.T) - pre_mean) * (np.sqrt(sim_thresholds.T) - self.winner_link_sim_th_mean) 

            self.increment_adjacent_edge_ages(winners[0])

            # EDGE DELETION
            self.delete_edge_handler(winners[0])


        # NODE DELETION
        if self.signal_num % self.delete_node_period == 0:
            self.delete_noise_handler()
        
        #TODO if signal_num becomes very large we might want to reset it.

        #SANITY CHECK
        if not len(self.nodes) == self.adjacency_mat.shape[0] == len(self.winning_times) == len(self.win_nums):
            raise ValueError("Inconsistent SOINN state: nodes, adjacency matrix, winning times, and win nums should all have the same length.")
        
    def show(self, dims=[0, 1], data=None, cluster_labels=None, winning_times=False, save=False, save_path="tmp.png", plt_axis=None):
        """
        Display SOINN's network in 2D.

        Parameters:
        - dims: Which two dimensions to show (default: [0, 1]).
        - data: Optional input data to plot as blue crosses.
        - cluster_labels: Optional cluster label array (same length as self.nodes).
        - winning_times: If True, show winning times as labels near each node.
        - save: If True, saves the figure to a file.
        - save_path: Filename to save the figure if save is True.
        """
        nodes_nparray = np.vstack(self.nodes)
        # TODO: Handle error when node is shutting down but no nodes exist yet.
        if data is None:
            data = np.empty((0, len(dims)))

        plt.figure()
        if plt_axis:
            plt.axis(plt_axis)
        plt.grid(True)
        plt.gca().set_aspect('auto')
        plt.title("SOINN Network Visualization")

        # Show input data
        if data.shape[0] > 0:
            plt.plot(data[:, dims[0]], data[:, dims[1]], 'xb')

        # Show edges from sparse adjacency matrix
        for j, neighbors in enumerate(self.adjacency_mat.rows):
            for k in list(neighbors):
                if k >= j:  # avoid duplicate lines
                    nj = nodes_nparray[j]
                    nk = nodes_nparray[k]
                    plt.plot([nj[dims[0]], nk[dims[0]]], [nj[dims[1]], nk[dims[1]]], 'k')

        # Show nodes with or without clustering
        if cluster_labels is not None and len(cluster_labels) == nodes_nparray.shape[0]:
            cluster_num = int(np.max(cluster_labels))
            colors = cu.get_rgb_vectors(cluster_num + 1)
            noise_idx = cluster_labels == -1
            plt.plot(nodes_nparray[noise_idx, dims[0]], nodes_nparray[noise_idx, dims[1]],
                    '.', markersize=20, color=colors[0])
            for i in range(1, cluster_num + 1):
                idx = cluster_labels == i
                plt.plot(nodes_nparray[idx, dims[0]], nodes_nparray[idx, dims[1]],
                        '.', markersize=20, color=colors[i])
        else:
            plt.plot(nodes_nparray[:, dims[0]], nodes_nparray[:, dims[1]], '.b', markersize=10)

        # Show winning times
        if winning_times and hasattr(self, 'winning_times'):
            node_coords = nodes_nparray[:, dims]
            delta = (np.max(node_coords) - np.min(node_coords)) * 0.005
            for i in range(nodes_nparray.shape[0]):
                x = nodes_nparray[i, dims[0]] + delta
                y = nodes_nparray[i, dims[1]] + delta
                plt.text(x, y, str(self.winning_times[i]), backgroundcolor='white')

        if save:
            plt.savefig(save_path)

        plt.show()

    def topo_err(self, data):
        num_err = 0
        for i in range(data.shape[0]):
            near_idx, _ = self.find_nearest_nodes(2, data[i, :])
            if self.adjacency_mat[near_idx[0], near_idx[1]]:
                num_err += 1
        
        err = num_err / data.shape[0]
        return err

    def check_signal(self, signal, on_inference=False):
        '''
        Check the input signal and convert into a row vector with shape (1, dimension).

        Parameters:
        ----------
        signal : array-like
            Input signal to be checked.

        Returns:
        -------
        np.ndarray
            Converted row vector.
        '''
        if isinstance(signal, np.ndarray):
            s = signal
        elif isinstance(signal, csr_matrix):
            s = signal.toarray()
        elif isinstance(signal, list):
            s = np.array(signal)
        else:
            raise ValueError("Input signal must be a numpy array, list, or sparse matrix.")
            
        # Remove singleton dimensions
        s = np.squeeze(s)

        # Now only accept 1D vector of correct length
        if s.ndim != 1:
            raise ValueError(f"Input signal must be 1D after squeezing, got shape {s.shape}")
        if on_inference and s.shape[0] != self.dimension-1:
            raise ValueError(f"Input vector for inference must have length {self.dimension-1}, got {s.shape[0]}")
        if not on_inference and s.shape[0] != self.dimension:
            raise ValueError(f"Input vector must have length {self.dimension}, got {s.shape[0]}")
        
        return s.reshape(1, -1)

    
    def add_node(self, signal):
        '''
        Add a new node to the network.

        Parameters:
        ----------
        signal : array-like
            Input signal to be added as a new node.
        '''
        num = len(self.nodes)
        self.nodes.append(signal)
        self.winning_times.append(1)
        self.win_nums.append(self.signal_num)

        if num == 0:
            self.adjacency_mat = lil_matrix((1, 1), dtype=int)
        else:
            self.adjacency_mat.resize((num + 1, num + 1))
        
        if self.enable_tracking: # Maybe make this one dictionary
            self.track_input.append(signal)
            self.track_input_idx.append(self.signal_num)

    def find_nearest_nodes(self, num, signal, on_inference=False):
        indices = np.zeros(num, dtype=int)
        sq_dists = np.zeros(num)

        if on_inference:
            delta = np.vstack([node[0,1:] - signal for node in self.nodes])
        else:
            delta = np.vstack([node - signal for node in self.nodes])
        D = np.sum(delta ** 2, axis=1)
        for i in range(num):
            sq_dists[i] = np.min(D)
            indices[i] = np.argmin(D)
            D[indices[i]] = np.inf
        
        return indices, sq_dists
    
    def calculate_similarity_thresholds(self, node_indices):
        sim_thresholds = np.zeros((len(node_indices), 1))
        for i in range(len(node_indices)):
            sim_thresholds[i] = self.calculate_similarity_threshold(node_indices[i])

        return sim_thresholds
    
    def calculate_similarity_threshold(self, node_index, on_inference=False):
        # NOTE: This function claculates the squared similarity threshold. Fine for comparison.
        connected_indices = self.adjacency_mat.rows[node_index]
        if connected_indices != []:
            pals = np.vstack([self.nodes[i] for i in connected_indices])
            node_vec = self.nodes[node_index]
            if on_inference:
                D = np.sum((pals[:,1:] - node_vec[:,1:]) ** 2, axis=1)
            else:
                D = np.sum((pals - node_vec) ** 2, axis=1)
            threshold = max(D)
        else:
            if on_inference:
                # find nearest node in but with cost dimension and then calculate distance
                # without cost dimension
                winner, _ = self.find_nearest_nodes(2, self.nodes[node_index])
                delta = self.nodes[winner[1]][0,1:] - self.nodes[node_index][0,1:]
                threshold = np.sum(delta ** 2)
            else:
                _, sq_dists = self.find_nearest_nodes(2, self.nodes[node_index])
                threshold = sq_dists[1]

        return threshold
    
    def add_edge(self, node_indices):
        if self.adjacency_mat[node_indices[0], node_indices[1]] or self.adjacency_mat[node_indices[1], node_indices[0]]:
            is_new = False
        else:
            is_new = True
        # Reset edge age to 1 if edge already exists, otherwise create new edge with age 1
        # NOTE: In paper this is set to zero but we set it to one to avoid confusion with deleted edges.
        self.set_edge_age(node_indices[0], node_indices[1], 1)
        return is_new
    
    def update_winner(self, winner_index, signal):
        '''
        Update the winning node with the new signal.
        Parameters:
        ----------
        winner_index : int
            Index of the winning node.
        signal : array-like
            Row Vector Input signal to update the winning node.
        '''
        self.winning_times[winner_index] += 1
        w = self.nodes[winner_index]
        self.nodes[winner_index] = w + (signal - w) / self.winning_times[winner_index]
        self.win_nums[winner_index] = self.signal_num # Equivalent with setting the idle time to 0

        if self.enable_tracking:
            self.track_input[winner_index] = [self.track_input[winner_index], signal]
            self.track_input_idx[winner_index] = [self.track_input_idx[winner_index], self.signal_num]
        
    def update_adjacent_nodes(self, winner_index, signal):
        pals = list(self.adjacency_mat.rows[winner_index])
        ages = list(self.adjacency_mat.data[winner_index])
        for pal, age in zip(pals, ages):
            if pal == winner_index or age <= 0:
                continue

            w = self.nodes[pal]
            self.nodes[pal] = w + (signal - w) / (self.pull_factor * self.winning_times[pal])

    def increment_adjacent_edge_ages(self, winner_index):
        # Increment only existing edges (strictly positive entries)
        pals = list(self.adjacency_mat.rows[winner_index])
        ages = list(self.adjacency_mat.data[winner_index])
        for pal, age in zip(pals, ages):
            if pal == winner_index or age <= 0:
                continue
            self.set_edge_age(winner_index, pal, age + 1)
    
    def set_edge_age(self, i, j, age):
        if age == 0:
            self._remove_lil_entry(i, j)
            self._remove_lil_entry(j, i)
        else:
            self.adjacency_mat[i, j] = age
            self.adjacency_mat[j, i] = age

    def _remove_lil_entry(self, i, j):
        row = self.adjacency_mat.rows[i]
        data = self.adjacency_mat.data[i]
        if j in row:
            idx = row.index(j)
            row.pop(idx)
            data.pop(idx)

    def get_cluster(self, seed):
        n_nodes = self.adjacency_mat.shape[0]
        if seed < 0 or seed >= n_nodes:
            return set(), set()

        visited = set()
        stack = [seed]
        unique_edges = set()

        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)

            pals = list(self.adjacency_mat.rows[node])
            ages = list(self.adjacency_mat.data[node])
            for pal, age in zip(pals, ages):
                if age <= 0:
                    continue
                if pal == node:
                    continue
                if pal not in visited:
                    stack.append(pal)
                edge = (node, pal) if node < pal else (pal, node)
                unique_edges.add(edge)
        return visited, unique_edges

    def collect_cluster_ages(self, cluster_edges):
        edge_age = []
        for u, v in cluster_edges:
            age = self.adjacency_mat[u, v]
            if age > 0: # This should be the case for all edges in cluster.
                edge_age.append(age) # Subtract 1 to get the actual age since we set new edges to 1 TODO: check validity

        return np.array(edge_age)

    def delete_old_edges_plus(self, winner_index):
        # Collect the edge ages of the winners cluster
        cluster_nodes, cluster_edges = self.get_cluster(winner_index)
        cluster_ages = self.collect_cluster_ages(cluster_edges)
        if cluster_ages.size == 0:
            return
        pal_ages = np.array(self.adjacency_mat.data[winner_index])
        pals = np.array(self.adjacency_mat.rows[winner_index])
        c = np.percentile(cluster_ages, 75)
        th = self.param_edge * iqr(cluster_ages)

        if th == 0:
            return 
        
        outlierness = c + th
        ratio = self.edge_count_del / (self.edge_count_del + cluster_ages.size)
        del_threshold = self.edge_avg_age_del*ratio + outlierness*(1-ratio)

        # Check if there are any edges to be deleted
        indices_to_delete = pals[pal_ages > del_threshold]
        deleted_ages = pal_ages[pal_ages > del_threshold]
        for i in indices_to_delete:
            self.set_edge_age(i, winner_index, 0)

        # Update average lifetime of deleted edges
        if indices_to_delete.size > 0:
            self.edge_avg_age_del = (self.edge_count_del*self.edge_avg_age_del + np.sum(deleted_ages)) / (self.edge_count_del + indices_to_delete.size)
            self.edge_count_del += indices_to_delete.size

        return

    def delete_nodes(self, indices):
        indices = np.asarray(list(set(indices)))

        # Build keep-mask once and reuse for all structures
        mask = np.ones(len(self.nodes), dtype=bool)
        mask[indices] = False

        # O(n) filter instead of O(n*k) sequential del
        self.nodes = [n for i, n in enumerate(self.nodes) if mask[i]]
        self.winning_times = [t for i, t in enumerate(self.winning_times) if mask[i]]
        self.win_nums = [n for i, n in enumerate(self.win_nums) if mask[i]]

        # Sparse matrix: delete rows/columns via mask
        adj = self.adjacency_mat.tocsr()
        self.adjacency_mat = adj[mask, :][:, mask].tolil()

        if self.enable_tracking:
            self.track_input = np.delete(self.track_input, indices)
            self.track_input_idx = np.delete(self.track_input_idx, indices)

        self.node_count_del += len(indices)

    def delete_noise_nodes_plus(self):
        degrees = self.adjacency_mat.getnnz(axis=1) # Number of connected edges per node
        noise_nodes = degrees < 1 # Nodes that are not connected to any other node
        active_nodes = ~noise_nodes
        if not np.any(active_nodes): # Do not delete if no active nodes are available.
            return

        win_nums_nparray = np.array(self.win_nums)
        idle_times = self.signal_num - win_nums_nparray
        winning_times = np.array(self.winning_times, dtype=float)
        unutility = idle_times / winning_times

        active_unutility = unutility[active_nodes]
        outlierness = np.median(active_unutility) + 2*median_abs_deviation(active_unutility, scale='normal')

        # Check if there are any nodes that should be deleted.
        active_count = np.sum(active_nodes, axis=0)
        ratio = self.node_count_del / (self.node_count_del + active_count)
        noise_lv = np.sum(noise_nodes, axis=0) / active_nodes.shape[0]
        del_threshold = self.avg_unutility_del * ratio + outlierness * (1 - ratio) * (1 - noise_lv)

        inactive_idx = unutility > del_threshold

        if np.any(inactive_idx & noise_nodes):
            # Tracking the average deleted idle time and unutility 
            self.avg_idle_del = (self.node_count_del * self.avg_idle_del + np.sum(idle_times[inactive_idx & noise_nodes], axis=0)) / (self.node_count_del + np.sum(inactive_idx & noise_nodes, axis=0))
            self.avg_unutility_del = (self.node_count_del * self.avg_unutility_del + np.sum(unutility[inactive_idx & noise_nodes], axis=0)) / (self.node_count_del + np.sum(inactive_idx & noise_nodes, axis=0))
            
            self.delete_nodes(np.where(inactive_idx & noise_nodes)[0])
            

