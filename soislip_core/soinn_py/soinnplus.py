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
    def __init__(self, dim=2, lambda_=300, age_max=50):
        '''
        Constructor for the class.

        Parameters:
        ----------
        lambda_ : int, optional (default=300)
            A period for deleting nodes. Nodes that do not satisfy
            certain conditions are removed every `lambda_` steps.

        age_max : int, optional (default=50)
            Maximum allowed age for edges. Edges exceeding this age
            are deleted.

        dim : int, optional (default=2)
            Dimensionality of the input signal.

        node : int (0 or 1), optional (default=1)
            Enable (1) or disable (0) the 'plus' version of node deletion.

        edge : int (0 or 1), optional (default=1)
            Enable (1) or disable (0) the 'plus' version of edge linking.
        '''

        # Parameters
        self.dimension = dim
        self.delete_node_period = lambda_
        self.max_edge_age = age_max

        # SOINN+ Options
        # self.node_flag = node
        # self.edge_flag = edge

        # Data related variable
        self.run_th_variance = np.zeros((1,2))
        self.run_th_mean = np.zeros((1,2))
        self.nodes = []
        self.track_input = []  #TODO check maybe delete this anyway
        self.track_input_idx = []   #TODO check maybe delete this anyway
        self.winning_times = []
        self.win_ts = []
        self.node_ts = []
        self.adjacency_mat = lil_matrix((0,0), dtype=int)
        self.links_created = 0
        self.signal_num = 0  
        self.node_deleted = 0
        self.edge_deleted = 0
        
        # Internal variables
        self.node_del_th = 0
        self.node_avg_idle_del = 0
        self.edge_avg_lt_del = 0

        self.cur_node_th = 0
        self.cur_edge_th = 0
        self.enable_tracking = False
        self.min_degree = 1
        self.param_edge = 2
        self.param_c = 2
        self.param_alpha = 2

        # if  not self.node_flag:
        #     self.delete_noise_handler = self.delete_noise_handler_original
        #     # TODO think about the handlers
        # else:
        #     self.delete_noise_handler = self.delete_noise_handler_plus
        #     self.delete_node_period = 1;
        
        # if not self.edge_flag:
        #     self.delete_edge_handler = self.delete_edge_handler_original
        # else:
        #     self.delete_edge_handler = self.delete_edge_handler_plus

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
        winner, dists = self.find_nearest_nodes(2, signal)
        sim_thresholds = self.calculate_similarity_thresholds(winner)

        # Check if the network should create the link between both winners or not
        if np.any(self.run_th_variance == 0):
            e_flag = True # create link unconditionally if there is no edge in the network
        else:
            # Check if edge between both winner should be created or not
            th = self.param_c*np.sqrt(self.run_th_variance/self.links_created)

            # Compute the degree (number of connections per node)
            degrees = self.adjacency_mat.tocsr().sum(axis=1).A1

            # Identify 'noisy' nodes (degree < min_degree)
            noises = degrees < self.min_degree

            # Calculate the trust level
            winning_times_nparray = np.array(self.winning_times)
            trust_lv = winning_times_nparray[winner] / np.max(winning_times_nparray[~noises])


            e_flag = np.any(np.sqrt(sim_thresholds) * (1 - trust_lv.T) < (self.run_th_mean + th).T)

        # Add node if either one of distance greater than corresponding similarity threshold
        if np.any(dists > sim_thresholds):
            self.add_node(signal)
        else:

            # Add edge if the condition is true
            if e_flag:
                is_new = self.add_edge(winner)
                if is_new:
                    self.links_created += 1
                    pre_mean = self.run_th_mean
                    # Update the mean and variance of similarity threshold
                    self.run_th_mean += (np.sqrt(sim_thresholds.T) - self.run_th_mean) / self.links_created
                    self.run_th_variance += (np.sqrt(sim_thresholds.T) - pre_mean) * (np.sqrt(sim_thresholds.T) - self.run_th_mean) 

            self.increment_edge_ages(winner[0])
            winner[0] = self.delete_edge_handler(winner[0])
            self.update_winner(winner[0], signal)
            self.update_adjacent_nodes(winner[0], signal)

        # Check if any node can be deleted
        if self.signal_num % self.delete_node_period == 0:
            self.delete_noise_handler()
        
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
            for k in neighbors:
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
            near_idx = self.find_nearest_nodes(2, data[i, :])
            if self.adjacency_mat[near_idx[0], near_idx[1]]:
                num_err += 1
        
        err = num_err / data.shape[0]
        return err

    # TODO implement label_with_breadth_first_search
        
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
        idx : int
            Index of the node to be added.
        '''
        num = len(self.nodes)
        self.nodes.append(signal)
        self.winning_times.append(1)
        self.win_ts.append(self.signal_num)
        self.node_ts.append(self.signal_num)

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
            is_new = True
        else:
            is_new = False
        self.adjacency_mat[node_indices[0], node_indices[1]] = 1
        self.adjacency_mat[node_indices[1], node_indices[0]] = 1
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
        self.win_ts[winner_index] = self.signal_num

        if self.enable_tracking: # TODO maybe this can be made to a dictionary
            self.track_input[winner_index] = [self.track_input[winner_index], signal]
            self.track_input_idx[winner_index] = [self.track_input_idx[winner_index], self.signal_num]
        
    def update_adjacent_nodes(self, winner_index, signal):
        pals = self.adjacency_mat.rows[winner_index]
        for pal in pals:
            w = self.nodes[pal]
            self.nodes[pal] = w + (signal - w) / (100 * self.winning_times[pal])

    def increment_edge_ages(self, winner_index):
        indices = self.adjacency_mat.rows[winner_index]
        for i in indices:
            self.increment_edge_age(winner_index, i)

    def increment_edge_age(self, i, j):
        self.adjacency_mat[i, j] += 1
        self.adjacency_mat[j, i] += 1
    
    def set_edge_age(self, i, j, age):
        self.adjacency_mat[i, j] = age
        self.adjacency_mat[j, i] = age

    # TODO implement delete_old_edges_original

    def collect_cluster_edge_age(self, seed):
        # Traverse the connected component of `seed` directly on the sparse adjacency
        # matrix and collect unique undirected edge ages.
        n_nodes = self.adjacency_mat.shape[0]
        if seed < 0 or seed >= n_nodes:
            return np.array([])

        visited = set()
        stack = [seed]
        unique_edges = set()

        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)

            neighbors = self.adjacency_mat.rows[node]
            weights = self.adjacency_mat.data[node]
            for neighbor, weight in zip(neighbors, weights):
                if weight <= 0:
                    continue
                if neighbor == node:
                    continue
                if neighbor not in visited:
                    stack.append(neighbor)
                edge = (node, neighbor) if node < neighbor else (neighbor, node)
                unique_edges.add(edge)

        edge_age = []
        for u, v in unique_edges:
            weight = self.adjacency_mat[u, v]
            if weight > 0:
                edge_age.append(weight)

        return np.array(edge_age)

    def delete_old_edges_plus(self, winner_index):
        edge_age = np.array(self.collect_cluster_edge_age(winner_index))
        if edge_age.size == 0:
            return winner_index
        
        edges = np.array(self.adjacency_mat.data[winner_index])
        indices = np.array(self.adjacency_mat.rows[winner_index])
        c = np.percentile(edge_age, 75)
        th = self.param_edge * iqr(edge_age)

        if th == 0:
            return winner_index
        
        cur_th = c + th
        ratio = self.edge_deleted / (self.edge_deleted + edge_age.size)

        # Check if there are any edges to be deleted
        del_threshold = self.edge_avg_lt_del*ratio + cur_th*(1-ratio)
        indices_to_delete = indices[edges > del_threshold]
        edges_to_delete = edges[edges > del_threshold]
        self.cur_node_th = del_threshold

        # Update average lifetime of deleted edges
        if indices_to_delete.size > 0:
            self.edge_avg_lt_del = (self.edge_deleted*self.edge_avg_lt_del + np.sum(edges_to_delete)) / (self.edge_deleted + indices_to_delete.size)
        
        self.edge_deleted += indices_to_delete.size
        deleted_node_indices = []
        for i in indices_to_delete:
            self.set_edge_age(i, winner_index, 0)
            if self.adjacency_mat.getcol(i).sum() == 0:
                deleted_node_indices.append(i)
        
        # Update winner index according to the deleted node
        winner_index -= np.sum(deleted_node_indices < winner_index)
        self.delete_nodes(deleted_node_indices)

        return winner_index

    def delete_nodes(self, indices):
        # Ensure indices are sorted in reverse to safely delete from lists
        indices = sorted(set(indices), reverse=True)

        for i in indices:
            del self.nodes[i]
            del self.winning_times[i]
            del self.win_ts[i]
            del self.node_ts[i]

        # Sparse matrix: delete rows/columns via mask
        adj = self.adjacency_mat.tocsr()
        mask = np.ones(adj.shape[0], dtype=bool)
        mask[indices] = False
        self.adjacency_mat = adj[mask,:][:, mask].tolil()

        if self.enable_tracking:
            self.track_input = np.delete(self.track_input, indices)
            self.track_input_idx = np.delete(self.track_input_idx, indices)

        self.node_deleted += np.sum(indices)

    # TODO implement delete_noise_nodes_original
    
    def delete_noise_nodes_plus(self):
        degrees = self.adjacency_mat.tocsr().sum(axis=1).A1  # Convert to flat NumPy array
        noises = degrees < self.min_degree
        data = ~noises
        win_ts_nparray = np.array(self.win_ts)
        IT = self.signal_num - win_ts_nparray
        UT = IT / np.array(self.winning_times)

        TF, l, u, c = isoutlier(UT[data])
        th = self.param_alpha * ((u - c)/3)
        cur_th = c + th

        #check if there are any nodes that should be deleted
        ratio = self.node_deleted / (self.node_deleted + np.sum(data, axis=0))
        noise_lv = np.sum(noises, axis=0) / data.shape[0]
        del_threshold = self.node_del_th * ratio + cur_th * (1 - ratio) * (1 - noise_lv)
        inactive_idx = UT > del_threshold
        self.cur_edge_th = del_threshold

        if np.any(inactive_idx & noises):
            # Tracking the average deleted idle time and unutility 
            self.node_avg_idle_del = (self.node_deleted * self.node_avg_idle_del + np.sum(IT[inactive_idx & noises], axis=0)) / (self.node_deleted + np.sum(inactive_idx & noises, axis=0))
            self.node_del_th = (self.node_deleted * self.node_del_th + np.sum(UT[inactive_idx & noises], axis=0)) / (self.node_deleted + np.sum(inactive_idx & noises, axis=0))

        self.delete_nodes(np.where(inactive_idx & noises)[0])

# ---
# TODO move this to a utils file
def isoutlier(data):
    '''
    Is Outlier function from MATLAB using "median":
    Outliers are defined as elements more than three scaled MAD from the median. The scaled MAD is 
    defined as c*median(abs(A-median(A))), where c=-1/(sqrt(2)*erfcinv(3/2)).

    

    Parameters:
    ----------
    data : array-like
        Input data to check for outliers.

    Returns:
    -------
    TF : array-like(data.shape), dtype=bool
        Boolean array indicating whether each element is an outlier.

    l : float
    the lower threshold value of the default outlier detection method is three scaled MAD below 
    the median of the input data.

    u : float
    the upper threshold value of the default outlier detection method is three scaled MAD above 
    the median of the input data.

    c : float
    the center value of the default outlier detection method is the median of the input data.
    '''
    c = np.median(data)
    mad = median_abs_deviation(data, scale='normal')
    l = c - 3 * mad
    u = c + 3 * mad

    TF = (data < l) | (data > u)

    return TF, l, u, c

    
