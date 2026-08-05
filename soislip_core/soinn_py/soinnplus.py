import numpy as np
import sys #TODO remove if not needed anymore
from typing import Callable, Sequence, TypeAlias
from .color_utils import ColorUtils as cu
from scipy.sparse import csr_matrix, lil_matrix
from scipy.stats import iqr, median_abs_deviation
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt

SignalLike: TypeAlias = np.ndarray | csr_matrix | list[float]
Prediction: TypeAlias = tuple[float | None, float | None]
BatchSignals: TypeAlias = Sequence[SignalLike] | np.ndarray
BatchPredictions: TypeAlias = Sequence[Prediction] | np.ndarray

#TODO check if all functions are functions and indices are indices
#TODO check all dimensions (vectors, matrices) are corerect
#TODO check if all used functions are implemented
#TODO unit tests the SOINN algorithm
#TODO unit tests the individual functions

class SoinnPlus:
    def __init__(self, dim: int = 2) -> None:
        '''
        Constructor for the class.

        Parameters:
        ----------
        dim : int, optional (default=2)
            Dimensionality of the input signal.

        '''

        # ------ Parameters ------
        self.dimension = dim
        self.pull_factor = 100  
        # Determines how much the winner node pulls its neighbors during update; higher means less pull.
        # In reference papers this is set to 100

        self.manually_label_clusters_period = 20 
        # Number of signals after which cluster labeling is performed (if no label is provided) 
        # Set to 1 to label every signal, set to a large number to only label when a new label is 
        # provided. Larger values can save computation time if many labels are expected.

        self.label_weighted_by_density = True 
        # Whether to weight the cluster labels by density when calculating the cluster prediction.
        # Set to False if many labels are expected or to save computation time.

        # ------ Data related variables ------
        self.winner_link_sim_th_M2 = np.zeros((1,2))
        self.winner_link_sim_th_mean = np.zeros((1,2))
        self.nodes = [] # List of node weight vectors (row vectors with shape (1, dimension))
        self.winning_count = [] # List of winning times for each node (number of times each node has won)
        self.idle_since = [] # List of signal timestamps the node has been idle since (node has not won since this signal number)
        self.labels = [] # List of labels for each node
        self.label_count = [] # List of label update times for each node (number of times each node's label has been updated)
        self.predictions = [] # List of label predictions for each node (tuple of (mean, variance))
        self.adjacency_mat = lil_matrix((0,0), dtype=int) # Track edges and edge ages. (0 = no edge, value-1 = age)
        self.track_input = []
        self.track_input_idx = []
        self.links_created = 0
        self.signal_num = 0  
        self.node_count_del = 0
        self.edge_count_del = 0
        
        # --- Internal variables ---
        self.avg_unutility_del = 0
        self.avg_idle_del = 0
        self.edge_avg_age_del = 0

        self.enable_tracking = False
        self.param_edge = 2
        self.param_c = 2
        self.param_alpha = 2
        self.fallback_count = 0

        self._delete_edge_handler: Callable[[int], None] = self._delete_old_edges_plus
        self._delete_noise_handler: Callable[[], None] = self._delete_noise_nodes_plus
        self._label_cluster_handler: Callable[[Sequence[int]], bool] = self._label_cluster_median
        self._fallback_prediction_handler: Callable[[SignalLike], Prediction] = self._next_winner_fallback
        
    def inference(self, signal: SignalLike, label_clusters: bool = True) -> Prediction:
        '''
        Inference from the trained network.
        
        Parameters:
        ----------
        signal: array-like
            row vector, input signal
        label_clusters: bool, optional (default=True)
            Leave true for semi- or unsupervised learning, otherwise clusters are labeled when a label is provided.
        
        Returns:
        prediction: tuple (mean, variance)
            None if no similar node is found.
        '''
        
        if len(self.nodes) < 3:
            raise ValueError("At least 3 nodes are required for inference, but only {} nodes are present.".format(len(self.nodes)))
        if self.labels.count(None) == len(self.labels):
            raise ValueError("No labels are available for inference. Please provide labels for the nodes before performing inference.")

        winner_index = self.find_winner(signal)

        if label_clusters:
            cluster_nodes, _ = self._get_cluster(winner_index)
            self._label_cluster_handler(cluster_nodes)

        prediction = self.predictions[winner_index]

        if prediction[0] is None:
            prediction = self._fallback_prediction_handler(signal)
            self.fallback_count += 1

        return prediction

    
    def batch_inference(self, signals: BatchSignals) -> BatchPredictions:
        '''
        Batch inference from the trained network.
        
        Parameters:
        ----------
        signals: list of array-like
            list of row vectors, input signals
        
        Returns:
        predictions: list of tuple (mean, variance)
            None if no similar node is found.
        '''
        signals_array = np.asarray(signals, dtype=object if isinstance(signals, list) else None)
        if isinstance(signals_array, np.ndarray) and signals_array.ndim == 1 and signals_array.size > 0 and not isinstance(signals_array[0], (np.ndarray, list, tuple)):
            signals_iterable = [signals_array]
        elif isinstance(signals_array, np.ndarray) and signals_array.ndim == 2:
            signals_iterable = signals_array
        else:
            signals_iterable = signals

        predictions = []
        for signal in signals_iterable:
            prediction = self.inference(signal)
            predictions.append(prediction)
        predictions = np.array(predictions, dtype=float)
        return predictions
    
    def _calculate_density(self, node_index: int) -> float:
        pals = list(self.adjacency_mat.rows[node_index])
        if len(pals) == 0:
            return 0.0
        node_vec = self.nodes[node_index]
        pal_vecs = np.vstack([self.nodes[i] for i in pals])
        D = np.sum(np.sqrt((pal_vecs - node_vec) ** 2), axis=1)
        density = 1 / (1 + np.mean(D))**2 if D.size > 0 else 0
        return density
    
    def _label_cluster_mean(self, cluster_nodes: list[int], density_weighted: bool = True) -> bool:
        labeled_nodes = [i for i in cluster_nodes if self.labels[i] is not None]
        if len(labeled_nodes) == 0:
            return False
        cluster_labels = np.array([self.labels[i] for i in labeled_nodes])
        if len(cluster_labels) == 1:
            prediction = cluster_labels[0]
        else:
            densities = np.array([self._calculate_density(i) for i in labeled_nodes])
            density_sum = densities.sum()
            weights = densities / density_sum if density_sum > 0 else np.ones(len(densities)) / len(densities)
            weighted_mean = np.dot(weights, cluster_labels)
            prediction = weighted_mean
        # weighted_var = np.dot(weights, (cluster_labels - weighted_mean) ** 2)
        for i in cluster_nodes:
            density = self._calculate_density(i)
            self.predictions[i] = (prediction, density)
        return True
    
    def _label_cluster_median(self, cluster_nodes: list[int], density_weighted: bool = True) -> bool:
        labeled_nodes = [i for i in cluster_nodes if self.labels[i] is not None]
        if len(labeled_nodes) == 0:
            return False
        cluster_labels = np.array([self.labels[i] for i in labeled_nodes])
        if len(cluster_labels) == 1:
            prediction = cluster_labels[0]
        else:
            densities = np.array([self._calculate_density(i) for i in labeled_nodes])
            density_sum = densities.sum()
            weights = densities / density_sum if density_sum > 0 else np.ones(len(densities)) / len(densities)
            sort_idx = np.argsort(cluster_labels)
            sorted_labels = cluster_labels[sort_idx]
            sorted_weights = weights[sort_idx]
            cumulative_weights = np.cumsum(sorted_weights)
            prediction = sorted_labels[np.searchsorted(cumulative_weights, 0.5, side='left')]
        for i in cluster_nodes:
            density = self._calculate_density(i)
            self.predictions[i] = (prediction, density)
        return True

    def _label_cluster_distance_based(self, cluster_nodes: list[int], density_weighted: bool = True) -> bool:
        labeled_nodes = [i for i in cluster_nodes if self.labels[i] is not None]
        if len(labeled_nodes) == 0:
            return False
        ln_vecs = np.vstack([self.nodes[i] for i in labeled_nodes])
        cn_vecs = np.vstack([self.nodes[i] for i in cluster_nodes])
        closest_ln = np.argmin(np.linalg.norm(ln_vecs[:, np.newaxis] - cn_vecs, axis=2), axis=0)
        closest_ln_idx = [labeled_nodes[i] for i in closest_ln]
        for pos, node_index in enumerate(cluster_nodes):
            prediction = self.labels[closest_ln_idx[pos]]
            density = self._calculate_density(node_index)
            self.predictions[node_index] = (prediction, density)
        return True
    
    def _next_winner_fallback(self, signal: SignalLike) -> Prediction:
        prediction = (None, None)
        seen_nodes = set()
        nn = 1
        while (len(seen_nodes) < len(self.nodes)):
            nearest_indices, _ = self._find_nearest_nodes(nn, signal)
            winner_index = nearest_indices[-1]
            if winner_index in seen_nodes:
                nn += 1
                continue
            cluster_nodes, _ = self._get_cluster(winner_index)
            seen_nodes.update(cluster_nodes) #TODO: check this.
            if self._label_cluster_handler(cluster_nodes):
                prediction = self.predictions[winner_index]
                break
            nn += 1
        return prediction

    def _closest_label_fallback(self, signal: SignalLike) -> Prediction:
        valid_nodes = [i for i, l in enumerate(self.labels) if l is not None]
        valid_vecs = np.vstack([self.nodes[i] for i in valid_nodes])
        closest_idx = np.argmin(np.linalg.norm(valid_vecs - signal, axis=1))
        closest_label_node = valid_nodes[closest_idx]
        cluster, _ = self._get_cluster(closest_label_node)
        self._label_cluster_handler(cluster)
        prediction = self.predictions[valid_nodes[closest_idx]]
        return prediction


    def count_clusters(self) -> int:
        if self.adjacency_mat.shape[0] == 0:
            return 0

        adj = self.adjacency_mat.tocsr()

        # Keep only nodes connected to at least one other node
        active = adj.getnnz(axis=1) > 0
        if not np.any(active):
            return 0

        sub = adj[active, :][:, active]

        # Binary connectivity (age values do not matter for component count)
        sub.data = np.ones_like(sub.data)

        n_components, _ = connected_components(sub, directed=False)
        return int(n_components)

    def _update_label(self, node_index: int, label: float | None, is_winner: bool = True) -> None:
        # Update labels like any other dimension
        if label is None:
            return
        if is_winner:
            self.label_count[node_index] += 1
            if self.labels[node_index] is None: 
                self.labels[node_index] = label
            else:
                self.labels[node_index] += (label - self.labels[node_index]) / (self.label_count[node_index])
        elif self.labels[node_index] is not None and self.label_count[node_index] > 0:
                self.labels[node_index] += (label - self.labels[node_index]) / (self.pull_factor*self.label_count[node_index])

    def input_signal(self, signal: SignalLike, label: float | None = None) -> None:
        '''
        Input a signal to the SOINN+ algorithm.

        Parameters:
        ----------
        signal : array-like
            row vector, new input signal
        label : optional
            label for the input signal (continous)
        '''
        signal = self._check_signal(signal)
        self.signal_num += 1

        # If in initialization state add node unconditionally
        if len(self.nodes) < 3:
            self._add_node(signal, (label, None), label)
            return
        
        # Find the winners and calculate similarity threshold
        winners, dists = self._find_nearest_nodes(2, signal)
        sim_thresholds = self._calculate_similarity_thresholds(winners)

        # Add node if either one of distance greater than corresponding similarity threshold
        if np.any(dists >= sim_thresholds):
            self._add_node(signal, self.predictions[winners[0]], label)

        else:
            # NODE MERGING
            self._update_winner(winners[0], signal, label)
            self._update_adjacent_nodes(winners[0], signal)
            
            # NODE LINKING
            # Calculate the trust level of first winner nodes.
            winning_count_nparray = np.array(self.winning_count)
            trustworthiness = (winning_count_nparray[winners] - 1) / (np.max(winning_count_nparray) - 1)
        
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
                is_new = self._add_edge(winners)
                if is_new:
                    self.links_created += 1
                    pre_mean = self.winner_link_sim_th_mean.copy()
                    # Update the mean and variance of similarity threshold
                    self.winner_link_sim_th_mean += (np.sqrt(sim_thresholds.T) - self.winner_link_sim_th_mean) / self.links_created
                    self.winner_link_sim_th_M2 += (np.sqrt(sim_thresholds.T) - pre_mean) * (np.sqrt(sim_thresholds.T) - self.winner_link_sim_th_mean) 

            self._increment_adjacent_edge_ages(winners[0])

            # EDGE DELETION
            self._delete_edge_handler(winners[0])
            
            # CLUSTER LABELING
            if label is not None or self.signal_num % self.manually_label_clusters_period == 0:
                cluster_nodes, _ = self._get_cluster(winners[0])
                self._label_cluster_handler(cluster_nodes)

        # NODE DELETION
        self._delete_noise_handler()
        
        #TODO if signal_num becomes very large we might want to reset it.

        #SANITY CHECK
        if not len(self.nodes) == self.adjacency_mat.shape[0] == len(self.winning_count) == len(self.idle_since):
            raise ValueError("Inconsistent SOINN state: nodes, adjacency matrix, winning times, and win nums should all have the same length.")
        
    def show(self, save: bool = False, save_path: str = "tmp.png") -> None:
        """
        Display SOINN's network in 3D: X/Y are the first two input dimensions,
        Z is the predicted label (mean) for each node. Nodes without a prediction
        are plotted at Z = -0.1.

        Parameters:
        - save: If True, saves the figure to a file.
        - save_path: Filename to save the figure if save is True.
        """
        nodes_nparray = np.vstack(self.nodes)
        node_labels = np.array([p[0] if p[0] is not None else -0.1 for p in self.predictions])

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Show nodes
        ax.scatter(nodes_nparray[:, 0], nodes_nparray[:, 1], node_labels,
                   c='g', s=10, alpha=0.9, label=f"Nodes ({len(self.nodes)})")

        # Show edges
        for j, neighbors in enumerate(self.adjacency_mat.rows):
            for k in neighbors:
                if k >= j:
                    nj = nodes_nparray[j]
                    nk = nodes_nparray[k]
                    ax.plot([nj[0], nk[0]], [nj[1], nk[1]],
                            [node_labels[j], node_labels[k]], 'k', alpha=0.5)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Prediction")
        ax.set_box_aspect([1, 1, 1])
        ax.legend()
        ax.set_title("SOINN Network Visualization")

        if save:
            plt.savefig(save_path)
        else:
            plt.show()

    def _topo_err(self, data: np.ndarray) -> float:
        num_err = 0
        for i in range(data.shape[0]):
            near_idx, _ = self._find_nearest_nodes(2, data[i, :])
            if self.adjacency_mat[near_idx[0], near_idx[1]]:
                num_err += 1
        
        err = num_err / data.shape[0]
        return err

    def _check_signal(self, signal: SignalLike) -> np.ndarray:
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
        if s.shape[0] != self.dimension:
            raise ValueError(f"Input vector must have length {self.dimension}, got {s.shape[0]}")
        
        return s.reshape(1, -1)

    
    def _add_node(self, signal: np.ndarray, winners_prediction: Prediction, label: float | None = None) -> None:
        '''
        Add a new node to the network.

        Parameters:
        ----------
        signal : array-like
            Input signal to be added as a new node.
        label : optional
            Label for the new node.
        '''
        num = len(self.nodes)
        self.nodes.append(signal)
        self.winning_count.append(1)
        self.idle_since.append(self.signal_num)
        self.labels.append(label)
        if label is not None:
            self.label_count.append(1)
        else:
            self.label_count.append(0)
        self.predictions.append(winners_prediction)

        if num == 0:
            self.adjacency_mat = lil_matrix((1, 1), dtype=int)
        else:
            self.adjacency_mat.resize((num + 1, num + 1))
        
        if self.enable_tracking: # Maybe make this one dictionary
            self.track_input.append(signal)
            self.track_input_idx.append(self.signal_num)

    def _find_nearest_nodes(self, num: int, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        indices = np.zeros(num, dtype=int)
        sq_dists = np.zeros(num)

        delta = np.vstack([node - signal for node in self.nodes])
        D = np.sum(delta ** 2, axis=1)
        for i in range(num):
            sq_dists[i] = np.min(D)
            indices[i] = np.argmin(D)
            D[indices[i]] = np.inf
        
        return indices, sq_dists
    
    def find_winner(self, signal: np.ndarray) -> int:
        """
        Find the index of the nearest neighbor node to the given signal.
        """
        checked_signal = self._check_signal(signal)
        indices, _ = self._find_nearest_nodes(1, checked_signal)
        return indices[0]
    
    def _calculate_similarity_thresholds(self, node_indices: Sequence[int] | np.ndarray) -> np.ndarray:
        sim_thresholds = np.zeros((len(node_indices), 1))
        for i in range(len(node_indices)):
            sim_thresholds[i] = self._calculate_similarity_threshold(node_indices[i])

        return sim_thresholds
    
    def _calculate_similarity_threshold(self, node_index: int) -> float:
        # NOTE: This function claculates the squared similarity threshold. Fine for comparison.
        connected_indices = self.adjacency_mat.rows[node_index]
        if connected_indices != []:
            pals = np.vstack([self.nodes[i] for i in connected_indices])
            node_vec = self.nodes[node_index]
            D = np.sum((pals - node_vec) ** 2, axis=1)
            threshold = max(D)
        else:
            _, sq_dists = self._find_nearest_nodes(2, self.nodes[node_index])
            threshold = sq_dists[1]

        return threshold
    
    def _add_edge(self, node_indices: Sequence[int] | np.ndarray) -> bool:
        if self.adjacency_mat[node_indices[0], node_indices[1]] or self.adjacency_mat[node_indices[1], node_indices[0]]:
            is_new = False
        else:
            is_new = True
        # Reset edge age to 1 if edge already exists, otherwise create new edge with age 1
        # NOTE: In paper this is set to zero but we set it to one to avoid confusion with deleted edges.
        self._set_edge_age(node_indices[0], node_indices[1], 1)
        return is_new
    
    def _update_winner(self, winner_index: int, signal: np.ndarray, label: float | None = None) -> None:
        '''
        Update the winning node with the new signal.
        Parameters:
        ----------
        winner_index : int
            Index of the winning node.
        signal : array-like
            Row Vector Input signal to update the winning node.
        label : optional
            Label for the winning node.
        '''
        self.winning_count[winner_index] += 1
        w = self.nodes[winner_index]
        self.nodes[winner_index] = w + (signal - w) / self.winning_count[winner_index]
        self.idle_since[winner_index] = self.signal_num # Equivalent with setting the idle time to 0
        self._update_label(winner_index, label)

        if self.enable_tracking:
            self.track_input[winner_index] = [self.track_input[winner_index], signal]
            self.track_input_idx[winner_index] = [self.track_input_idx[winner_index], self.signal_num]
        
    def _update_adjacent_nodes(self, winner_index: int, signal: np.ndarray) -> None:
        pals = list(self.adjacency_mat.rows[winner_index])
        ages = list(self.adjacency_mat.data[winner_index])
        for pal, age in zip(pals, ages):
            if pal == winner_index or age <= 0:
                continue

            w = self.nodes[pal]
            self.nodes[pal] = w + (signal - w) / (self.pull_factor * self.winning_count[pal])
            self._update_label(pal, self.labels[winner_index], is_winner=False)

    def _increment_adjacent_edge_ages(self, winner_index: int) -> None:
        # Increment only existing edges (strictly positive entries)
        pals = list(self.adjacency_mat.rows[winner_index])
        ages = list(self.adjacency_mat.data[winner_index])
        for pal, age in zip(pals, ages):
            if pal == winner_index or age <= 0:
                continue
            self._set_edge_age(winner_index, pal, age + 1)
    
    def _set_edge_age(self, i: int, j: int, age: int) -> None:
        if age == 0:
            self._remove_lil_entry(i, j)
            self._remove_lil_entry(j, i)
        else:
            self.adjacency_mat[i, j] = age
            self.adjacency_mat[j, i] = age

    def _remove_lil_entry(self, i: int, j: int) -> None:
        row = self.adjacency_mat.rows[i]
        data = self.adjacency_mat.data[i]
        if j in row:
            idx = row.index(j)
            row.pop(idx)
            data.pop(idx)

    def _get_cluster(self, seed: int) -> tuple[set[int], set[tuple[int, int]]]:
        n_nodes = self.adjacency_mat.shape[0]
        if seed < 0 or seed >= n_nodes:
            raise ValueError(f"Seed index {seed} is out of bounds for adjacency matrix with {n_nodes} nodes.")

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

    def _collect_cluster_ages(self, cluster_edges: set[tuple[int, int]]) -> np.ndarray:
        edge_age = []
        for u, v in cluster_edges:
            age = self.adjacency_mat[u, v]
            if age > 0: # This should be the case for all edges in cluster.
                edge_age.append(age) # Subtract 1 to get the actual age since we set new edges to 1 TODO: check validity

        return np.array(edge_age)

    def _delete_old_edges_plus(self, winner_index: int) -> None:
        # Collect the edge ages of the winners cluster
        cluster_nodes, cluster_edges = self._get_cluster(winner_index)
        cluster_ages = self._collect_cluster_ages(cluster_edges)
        if cluster_ages.size == 0:
            return
        pal_ages = np.array(self.adjacency_mat.data[winner_index])
        pals = np.array(self.adjacency_mat.rows[winner_index])
        c = np.percentile(cluster_ages, 75)
        th = self.param_edge * iqr(cluster_ages)
        
        outlierness = c + th
        ratio = self.edge_count_del / (self.edge_count_del + cluster_ages.size)
        del_threshold = self.edge_avg_age_del*ratio + outlierness*(1-ratio)

        # Check if there are any edges to be deleted
        indices_to_delete = pals[pal_ages > del_threshold]
        deleted_ages = pal_ages[pal_ages > del_threshold]
        for i in indices_to_delete:
            self._set_edge_age(i, winner_index, 0)

        # Update average lifetime of deleted edges
        if indices_to_delete.size > 0:
            self.edge_avg_age_del = (self.edge_count_del*self.edge_avg_age_del + np.sum(deleted_ages)) / (self.edge_count_del + indices_to_delete.size)
            self.edge_count_del += indices_to_delete.size

        return

    def _delete_nodes(self, indices: Sequence[int] | np.ndarray) -> None:
        indices = np.asarray(list(set(indices)))

        # Build keep-mask once and reuse for all structures
        mask = np.ones(len(self.nodes), dtype=bool)
        mask[indices] = False

        self.nodes = [n for i, n in enumerate(self.nodes) if mask[i]]
        self.winning_count = [t for i, t in enumerate(self.winning_count) if mask[i]]
        self.idle_since = [n for i, n in enumerate(self.idle_since) if mask[i]]
        self.labels = [l for i, l in enumerate(self.labels) if mask[i]]
        self.predictions = [p for i, p in enumerate(self.predictions) if mask[i]]

        # Sparse matrix: delete rows/columns via mask
        adj = self.adjacency_mat.tocsr()
        self.adjacency_mat = adj[mask, :][:, mask].tolil()

        if self.enable_tracking:
            self.track_input = np.delete(self.track_input, indices)
            self.track_input_idx = np.delete(self.track_input_idx, indices)

        self.node_count_del += len(indices)

    def _delete_noise_nodes_plus(self) -> None:
        degrees = self.adjacency_mat.getnnz(axis=1) # Number of connected edges per node
        noise_nodes = degrees < 1 # Nodes that are not connected to any other node
        active_nodes = ~noise_nodes
        if not np.any(active_nodes): # Do not delete if no active nodes are available.
            return

        win_nums_nparray = np.array(self.idle_since)
        idle_times = self.signal_num - win_nums_nparray
        winning_count = np.array(self.winning_count, dtype=float)
        unutility = idle_times / winning_count

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
            
            self._delete_nodes(np.where(inactive_idx & noise_nodes)[0])
            

