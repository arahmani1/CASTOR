import warnings
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.gaussian_process.kernels import Kernel

from causal_structure import StructureModel
from data_gen_dyn_ts import (
    generate_structure_dynamic,
)




def generate_dataframe_dynamic_regime(  # pylint: disable=R0914
    g,
    first_start_lag,
    n_samples: int = 1000,
    burn_in: int = 100,
    sem_type: str = "linear-gauss",
    noise_scale: float = 1.0,
    drift: np.ndarray = None,
):
    """Simulate samples from dynamic SEM with specified type of noise.
    Args:
        g: Dynamic DAG
        n_samples: number of samples
        burn_in: number of samples to discard
        sem_type: {linear-gauss,linear-exp,linear-gumbel}
        noise_scale: scale parameter of noise distribution in linear SEM
        drift: array of drift terms for each node, if None then the drift is 0
    Returns:
        X: [n,d] sample matrix, row t is X_t
        Y: [n,d*p] sample matrix, row t is [X_{t-1}, ..., X_{t-p}]
    Raises:
        ValueError: if sem_type isn't linear-gauss/linear_exp/linear-gumbel
    """
    s_types = ("linear-gauss", "linear-exp", "linear-gumbel")
    if sem_type not in s_types:
        raise ValueError(f"unknown sem type {sem_type}. Available types are: {s_types}")
    intra_nodes = sorted(el for el in g.nodes if "_lag0" in el)
    inter_nodes = sorted(el for el in g.nodes if "_lag0" not in el)
    w_mat = nx.to_numpy_array(g, nodelist=intra_nodes)
    a_mat = nx.to_numpy_array(g, nodelist=intra_nodes + inter_nodes)[
        len(intra_nodes) :, : len(intra_nodes)
    ]
    g_intra = nx.DiGraph(w_mat)
    g_inter = nx.bipartite.from_biadjacency_matrix(
        csr_matrix(a_mat), create_using=nx.DiGraph
    )
    d = w_mat.shape[0]
    p = a_mat.shape[0] // d
    total_length = n_samples + burn_in
    X = np.zeros([total_length, d])
    Xlags = np.zeros([total_length, p * d])
    
    Xlags[0,:] = first_start_lag
    ordered_vertices = list(nx.topological_sort(g_intra))
    if drift is None:
        drift = np.zeros(d)
    for t in range(total_length):
        for j in ordered_vertices:
            parents = list(g_intra.predecessors(j))
            parents_prev = list(g_inter.predecessors(j + p * d))
            X[t, j] = (
                drift[j]
                + X[t, parents].dot(w_mat[parents, j])
                + Xlags[t, parents_prev].dot(a_mat[parents_prev, j])
            )
            if sem_type == "linear-gauss":
                X[t, j] = X[t, j] + np.random.normal(scale=noise_scale)
            elif sem_type == "linear-exp":
                X[t, j] = X[t, j] + np.random.exponential(scale=noise_scale)
            elif sem_type == "linear-gumbel":
                X[t, j] = X[t, j] + np.random.gumbel(scale=noise_scale)

        if (t + 1) < total_length:
            Xlags[t + 1, :] = np.concatenate([X[t, :], Xlags[t, :]])[: d * p]
    return pd.concat(
        [
            pd.DataFrame(X[-n_samples:], columns=intra_nodes),
            pd.DataFrame(Xlags[-n_samples:], columns=inter_nodes),
        ],
        axis=1,
    )



def gen_stationary_dyn_net_and_df_withlag(  # pylint: disable=R0913, R0914
    first_lag,                             
    num_nodes: int = 10,
    n_samples: int = 100,
    p: int = 1,
    degree_intra: float = 3,
    degree_inter: float = 3,
    graph_type_intra: str = "erdos-renyi",
    graph_type_inter: str = "erdos-renyi",
    w_min_intra: float = 0.5,
    w_max_intra: float = 0.5,
    w_min_inter: float = 0.5,
    w_max_inter: float = 0.5,
    w_decay: float = 1.0,
    sem_type: str = "linear-gauss",
    noise_scale: float = 1,
    max_data_gen_trials: int = 1000,
):
    """
    Generates a dynamic structure model as well a dataframe representing a time series realisation of that model.
    We do checks to verify the network is stationary, and iterate until the resulting network is stationary.
    Args:
        num_nodes: number of nodes in the intra-slice structure
        n_samples: number of points to sample from the model, as a time series
        p: lag value for the dynamic structure
        degree_intra: expected degree for intra_slice nodes
        degree_inter: expected degree for inter_slice nodes
        graph_type_intra:
            - erdos-renyi: constructs a graph such that the probability of any given edge is degree / (num_nodes - 1)
            - barabasi-albert: constructs a scale-free graph from an initial connected graph of (degree / 2) nodes
            - full: constructs a fully-connected graph - degree has no effect
        graph_type_inter:
            - erdos-renyi: constructs a graph such that the probability of any given edge is degree / (num_nodes - 1)
            - full: connect all past nodes to all present nodesw_min_intra:
        w_min_intra: minimum weight on intra-slice adjacency matrix
        w_max_intra: maximum weight on intra-slice adjacency matrix
        w_min_inter: minimum weight on inter-slice adjacency matrix
        w_max_inter: maximum weight on inter-slice adjacency matrix
        w_decay: exponent of weights decay for slices that are farther apart. Default is 1.0, which implies no decay
        sem_type: {linear-gauss,linear-exp,linear-gumbel}
        noise_scale: scale parameter of noise distribution in linear SEM
        max_data_gen_trials: maximun number of attempts until obtaining a seemingly stationary model
    Returns:
        Tuple with:
        - the model created,as a Structure model
        - DataFrame representing the time series created from the model
        - Intra-slice nodes names
        - Inter-slice nodes names
    """

    with np.errstate(over="raise", invalid="raise"):
        burn_in = max(n_samples // 10, 50)

        simulate_flag = True
        g, intra_nodes, inter_nodes = None, None, None

        while simulate_flag:
            max_data_gen_trials -= 1
            if max_data_gen_trials <= 0:
                simulate_flag = False

            try:
                simulate_graphs_flag = True
                while simulate_graphs_flag:

                    g = generate_structure_dynamic(
                        num_nodes=num_nodes,
                        p=p,
                        degree_intra=degree_intra,
                        degree_inter=degree_inter,
                        graph_type_intra=graph_type_intra,
                        graph_type_inter=graph_type_inter,
                        w_min_intra=w_min_intra,
                        w_max_intra=w_max_intra,
                        w_min_inter=w_min_inter,
                        w_max_inter=w_max_inter,
                        w_decay=w_decay,
                    )
                    intra_nodes = sorted([el for el in g.nodes if "_lag0" in el])
                    inter_nodes = sorted([el for el in g.nodes if "_lag0" not in el])
                    # Exclude empty graphs from consideration unless input degree is 0
                    if (
                        (
                            [(u, v) for u, v in g.edges if u in intra_nodes]
                            and [(u, v) for u, v in g.edges if u in inter_nodes]
                        )
                        or degree_intra == 0
                        or degree_inter == 0
                    ):
                        simulate_graphs_flag = False

                # generate single time series
                df = (
                    generate_dataframe_dynamic_regime(
                        g,
                        first_lag,
                        n_samples=n_samples + burn_in,
                        sem_type=sem_type,
                        noise_scale=noise_scale,
                    )
                    .loc[burn_in:, intra_nodes + inter_nodes]
                    .reset_index(drop=True)
                )

                if df.isna().any(axis=None):
                    continue
            except (OverflowError, FloatingPointError):
                continue
            if (df.abs().max().max() < 1e3) or (max_data_gen_trials <= 0):
                simulate_flag = False
                            
        if max_data_gen_trials <= 0:
            warnings.warn(
                "Could not simulate data, returning constant dataframe", UserWarning
            )

            df = pd.DataFrame(
                np.ones((n_samples, num_nodes * (1 + p))),
                columns=intra_nodes + inter_nodes,
            )
        
    return g, df, intra_nodes, inter_nodes

def gen_stationary_dyn_net_and_df_regime(
    regime,
    num_nodes: int = 10,
    n_samples: List = [100],
    p: int = 1,
    degree_intra: float = 3,
    degree_inter: float = 3,
    graph_type_intra: str = "erdos-renyi",
    graph_type_inter: str = "erdos-renyi",
    w_min_intra: float = 0.5,
    w_max_intra: float = 0.5,
    w_min_inter: float = 0.5,
    w_max_inter: float = 0.5,
    w_decay: float = 1.0,
    sem_type: str = "linear-gauss",
    noise_scale: float = 1,
    max_data_gen_trials: int = 1000,
):
    g_list = []
    first_lag = np.zeros(p*num_nodes)
    for i in range(regime):
        g, df, intra_nodes, inter_nodes = gen_stationary_dyn_net_and_df_withlag(first_lag, num_nodes, n_samples[i],
                                                                                     p, degree_intra,degree_inter,
                                                                                     graph_type_intra,graph_type_inter,
                                                                                     w_min_intra, w_max_intra, w_min_inter,
                                                                                     w_max_inter, w_decay, sem_type, noise_scale,
                                                                                     max_data_gen_trials)
        g_list.append(g)
        first_lag = df.to_numpy()[0,:p*num_nodes]
        if i == 0:
            df_total = df
        else:
            df_total = pd.concat([df_total,df],ignore_index=True)
    return g_list,df_total, intra_nodes, inter_nodes