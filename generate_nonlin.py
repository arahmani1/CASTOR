import random
import numpy as np
import pandas as pd
from scipy.stats import bernoulli
import warnings
from typing import List, Optional, Tuple
import networkx as nx
import numpy as np
import pandas as pd
from data_gen_dyn_ts import generate_structure_dynamic
from causal_structure import StructureModel




def generate_ANM_dynamic(  # pylint: disable=R0914
    g: StructureModel,
    func,
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
    
    d = w_mat.shape[0]
    p = a_mat.shape[0] // d
    total_length = n_samples + burn_in
    X = np.zeros([total_length, d])
    Xlags = np.zeros([total_length, p * d])
    Xlags[0,:] = first_start_lag
    ordered_vertices = list(nx.topological_sort(g_intra))
   
    lamda_l = np.random.uniform(0.5, 2, d)
    for t in range(total_length):
        for j in ordered_vertices:
            lamda = lamda_l[j]
            if (np.sum(w_mat[:, j]) != 0 or np.sum(a_mat[:, j]) != 0 ):
                inst_parents = X[t, :].dot(w_mat[:, j])
                lag_parents = Xlags[t, :].dot(a_mat[:, j])
                #input = inst_parents+lag_parents# np.hstack((inst_parents, lag_parents))
                X[t, j] = lamda*func((inst_parents+lag_parents))
                
            if sem_type == "linear-gauss":
                X[t, j] = X[t, j] + np.random.normal(scale=noise_scale)
        if (t + 1) < total_length:
            Xlags[t + 1, :] = np.concatenate([X[t, :], Xlags[t, :]])[: d * p]
    
    return pd.concat(
        [
            pd.DataFrame(X[-n_samples:], columns=intra_nodes),
            pd.DataFrame(Xlags[-n_samples:], columns=inter_nodes),
        ],
        axis=1,
    )
    
    
    
def gen_stationary_dyn_ANM(  # pylint: disable=R0913, R0914
    first_start_lag,
    func,
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
                    generate_ANM_dynamic(
                        g,
                        func,
                        first_start_lag,
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
    func_l,
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
    noise_scale: list = [1],
    max_data_gen_trials: int = 1000,
):
    g_list = []
    first_lag = np.zeros(p*num_nodes)
    for i in range(regime):
        g, df, intra_nodes, inter_nodes = gen_stationary_dyn_ANM(first_lag, func_l[i],num_nodes, n_samples[i],
                                                                                     p, degree_intra,degree_inter,
                                                                                     graph_type_intra,graph_type_inter,
                                                                                     w_min_intra, w_max_intra, w_min_inter,
                                                                                     w_max_inter, w_decay, sem_type, noise_scale[i],
                                                                                     max_data_gen_trials)
        g_list.append(g)
        first_lag = df.to_numpy()[0,:p*num_nodes]
        if i == 0:
            df_total = df
        else:
            df_total = pd.concat([df_total,df],ignore_index=True)
    return g_list,df_total, intra_nodes, inter_nodes