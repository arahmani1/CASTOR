import warnings
from typing import Callable, List
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.gaussian_process.kernels import Kernel
from utils import identity
from causal_structure import StructureModel




def simulate_dag(
    num_nodes: int,
    degree: float,
    graph_type: str = "erdos-renyi",
):
    """
    Simulate random DAG with some expected degree.
    """

    if num_nodes < 2:
        raise ValueError("DAG must have at least 2 nodes")

    if graph_type == "erdos-renyi":
        p_threshold = float(degree) / (num_nodes - 1)
        p_edge = (np.random.rand(num_nodes, num_nodes) < p_threshold).astype(float)
        edge_flags = np.tril(p_edge, k=-1)

    elif graph_type == "barabasi-albert":
        m = int(round(degree / 2))
        edge_flags = np.zeros([num_nodes, num_nodes])
        bag = [0]
        for i in range(1, num_nodes):
            dest = np.random.choice(bag, size=m)
            for j in dest:
                edge_flags[i, j] = 1
            bag.append(i)
            bag.extend(dest)

    elif graph_type == "full":  # ignore degree
        edge_flags = np.tril(np.ones([num_nodes, num_nodes]), k=-1)

    else:
        raise ValueError(
            f"Unknown graph type {graph_type}. "
            "Available types are ['erdos-renyi', 'barabasi-albert', 'full']"
        )

    perms = np.random.permutation(np.eye(num_nodes, num_nodes))
    edge_flags = perms.T.dot(edge_flags).dot(perms)

    adj_matrix = edge_flags
    graph = StructureModel(adj_matrix)
    return graph

def simulate_lag_graph(
    num_nodes: int,
    p: int,
    degree: float,
    graph_type: str,
):
    """Simulate time lag graph.
    """
    if graph_type == "erdos-renyi":
        prob = degree / num_nodes
        b = (np.random.rand(p * num_nodes, num_nodes) < prob).astype(float)
    elif graph_type == "full":  # ignore degree, only for experimental use
        b = np.ones([p * num_nodes, num_nodes])
    else:
        raise ValueError(
            f"Unknown inter-slice graph type `{graph_type}`. "
            "Valid types are 'erdos-renyi' and 'full'"
        )
    
    a = b

    df = pd.DataFrame(
        a,
        index=[
            f"{var}_lag{l_val}" for l_val in range(1, p + 1) for var in range(num_nodes)
        ],
        columns=[f"{var}_lag0" for var in range(num_nodes)],
    )
    idxs, cols = list(df.index), list(df.columns)
    for i in idxs:
        df[i] = 0
    for i in cols:
        df.loc[i, :] = 0

    g_inter = StructureModel(df)
    return g_inter


def generate_structure_dynamic( 
    num_nodes: int,
    p: int,
    degree_intra: float,
    degree_inter: float,
    graph_type_intra: str = "erdos-renyi",
    graph_type_inter: str = "erdos-renyi",
):
    """
    Generates a Temporal DAG at random.
    """
    sm_intra = simulate_dag(
        num_nodes=num_nodes,
        degree=degree_intra,
        graph_type=graph_type_intra,
    )
    sm_inter = simulate_lag_graph(
        num_nodes=num_nodes,
        p=p,
        degree=degree_inter,
        graph_type=graph_type_inter,
    )
    res = StructureModel()
    res.add_nodes_from(sm_inter.nodes)
    res.add_nodes_from([f"{u}_lag0" for u in sm_intra.nodes])
    res.add_weighted_edges_from(sm_inter.edges.data("weight"))
    res.add_weighted_edges_from(
        [(f"{u}_lag0", f"{v}_lag0", w) for u, v, w in sm_intra.edges.data("weight")]
    )
    return res

def simulate_SEM( 
    g: StructureModel,
    func: Callable[[np.ndarray], np.ndarray],
    first_start_lag: np.ndarray,
    n_samples: int = 1000,
    burn_in: int = 100,
    sem_type: str = "linear-gauss",
    noise_scale: float = 1.0,
    ANM_type: str = "Linear",
):
    """
    Simulate samples from dynamic SEM with specified type of noise.
    """
    s_types = ("linear-gauss", "linear-exp")
    ANM_types = ("Linear", "Non-Linear")
    if sem_type not in s_types:
        raise ValueError(f"unknown sem type {sem_type}. Available types are: {s_types}")
    if ANM_type not in ANM_types:
        raise ValueError(f"unknown causal relationship {sem_type}. Available types are: {s_types}")           
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
    if ANM_type == "Linear":           
        for t in range(total_length):
               for j in ordered_vertices:
                              parents = list(g_intra.predecessors(j))
                              parents_prev = list(g_inter.predecessors(j + p * d))
                              X[t, j] = (
                                             
                                             + X[t, parents].dot(w_mat[parents, j])
                                             + Xlags[t, parents_prev].dot(a_mat[parents_prev, j])
                              )
                              if sem_type == "linear-gauss":
                                             X[t, j] = X[t, j] + np.random.normal(scale=noise_scale)
                              elif sem_type == "linear-exp":
                                             X[t, j] = X[t, j] + np.random.exponential(scale=noise_scale)

               if (t + 1) < total_length:
                              Xlags[t + 1, :] = np.concatenate([X[t, :], Xlags[t, :]])[: d * p]
    elif ANM_type == "Non-Linear": 
               lamda_l = np.random.uniform(0.5, 2, d)
               for t in range(total_length):
                              for j in ordered_vertices:
                                             lamda = lamda_l[j]
                                             if (np.sum(w_mat[:, j]) != 0 or np.sum(a_mat[:, j]) != 0 ):
                                                            inst_parents = X[t, :].dot(w_mat[:, j])
                                                            lag_parents = Xlags[t, :].dot(a_mat[:, j])
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
    
def simulate_stationary_regime( 
    func: Callable[[np.ndarray], np.ndarray],
    first_lag: np.ndarray,                             
    num_nodes: int = 10,
    n_samples: int = 100,
    p: int = 1,
    degree_intra: float = 3,
    degree_inter: float = 3,
    graph_type_intra: str = "erdos-renyi",
    graph_type_inter: str = "erdos-renyi",
    sem_type: str = "linear-gauss",
    noise_scale: float = 1,
    ANM_type: str = "Linear",
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
                    simulate_SEM(
                        g,
                        func,
                        first_lag,
                        n_samples=n_samples + burn_in,
                        sem_type=sem_type,
                        noise_scale=noise_scale,
                        ANM_type=ANM_type,
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
    

def simulate_regime_MTS(
    regime:int,
    func_l: List,
    num_nodes: int = 10,
    n_samples: List = [100],
    p: int = 1,
    degree_intra: float = 3,
    degree_inter: float = 3,
    graph_type_intra: str = "erdos-renyi",
    graph_type_inter: str = "erdos-renyi",
    sem_type: str = "linear-gauss",
    noise_scale: float = 1,
    ANM_type: str = "Linear",
    max_data_gen_trials: int = 1000,
):
    g_list = []
    first_lag = np.zeros(p*num_nodes)
    if ANM_type == "Linear":
                   func_l = [identity for _ in range(regime)]           
    
    for i in range(regime):
        g, df, intra_nodes, inter_nodes = simulate_stationary_regime(func_l[i],first_lag, num_nodes, n_samples[i],
                                                                           p, degree_intra,degree_inter,
                                                                           graph_type_intra,graph_type_inter,
                                                                           sem_type, noise_scale, ANM_type,
                                                                           max_data_gen_trials)
        g_list.append(g)
        first_lag = df.to_numpy()[0,:p*num_nodes]
        if i == 0:
            df_total = df
        else:
            df_total = pd.concat([df_total,df],ignore_index=True)
    return g_list,df_total, intra_nodes, inter_nodes