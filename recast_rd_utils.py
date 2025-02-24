import numpy as np
import networkx as nx
from scipy.stats import multivariate_normal
import pandas as pd
import scipy.linalg as slin
import scipy.optimize as sopt
import warnings 
import matplotlib.pyplot as plt
from data_gen_ts import generate_structure,sem_generator
from tqdm import tqdm
from metrics import shd, classification_metrics, threshold_metrics
from data_gen_reg import gen_stationary_dyn_net_and_df_regime
from dynotears import from_pandas_dynamic,from_numpy_dynamic,dynotears_perso


def tgraph_to_graph(tg):
    g = nx.DiGraph()
    og = nx.DiGraph()
    sg = nx.DiGraph()
    g.add_nodes_from(tg.nodes)
    og.add_nodes_from(tg.nodes)
    sg.add_nodes_from(tg.nodes)
    for cause, effects in tg.adj.items():
        for effect, _ in effects.items():
            if cause != effect:
                og.add_edges_from([(cause, effect)])
                g.add_edges_from([(cause, effect)])
            else:
                sg.add_edges_from([(cause, effect)])
                g.add_edges_from([(cause, effect)])
    return g, og, sg


def tgraph_to_list(tg):
    list_tg = []
    for cause, effects in tg.adj.items():
        for effect, eattr in effects.items():
            t_list = eattr['time']
            for t in t_list:
                list_tg.append((cause, effect, t))
    return list_tg


def print_graph(g):
    for cause, effects in g.adj.items():
        for effect, eattr in effects.items():
            print('(%s -> %s)' % (cause, effect))


def print_temporal_graph(tg):
    list_tg_hat = tgraph_to_list(tg)
    for edge in list_tg_hat:
        print('(%s -> %s with t= %d)' % (edge[0], edge[1], edge[2]))


def draw_graph(g, node_size=300):
    pos = nx.spring_layout(g, k=0.25, iterations=20)
    nx.draw(g, pos, with_labels=True, font_weight='bold', node_size=node_size)
    # nx.draw_shell(g, nlist=[range(4)], with_labels=True, font_weight='bold')
    plt.show()


def draw_temporal_graph(tg, node_size=300):
    pos = nx.spring_layout(tg, k=0.25, iterations=20)
    nx.draw(tg, pos, with_labels=True, font_weight='bold', node_size=node_size)
    edge_labels = nx.get_edge_attributes(tg, 'time')
    nx.draw_networkx_edge_labels(tg, pos, labels=edge_labels)
    plt.show()


def string_nodes(nodes):
    new_nodes = []
    for col in nodes:
        try:
            int(col)
            new_nodes.append("V" + str(col))
        except ValueError:
            new_nodes.append(col)
    return new_nodes

def three_col_format_to_graphs(nodes, three_col_format):
            tgtrue = nx.DiGraph()
            tgtrue.add_nodes_from(nodes)
            for i in range(three_col_format.shape[0]):
                        c = "V"+str(int(three_col_format[i, 0]))
                        e = "V"+str(int(three_col_format[i, 1]))
                        tgtrue.add_edges_from([(c, e)])
                        tgtrue.edges[c, e]['time'] = [int(three_col_format[i, 2])]

            gtrue, ogtrue, sgtrue = tgraph_to_graph(tgtrue)
            return gtrue, ogtrue, sgtrue, tgtrue



class GraphicalModel:
    def __init__(self, nodes):
        super(GraphicalModel, self).__init__()
        nodes = string_nodes(nodes)
        self.ghat = nx.DiGraph()
        self.ghat.add_nodes_from(nodes)
        self.oghat = nx.DiGraph()
        self.oghat.add_nodes_from(nodes)
        self.sghat = nx.DiGraph()
        self.sghat.add_nodes_from(nodes)

    def infer_from_data(self, data):
        raise NotImplementedError

    def _dataframe_to_graph(self, df):
        for name_x in df.columns:
            if df[name_x].loc[name_x] > 0:
                self.sghat.add_edges_from([(name_x, name_x)])
                self.ghat.add_edges_from([(name_x, name_x)])
            for name_y in df.columns:
                if name_x != name_y:
                    if df[name_y].loc[name_x] == 2:
                        self.oghat.add_edges_from([(name_x, name_y)])
                        self.ghat.add_edges_from([(name_x, name_y)])

    def draw(self, node_size=300):
        draw_graph(self.ghat, node_size=node_size)

    def print_graph(self):
        print_graph(self.ghat)

    def print_other_graph(self):
        print_graph(self.oghat)

    def print_self_graph(self):
        print_graph(self.sghat)

    def evaluation(self, gtrue, evaluation_measure="f1_oriented"):
        if evaluation_measure == "precision_adjacent":
            res = self._precision(gtrue, method="all_adjacent")
        elif evaluation_measure == "recall_adjacent":
            res = self._recall(gtrue, method="all_adjacent")
        elif evaluation_measure == "f1_adjacent":
            res = self._f1(gtrue, method="all_adjacent")
        elif evaluation_measure == "precision_oriented":
            res = self._precision(gtrue, method="all_oriented")
        elif evaluation_measure == "recall_oriented":
            res = self._recall(gtrue, method="all_oriented")
        elif evaluation_measure == "f1_oriented":
            res = self._f1(gtrue, method="all_oriented")
        elif evaluation_measure == "other_precision_adjacent":
            res = self._precision(gtrue, method="other_adjacent")
        elif evaluation_measure == "other_recall_adjacent":
            res = self._recall(gtrue, method="other_adjacent")
        elif evaluation_measure == "other_f1_adjacent":
            res = self._f1(gtrue, method="other_adjacent")
        elif evaluation_measure == "other_precision_oriented":
            res = self._precision(gtrue, method="other_oriented")
        elif evaluation_measure == "other_recall_oriented":
            res = self._recall(gtrue, method="other_oriented")
        elif evaluation_measure == "other_f1_oriented":
            res = self._f1(gtrue, method="other_oriented")
        elif evaluation_measure == "self_precision":
            res = self._precision(gtrue, method="self")
        elif evaluation_measure == "self_recall":
            res = self._recall(gtrue, method="self")
        elif evaluation_measure == "self_f1":
            res = self._f1(gtrue, method="self")
        else:
            raise AttributeError(evaluation_measure)
        return res

    def _hamming_distance(self, gtrue):
        # todo: check if it's correct (maybe it's not truely hamming distance)
        res = nx.graph_edit_distance(self.ghat, gtrue)
        return 1 - res/max(self.ghat.number_of_edges(), gtrue.number_of_edges())

    def _tp(self, gtrue, method="all_oriented"):  # oriented or adjacent
        if method == "all_oriented":
            tp = nx.intersection(gtrue, self.ghat)
        elif method == "all_adjacent":
            undirected_true = gtrue.to_undirected()
            undirected_hat = self.ghat.to_undirected()
            tp = nx.intersection(undirected_true, undirected_hat)
        elif method == "other_oriented":
            tp = nx.intersection(gtrue, self.oghat)
        elif method == "other_adjacent":
            undirected_true = gtrue.to_undirected()
            undirected_hat = self.oghat.to_undirected()
            tp = nx.intersection(undirected_true, undirected_hat)
        elif method == "self":
            tp = nx.intersection(gtrue, self.sghat)
        else:
            raise AttributeError(method)
        return len(tp.edges)

    def _fp(self, gtrue, method="all_oriented"):  # oriented or adjacent
        if method == "all_oriented":
            fp = nx.difference(self.ghat, gtrue)
        elif method == "all_adjacent":
            undirected_true = gtrue.to_undirected()
            undirected_hat = self.ghat.to_undirected()
            fp = nx.difference(undirected_hat, undirected_true)
        elif method == "other_oriented":
            fp = nx.difference(self.oghat, gtrue)
        elif method == "other_adjacent":
            undirected_true = gtrue.to_undirected()
            undirected_hat = self.oghat.to_undirected()
            fp = nx.difference(undirected_hat, undirected_true)
        elif method == "self":
            fp = nx.difference(self.sghat, gtrue)
        else:
            raise AttributeError(method)
        return len(fp.edges)

    def _fn(self, gtrue, method="all_oriented"):  # oriented or adjacent
        if method == "all_oriented":
            fn = nx.difference(gtrue, self.ghat)
        elif method == "all_adjacent":
            undirected_true = gtrue.to_undirected()
            undirected_hat = self.ghat.to_undirected()
            fn = nx.difference(undirected_true, undirected_hat)
        elif method == "other_oriented":
            fn = nx.difference(gtrue, self.oghat)
        elif method == "other_adjacent":
            undirected_true = gtrue.to_undirected()
            undirected_hat = self.oghat.to_undirected()
            fn = nx.difference(undirected_true, undirected_hat)
        elif method == "self":
            fn = nx.difference(gtrue, self.sghat)
        else:
            raise AttributeError(method)
        return len(fn.edges)

    def _topology(self, gtrue, method="all_oriented"):
        correct = self._tp(gtrue, method)
        added = self._fp(gtrue, method)
        missing = self._fn(gtrue, method)
        return correct/(correct + missing + added)

    def _false_positive_rate(self, gtrue, method="all_oriented"):
        true_pos = self._tp(gtrue, method)
        false_pos = self._fp(gtrue, method)
        if false_pos == 0:
            return 0
        else:
            return false_pos / (true_pos + false_pos)

    def _precision(self, gtrue, method="all_oriented"):
        true_pos = self._tp(gtrue, method)
        false_pos = self._fp(gtrue, method)
        if true_pos == 0:
            return 0
        else:
            return true_pos / (true_pos + false_pos)

    def _recall(self, gtrue, method="all_oriented"):
        true_pos = self._tp(gtrue, method)
        false_neg = self._fn(gtrue, method)
        if true_pos == 0:
            return 0
        else:
            return true_pos / (true_pos + false_neg)

    def _f1(self, gtrue, method="all_oriented"):
        p = self._precision(gtrue, method)
        r = self._recall(gtrue, method)
        if (p == 0) and (r == 0):
            return 0
        else:
            return 2 * p * r / (p + r)


class TemporalGraphicalModel(GraphicalModel):
    def __init__(self, nodes):
        GraphicalModel.__init__(self, nodes)
        nodes = string_nodes(nodes)
        self.tghat = nx.DiGraph()
        self.tghat.add_nodes_from(nodes)

    def infer_from_data(self, data):
        raise NotImplementedError

    def _dict_to_tgraph(self, temporal_dict):
        for name_y in temporal_dict.keys():
            for name_x, t_xy in temporal_dict[name_y]:
                if (name_x, name_y) in self.tghat.edges:
                    self.tghat.edges[name_x, name_y]['time'].append(-t_xy)
                else:
                    self.tghat.add_edges_from([(name_x, name_y)])
                    self.tghat.edges[name_x, name_y]['time'] = [-t_xy]
                # self.TGhat.add_weighted_edges_from([(name_x, name_y, t_xy)])

    def _tgraph_to_graph(self):
        self.ghat, self.oghat, self.sghat = tgraph_to_graph(self.tghat)

    def draw_temporal_graph(self, node_size=300):
        draw_temporal_graph(self.tghat, node_size=node_size)

    def print_temporal_graph(self):
        print_temporal_graph(self.tghat)

    def temporal_evaluation(self, tgtrue, evaluation_measure="f1"):
        if evaluation_measure == "precision":
            res = self._temporal_precision(tgtrue)
        elif evaluation_measure == "recall":
            res = self._temporal_recall(tgtrue)
        elif evaluation_measure == "f1":
            res = self._temporal_f1(tgtrue)
        else:
            raise AttributeError(evaluation_measure)
        return res

    def _temporal_tp(self, tgtrue):
        list_tg_true = tgraph_to_list(tgtrue)
        list_tg_hat = tgraph_to_list(self.tghat)
        tp = set(list_tg_true).intersection(list_tg_hat)
        return len(tp)

    def _temporal_fp(self, tgtrue):
        list_tg_true = tgraph_to_list(tgtrue)
        list_tg_hat = tgraph_to_list(self.tghat)
        fp = set(list_tg_hat).difference(list_tg_true)
        return len(fp)

    def _temporal_fn(self, tgtrue):
        list_tg_true = tgraph_to_list(tgtrue)
        list_tg_hat = tgraph_to_list(self.tghat)
        fn = set(list_tg_true).difference(list_tg_hat)
        return len(fn)

    def _temporal_false_positive_rate(self, tgtrue):
        true_pos = self._temporal_tp(tgtrue)
        false_pos = self._temporal_fp(tgtrue)
        return false_pos / (true_pos + false_pos)

    def _temporal_precision(self, tgtrue):
        true_pos = self._temporal_tp(tgtrue)
        false_pos = self._temporal_fp(tgtrue)
        if true_pos == 0:
            return 0
        else:
            return true_pos / (true_pos + false_pos)

    def _temporal_recall(self, tgtrue):
        true_pos = self._temporal_tp(tgtrue)
        false_neg = self._temporal_fn(tgtrue)
        if true_pos == 0:
            return 0
        else:
            return true_pos / (true_pos + false_neg)

    def _temporal_f1(self, tgtrue):
        p = self._temporal_precision(tgtrue)
        r = self._temporal_recall(tgtrue)
        if (p == 0) and (r == 0):
            return 0
        else:
            return 2 * p * r / (p + r)


def dynotears(data,X,Xlags, tau_max=5, alpha=0.0):
    graph_dict = dict()
    for name in data.columns:
        graph_dict[name] = []

    g,sm = dynotears_perso(data,X,Xlags, p=tau_max, w_threshold=0.01, lambda_w=0.05, lambda_a=0.05)
    #print(sm.edges)
    # print(sm.edges)
    # print(sm.pred)

    tname_to_name_dict = dict()
    count_lag = 0
    idx_name = 0
    for tname in sm.nodes:
        tname_to_name_dict[tname] = data.columns[idx_name]
        if count_lag == tau_max:
            idx_name = idx_name +1
            count_lag = -1
        count_lag = count_lag +1
    
    for ce in sm.edges:
        c = ce[0]
        e = ce[1]
        tc = int(c.partition("lag")[2])
        te = int(e.partition("lag")[2])
        t = tc - te
        if (tname_to_name_dict[c], -t) not in graph_dict[tname_to_name_dict[e]]:
            graph_dict[tname_to_name_dict[e]].append((tname_to_name_dict[c], -t))
    
    # g = sm.to_directed()
    return graph_dict,nx.to_numpy_array(g)

class Dynotears(TemporalGraphicalModel):
            def __init__(self, nodes, sig_level=0.05, nlags=5):
                        TemporalGraphicalModel.__init__(self, nodes)
                        self.sig_level = sig_level
                        self.nlags = nlags

            def infer_from_data(self, data,X,Xlags,):
                        data.columns = list(self.ghat.nodes)
                        tg_dict,g = dynotears(data,X,Xlags, tau_max=self.nlags, alpha=self.sig_level)
                        self._dict_to_tgraph(tg_dict)
                        self._tgraph_to_graph()
                        return g