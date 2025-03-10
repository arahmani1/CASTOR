import numpy as np
import networkx as nx
from dynotears import dynotears_perso

def dynotears(data,X,Xlags, tau_max=5,w_threshold=0.001):
    
    g,sm = dynotears_perso(data,X,Xlags, p=tau_max, w_threshold = w_threshold, lambda_w=0.05, lambda_a=0.05)
    #print(sm.edges)
    # print(sm.edges)
    # print(sm.pred)

    return nx.to_numpy_array(g)


class CASTOR_linear_1_regime:
            def __init__(self, nlags=5):
                        self.nlags = nlags

            def infer_from_data(self, data,X,Xlags,w_threshold=0.001):
                        
                        g = dynotears(data,X,Xlags, tau_max=self.nlags,w_threshold=w_threshold)
                        return g