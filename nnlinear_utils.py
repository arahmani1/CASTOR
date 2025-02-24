import random
import numpy as np
import pandas as pd
from scipy.stats import bernoulli
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.gaussian_process import GaussianProcessRegressor
import torch as th
import copy
import warnings
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.gaussian_process.kernels import Kernel
import torch.nn.functional as F
from data_gen_dyn_ts import generate_structure_dynamic
from causal_structure import StructureModel
import statsmodels.api as sm
import math
import logging
import torch
import torch.nn as nn 
import numpy as np
import pandas as pd
from collections.abc import Iterable
from pandas import Index, RangeIndex

class LocallyConnected(nn.Module):
    def __init__(self, num_linear, input_features, output_features, bias=True):
        super(LocallyConnected, self).__init__()
        self.num_linear = num_linear
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(num_linear,
                                                input_features,
                                                output_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_linear, output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        k = 1.0 / self.input_features
        bound = math.sqrt(k)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input_x: torch.Tensor):
        # [n, d, 1, m2] = [n, d, 1, m1] @ [1, d, m1, m2]
        out = torch.matmul(input_x.unsqueeze(dim=2), self.weight.unsqueeze(dim=0))
        out = out.squeeze(dim=2)
        if self.bias is not None:
            # [n, d, m2] += [d, m2]
            out += self.bias
        return out

    def extra_repr(self):
        """
        (Optional)Set the extra information about this module. You can test
        it by printing an object of this class.

        Returns
        -------

        """

        return 'num_linear={}, in_features={}, out_features={}, bias={}'.format(
            self.num_linear, self.in_features, self.out_features,
            self.bias is not None
        )
        


class MLPModel_lag(nn.Module):
    def __init__(self, dims, p,bias=True, device=None):
        
        super(MLPModel_lag, self).__init__()
        if len(dims) < 2:
            raise ValueError(f"The size of dims at least greater equal to 2, contains one "
                             f"one hidden layer and one output_layer")
        if dims[-1] != 1:
            raise ValueError(f"The dimension of output layer must be 1, but got {dims[-1]}.")
        d = dims[0]
        self.dims = dims
        self.device = device
        # fc1: variable splitting for l1
        self.fc1_pos = nn.Linear(d, d * dims[1], bias=bias, device=self.device)
        self.fc1_neg = nn.Linear(d, d * dims[1], bias=bias, device=self.device)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        # fc1: variable splitting for l1
        self.fc1_pos_lag = nn.Linear(p*d, d * dims[1], bias=bias, device=self.device)
        self.fc1_neg_lag = nn.Linear(p*d, d * dims[1], bias=bias, device=self.device)
        self.fc1_pos_lag.weight.bounds = self._bounds()
        self.fc1_neg_lag.weight.bounds = self._bounds()
        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            if l ==0:
                     layers.append(LocallyConnected(d, 2*dims[l + 1], dims[l + 2], bias=bias))  
            else:     
                layers.append(LocallyConnected(d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers).to(device=self.device)


    def _bounds(self):
        d = self.dims[0]
        bounds = []
        for j in range(d):
            for m in range(self.dims[1]):
                for i in range(d):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds

    def forward(self, x,y):  # [n, d] -> [n, d]
        x = self.fc1_pos(x) - self.fc1_neg(x)  # [n, d * m1]
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        y = self.fc1_pos_lag(y) - self.fc1_neg_lag(y)  # [n, d * m1]
        y = y.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        x = torch.concat((x , y), axis =2)
        for fc in self.fc2:
            x = torch.sigmoid(x)  # [n, d, m1]
            x = fc(x)  # [n, d, m2]
        x = x.squeeze(dim=2)  # [n, d]
        return x

    def h_func(self):
        """
        Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG.

        Returns
        -------

        """
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        # h = trace_expm(A) - d  # (Zheng et al. 2018)
        init_e = torch.eye(d).to(self.device)
        M = init_e + A / d  # (Yu et al. 2019)
        E = torch.matrix_power(M, d - 1)
        h = (E.t() * M).sum() - d
        return h

    def l2_reg(self):
        """
        Take 2-norm-squared of all parameters.

        Returns
        -------

        """
        reg = 0.
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight_lag = self.fc1_pos_lag.weight - self.fc1_neg_lag.weight
        reg += torch.sum(fc1_weight ** 2)+ torch.sum(fc1_weight_lag ** 2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight ** 2)
        return reg

    def fc1_l1_reg(self):
        """
        Take l1 norm of fc1 weight.

        Returns
        -------

        """
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        reg_lag = torch.sum(self.fc1_pos_lag.weight + self.fc1_neg_lag.weight)
        return reg + reg_lag

    @torch.no_grad()
    def fc1_to_adj(self):  
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        W = torch.sqrt(A)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W
    @torch.no_grad()
    def fc1_lag_to_adj(self):
        """
        Get W from fc1 weights, take 2-norm over m1 dim.

        Returns
        -------

        """
        d = self.dims[0]
        fc1_weight = self.fc1_pos_lag.weight - self.fc1_neg_lag.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        W = torch.sqrt(A)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W


def squared_loss(output, target):
    """
    Least squares loss function.

    Parameters
    ----------
    output: torch.tenser
        network output
    target: torch.tenser
        raw input
    Returns
    -------
    : torch.tenser
        loss value
    """
    n = target.shape[0]
    loss = 0.5 / n * torch.sum((output - target) ** 2)
    return loss


import abc



class BaseLearner(metaclass=abc.ABCMeta):

    def __init__(self):

        self._causal_matrix = None

    @abc.abstractmethod
    def learn(self, data, *args, **kwargs):

        raise NotImplementedError

    @property
    def causal_matrix(self):
        return self._causal_matrix

    @causal_matrix.setter
    def causal_matrix(self, value):
        self._causal_matrix = value

class Tensor(np.ndarray):

    def __new__(cls, object=None, index=None, columns=None):

        if object is None:
            raise TypeError("Tensor() missing required argument 'object' (pos 0)")
        elif isinstance(object, list):
            object = np.array(object)
        elif isinstance(object, pd.DataFrame):
            index = object.index
            columns = object.columns
            object = object.values
        elif isinstance(object, (np.ndarray, cls)):
            pass
        else:
            raise TypeError(
                "Type of the required argument 'object' must be array-like."
            )
        if index is None:
            index = range(object.shape[0])
        if columns is None:
            columns = range(object.shape[1])
        obj = np.asarray(object).view(cls)
        obj.index = index
        obj.columns = columns

        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        if self.ndim == 0: return
        elif self.ndim == 1:
            self.columns = RangeIndex(0, 1, step=1, dtype=int)
        else:
            self.columns = RangeIndex(0, self.shape[1], step=1, dtype=int)
        self.index = RangeIndex(0, self.shape[0], step=1, dtype=int)

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        assert isinstance(value, Iterable)
        if len(list(value)) != self.shape[0]:
            raise ValueError("Size of value is not equal to the shape[0].")
        self._index = Index(value)

    @property
    def columns(self):
        return self._columns

    @columns.setter
    def columns(self, value):
        assert isinstance(value, Iterable)
        if (self.ndim > 1 and len(list(value)) != self.shape[1]):
            raise ValueError("Size of value is not equal to the shape[1].")
        self._columns = Index(value)
        

import scipy.optimize as sopt


class LBFGSBScipy(torch.optim.Optimizer):
    """
    Wrap L-BFGS-B algorithm, using scipy routines.
    
    Courtesy: Arthur Mensch's gist
    https://gist.github.com/arthurmensch/c55ac413868550f89225a0b9212aa4cd
    """

    def __init__(self, params):
        defaults = dict()
        super(LBFGSBScipy, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("LBFGSBScipy doesn't support per-parameter options"
                             " (parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel = sum([p.numel() for p in self._params])

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _gather_flat_bounds(self):
        bounds = []
        for p in self._params:
            if hasattr(p, 'bounds'):
                b = p.bounds
            else:
                b = [(None, None)] * p.numel()
            bounds += b
        return bounds

    def _gather_flat_params(self):
        views = []
        for p in self._params:
            if p.data.is_sparse:
                view = p.data.to_dense().view(-1)
            else:
                view = p.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _distribute_flat_params(self, params):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.data = params[offset:offset + numel].view_as(p.data)
            offset += numel
        assert offset == self._numel

    def step(self, closure, device):
        """
        Performs a single optimization step.

        Parameters
        ----------
        closure: callable
            A closure that reevaluates the model and returns the loss.
        device: option, default: None
            torch.device('cpu') or torch.device('cuda').

        """

        assert len(self.param_groups) == 1

        def wrapped_closure(flat_params):
            """closure must call zero_grad() and backward()"""
            flat_params = torch.from_numpy(flat_params)
            flat_params = flat_params.to(torch.get_default_dtype()).to(device)
            self._distribute_flat_params(flat_params)
            loss = closure()
            loss = loss.item()
            flat_grad = self._gather_flat_grad().cpu().detach().numpy()
            return loss, flat_grad.astype('float64')

        initial_params = self._gather_flat_params()
        initial_params = initial_params.cpu().detach().numpy()

        bounds = self._gather_flat_bounds()

        # Magic
        sol = sopt.minimize(wrapped_closure,
                            initial_params,
                            method='L-BFGS-B',
                            jac=True,
                            bounds=bounds)

        final_params = torch.from_numpy(sol.x)
        final_params = final_params.to(torch.get_default_dtype())
        self._distribute_flat_params(final_params)
        

class NotearsNonlinear_lag(BaseLearner):
    def __init__(self, 
                 lambda1: float = 0.01,
                 lambda2: float = 0.01,
                 max_iter: int = 200,
                 h_tol: float = 1e-8,
                 rho_max: float = 1e+16,
                 w_threshold: float = 0.3,
                 A_threshold: float = 0.3,
                 hidden_layers: tuple = (10, 1),
                 expansions: int = 10,
                 bias: bool = True,
                 model_type: str = "mlp",
                 device_type: str = "cpu",
                 device_ids=None):

        super().__init__()
        self.A_lag = torch.zeros((10,10))
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.w_threshold = w_threshold
        self.A_threshold = A_threshold
        self.hidden_layers = hidden_layers
        self.expansions = expansions
        self.bias = bias
        self.model_type = model_type
        self.device_type = device_type
        self.device_ids = device_ids
        self.rho, self.alpha, self.h = 1.0, 0.0, np.inf
        self.W_no_thres = torch.zeros((10,10))
        self.A_no_thres = torch.zeros((10,10))
        
        if torch.cuda.is_available():
            logging.info('GPU is available.')
        else:
            logging.info('GPU is unavailable.')
            if self.device_type == 'gpu':
                raise ValueError("GPU is unavailable, "
                                 "please set device_type = 'cpu'.")
        if self.device_type == 'gpu':
          
            device = torch.device('cuda:1')
        else:
            device = torch.device('cpu')
        self.device = device

    def learn(self, data, columns=None, **kwargs):
        """
        Set up and run the NotearsNonlinear algorithm.

        Parameters
        ----------
        data: castle.Tensor or numpy.ndarray
            The castle.Tensor or numpy.ndarray format data you want to learn.
        columns : Index or array-like
            Column labels to use for resulting tensor. Will default to
            RangeIndex (0, 1, 2, ..., n) if no column labels are provided.
        """
        X = Tensor(data, columns=columns)

        input_dim = int(X.shape[1]//2)
        model = self.get_model(input_dim)
        if model:
            W_est,A, modell = self.notears_nonlinear(model, X)
            self.W_no_thres = W_est
            self.A_no_thres = A
            causal_matrix = (abs(W_est) > self.w_threshold).astype(int)
            A_matrix = (abs(A) > self.A_threshold).astype(int)
            self.weight_causal_matrix = Tensor(W_est,
                                               index=X[:,:input_dim].columns,
                                               columns=X[:,:input_dim].columns)
            self.causal_matrix = Tensor(causal_matrix, index=X[:,:input_dim].columns, columns=X[:,:input_dim].columns)
            
            self.A_lag = A_matrix
            self.modell = modell

    def dual_ascent_step(self, model, X,Y):
        """
        Perform one step of dual ascent in augmented Lagrangian.

        Parameters
        ----------
        model: nn.Module
            network model
        X: torch.tenser
            sample data

        Returns
        -------
        :tuple
            cycle control parameter
        """
        h_new = None
        optimizer = LBFGSBScipy(model.parameters())
        X_torch = torch.from_numpy(X)
        Y_torch = torch.from_numpy(Y)
        while self.rho < self.rho_max:
            X_torch = X_torch.to(self.device)
            Y_torch = Y_torch.to(self.device)
            def closure():
                optimizer.zero_grad()
                X_hat = model(X_torch.float(),Y_torch.float())
                loss = squared_loss(X_hat, X_torch)
                h_val = model.h_func()
                penalty = 0.5 * self.rho * h_val * h_val + self.alpha * h_val
                l2_reg = 0.5 * self.lambda2 * model.l2_reg()
                l1_reg = self.lambda1 * model.fc1_l1_reg()
                primal_obj = loss + penalty + l2_reg + l1_reg
                primal_obj.backward()
                return primal_obj

            optimizer.step(closure, self.device)  # NOTE: updates model in-place
            with torch.no_grad():
                model = model.to(self.device)
                h_new = model.h_func().item()
            if h_new > 0.25 * self.h:
                self.rho *= 10
            else:
                break
        self.alpha += self.rho * h_new
        self.h = h_new

    def notears_nonlinear(self,
                          model: nn.Module,
                          X: np.ndarray):
        """
        notaears frame entrance.

        Parameters
        ----------
        model: nn.Module
            network model
        X: castle.Tensor or numpy.ndarray
            sample data

        Returns
        -------
        :tuple
            Prediction Graph Matrix Coefficients.
        """
        logging.info('[start]: n={}, d={}, iter_={}, h_={}, rho_={}'.format(
            X.shape[0], X.shape[1], self.max_iter, self.h_tol, self.rho_max))
        dim = int(X.shape[1]//2)
        for _ in range(self.max_iter):
            self.dual_ascent_step(model, X[:,:dim], X[:,dim:2*dim])

            logging.debug('[iter {}] h={:.3e}, rho={:.1e}'.format(_, self.h, self.rho))

            if self.h <= self.h_tol or self.rho >= self.rho_max:
                break
        W_est = model.fc1_to_adj()
        A_lag = model.fc1_lag_to_adj()
        

        return W_est,A_lag, model

    def get_model(self, input_dim):
        """
            Choose a different model.
        Parameters
        ----------
        input_dim: int
            Enter the number of data dimensions.

        Returns
        -------

        """
        if self.model_type == "mlp":
            model = MLPModel_lag(dims=[input_dim, *self.hidden_layers],p=1,
                             bias=self.bias, device=self.device)
            return model