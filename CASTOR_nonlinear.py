import random
import numpy as np
import pandas as pd
import scipy.optimize as sopt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn 
from utils import Tensor, squared_loss
import abc
from MLP_lag import MLPModel_lag

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


class CASTOR_nonlinear(BaseLearner):
    """
    We used and modified the code of Xun Zheng https://github.com/xunzheng/notears/tree/master
    """
    def __init__(self, 
                 p: int = 1,
                 lambda1: float = 0.01,
                 lambda2: float = 0.01,
                 max_iter: int = 200,
                 h_tol: float = 1e-8,
                 rho_max: float = 1e+16,
                 w_threshold: float = 0.3,
                 A_threshold: float = 0.3,
                 hidden_layers: tuple = (10, 1),
                 bias: bool = True,
                 model_type: str = "mlp",
                 device_type: str = "gpu",
                 ):

        super().__init__()
        self.p = p
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.w_threshold = w_threshold
        self.A_threshold = A_threshold
        self.hidden_layers = hidden_layers
        self.bias = bias
        self.model_type = model_type
        self.device_type = device_type
        self.rho, self.alpha, self.h = 1.0, 0.0, np.inf
        
        # We will overwrite these matrices with our outputs
        self.W_no_thres = torch.zeros((10,10))
        self.A_no_thres = torch.zeros((10,10))
        self.A_lag = torch.zeros((10,10))
        if self.device_type == 'gpu':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        

    def learn(self, data_inst, data_lag, columns=None, **kwargs):
       
        X = Tensor(data_inst, columns=columns)
        X_lag = Tensor(data_lag, columns=columns)
        input_dim = int(X.shape[1])
        model = self.get_model(input_dim)
        if model:
            W_est,A, nonlinear_mixing = self.CASTOR_nonlinear_optimization(model, X, X_lag)
            self.W_no_thres = W_est
            self.A_no_thres = A
            causal_matrix = (abs(W_est) > self.w_threshold).astype(int)
            A_matrix = (abs(A) > self.A_threshold).astype(int)
            self.weight_causal_matrix = Tensor(W_est,
                                               index=X[:,:input_dim].columns,
                                               columns=X[:,:input_dim].columns)
            self.causal_matrix = Tensor(causal_matrix, index=X[:,:input_dim].columns, columns=X[:,:input_dim].columns)
            
            self.A_lag = A_matrix
            self.nonlinear_mixing = nonlinear_mixing

    def dual_ascent_step(self, model, X,Y):
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

    def CASTOR_nonlinear_optimization(self,
                          model,
                          X,
                          X_lag):
        
        
        dim = int(X.shape[1])
        for _ in range(self.max_iter):
            self.dual_ascent_step(model, X, X_lag)


            if self.h <= self.h_tol or self.rho >= self.rho_max:
                break
        W_est = model.fc1_to_adj()
        A_lag = model.fc1_lag_to_adj()
        

        return W_est,A_lag, model

    def get_model(self, input_dim):
        if self.model_type == "mlp":
            model = MLPModel_lag(dims=[input_dim, *self.hidden_layers],p=self.p ,
                             bias=self.bias, device=self.device)
            return model