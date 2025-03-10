import torch
import torch.nn as nn 
import math

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