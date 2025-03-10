import numpy as np
import torch
from utils import train_pi
from time_varying_weights import pi_tn
from linear_utils import CASTOR_linear_1_regime
from scipy.stats import multivariate_normal
from CASTOR_nonlinear import CASTOR_nonlinear
from utils import Tensor


class CASTOR:
              def __init__(self, 
                            data, 
                            X,
                            Xlags,
                            lags,
                            ):
                              self.data, self.X, self.Xlags, self.lags = data, X, Xlags,lags
               
                              
                              
              def run_linear(self, max_it, loss_thres, w_threshold, window, zeta):
                              n = int(self.X.shape[1])
                              m = self.X.shape[0]
                              N_regime = m//window
                              L = np.zeros((self.lags*n, self.lags*n, N_regime))
                              for it in range(max_it):
                                             model_n = [CASTOR_linear_1_regime(nlags=self.lags)  for _ in range(N_regime)]
                                             graphss = [1 for i in range(N_regime)]
                                             yl = np.zeros((m, n , N_regime))
                                             if it == 0:
                                                            p = np.zeros((m,N_regime))
                                                            for c in range(N_regime):
                                                                           if c  == N_regime -1:
                                                                                          p[c*window:,c] = np.ones(m-c*window)
                                                                                          L[:, :, c] = model_n[c].infer_from_data(self.data,
                                                                                                                                  self.X[c*window:,:],
                                                                                                                                       self.Xlags[c*window:,:],
                                                                                                                                       w_threshold=w_threshold)
                                                                           else:
                                                                                          p[c*window:(c+1)*window,c] = np.ones(window)
                                                                                          L[:, :, c] = model_n[c].infer_from_data(self.data,
                                                                                                                                  self.X[c*window:(c+1)*window,:],
                                                                                                                                  self.Xlags[c*window:(c+1)*window,:],
                                                                                                                                  w_threshold=w_threshold)
                                             

                                             else:
                                                            for c in range(N_regime):
                                                                           gamma = gamma_hat[:,c].reshape((m,1))
                                                                           L[:, :, c] = model_n[c].infer_from_data(self.data,
                                                                                                                   gamma*self.X,
                                                                                                                   gamma*self.Xlags,
                                                                                                                   w_threshold=w_threshold)
                                                
                                             pall = 0
                                             gamma_hat = np.zeros((m, N_regime))
                                             for class_idx in range(N_regime):
                                    

                                                            yl[:, :, class_idx] = self.X - self.X.dot(L[:n, :n, class_idx]) - self.Xlags.dot(L[n:self.lags*n,:n, class_idx])

                                                            pall = pall + p[:,class_idx] * multivariate_normal.pdf(yl[:, :, class_idx], mean= np.zeros(n),
                                                                                                                        cov=1*np.eye(n))
                                                            gamma_hat[:, class_idx] = p[:,class_idx] * multivariate_normal.pdf(yl[:, :, class_idx], mean= np.zeros(n),
                                                                                                                                       cov=1*np.eye(n))
                                             idx = np.argmax(gamma_hat/pall.reshape((m,1)), axis=-1)
                                             gamma_hat = np.zeros( gamma_hat.shape )
                                             gamma_hat[ np.arange(gamma_hat.shape[0]), idx] = 1

                                             time_t = np.linspace(0,20*N_regime,self.X.shape[0]).reshape((self.X.shape[0],1))
                                             t = torch.tensor(time_t)
                                             
                                             model = pi_tn(N_regime)
                                             
                                             p,loss,model_ = train_pi(500,model, t.float(), torch.tensor(gamma_hat).float())
                                             #print("yes")
                                             p = p.detach().numpy()
                                             print(loss)
                                             if N_regime == 2:
                                                            loss_thres = 0.8
                                             while loss>=loss_thres:
                                                            p,loss,model_ = train_pi(100,model_, t.float(), torch.tensor(gamma_hat).float())
                                                            p = p.detach().numpy()
                              
                                             gamma_sum = np.sum(gamma_hat, axis=0)
                                             gamma_sum[gamma_sum<zeta] = 0
                                             indexes, = np.where(gamma_sum!= 0)
                                             gamma_hat = gamma_hat[:,indexes]
                                             p = p[:,indexes]
                                             N_regime = len(indexes)
                                             print(str(np.sum(gamma_hat, axis=0))+" iter: "+str(it)+" , p: "+str(np.sum(p, axis=0)))
                              return model_n,graphss,gamma_hat, L
                            
              def run_nonlinear(self,max_it,loss_thres,window,device,zeta):
                n = int(self.X.shape[1])
                m = self.X.shape[0]
                N_regime = m//window
              
                for it in range(max_it):
                        model_n = [CASTOR_nonlinear(hidden_layers = (10,1),device = device)  for _ in range(N_regime)]
            
                        yl = np.zeros((m, n , N_regime))
                        if it == 0:
                                    p = np.zeros((m,N_regime))
                                    for c in range(N_regime):
                                                if c  == N_regime -1:
                                                            p[c*window:,c] = np.ones(m-c*window)
                                                            model_n[c].learn(Tensor(self.X[c*window:,:]), 
                                                                             Tensor(self.Xlags[c*window:,:]))
                                                else:
                                                            p[c*window:(c+1)*window,c] = np.ones(window)
                                                            model_n[c].learn(Tensor(self.X[c*window:(c+1)*window,:]),
                                                                             Tensor(self.Xlags[c*window:(c+1)*window,:],))
                                                
                        else:
                                    for c in range(N_regime):
                                                gamma = gamma_hat[:,c].reshape((m,1))
                                                model_n[c].learn(Tensor(gamma*self.X),
                                                                 Tensor(gamma*self.Xlags))
                        pall = 0                        
                        gamma_hat = np.zeros((m, N_regime))
                        for class_idx in range(N_regime):
                                    X_torch = torch.from_numpy(self.X)
                                    Y_torch = torch.from_numpy(self.Xlags)
                                    X_torch = X_torch.to(device)
                                    Y_torch = Y_torch.to(device)
                                    vector = X_torch - model_n[class_idx].nonlinear_mixing(X_torch.float(),Y_torch.float())
                                    yl[:, :, class_idx] = vector.cpu().detach().numpy()
                                    pall = pall + p[:,class_idx] * multivariate_normal.pdf(yl[:, :, class_idx], mean= np.zeros(n),
                                                                                    cov=1*np.eye(n))
                                    gamma_hat[:, class_idx] = p[:,class_idx] * multivariate_normal.pdf(yl[:, :, class_idx], mean= np.zeros(n),
                                                                                    cov=1*np.eye(n))
                        idx = np.argmax(gamma_hat/pall.reshape((m,1)), axis=-1)
                        gamma_hat = np.zeros( gamma_hat.shape )
                        gamma_hat[ np.arange(gamma_hat.shape[0]), idx] = 1
                               
                        t = torch.tensor(np.linspace(0,20*N_regime,self.X.shape[0]).reshape((self.X.shape[0],1)))
                        model = pi_tn(N_regime)
                        p,loss,model_ = train_pi(200,model, t.float(), torch.tensor(gamma_hat).float())
                        p = p.detach().numpy()
                        while loss>=loss_thres:
                                    p,loss,model_ = train_pi(100,model_, t.float(), torch.tensor(gamma_hat).float())
                                    p = p.detach().numpy()
                        gamma_sum = np.sum(gamma_hat, axis=0) 
                        gamma_sum[gamma_sum<zeta] = 0  
                        indexes, = np.where(gamma_sum!= 0)
                        gamma_hat = gamma_hat[:,indexes]
                        p = p[:,indexes]
                        N_regime = len(indexes)          
                        print(str(np.sum(gamma_hat, axis=0))+" iter: "+str(it)+" , p: "+str(np.sum(p, axis=0)))
                return gamma_hat, model_n               
               