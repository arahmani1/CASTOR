import torch
import torch.nn as nn

class pi_tn(nn.Module):
            def __init__(self, regime):
                        super(pi_tn, self).__init__()
                        self.regime = regime
                        self.linear = nn.Linear(1,self.regime)
                        
            def forward(self,t):
                        outp = self.linear(t)
                        return outp
         