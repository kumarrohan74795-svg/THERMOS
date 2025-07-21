import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

############################################################################################
# Function to initialize weights
############################################################################################
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class MOPPO_Actor_Critic(nn.Module):
    def __init__(self, args, actor):
        super().__init__()

        self.args = args
        self.state_size = args.obs_shape
        self.action_size = args.action_shape
        self.reward_size = args.reward_size
        # self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size    
        self.actor = actor


        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.state_size+self.reward_size, self.hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden_size, self.hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden_size, self.reward_size), std=1.0),
        )

        self.actor = actor

class FDDTN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_input = args.obs_shape + args.reward_size
        self.n_output = args.action_shape
        self.labels = None
        self.depth = args.tree_depth
        
        param_init_func = {'uniform': lambda *x: 0.5 * torch.ones(*x),
                    'random': torch.rand,
                    'zero': torch.zeros}
        
        self.param_initer =  param_init_func['random']
        
        self.leafs = nn.Parameter(self.param_initer(2 ** (self.depth - 1), self.n_output))
        self.nodes_beta = nn.ParameterList()
        self.nodes_phi = nn.ParameterList()

        self.alpha = torch.Tensor([args.tree_alpha])
        self.alpha.requires_grad = True
        self.alpha = nn.Parameter(self.alpha)
        
        self.action_labels = None
        self.hard_mode = False

        self.labels = None

        if self.action_labels is not None:
            # you need the same number of action labels as number of outputs
            assert len(self.action_labels) == self.n_output
            #this needs to be a regular python list for rendering
            assert isinstance(self.action_labels, list)
        if self.labels is not None:
            # you need the same number of labels as inputs
            assert len(self.labels) == self.n_input
            #this needs to be a regular python list for rendering
            assert isinstance(self.labels, list)
        for d in range(self.depth - 1, 0, -1):
            self.nodes_beta.append(
                nn.Parameter(self.param_initer(2 ** (d - 1), self.n_input)))
            self.nodes_phi.append(nn.Parameter(self.param_initer(2 ** (d - 1), 1)))
            # self.nodes_alpha.append(nn.Parameter(param_initer(2 ** (d - 1), 1)))
            
    def make_hard(self):
        self.hard_mode = True

    def make_soft(self):
        self.hard_mode = False

    def soft_forward(self, x):
        if x.ndimension() == 1:
            x = x.unsqueeze(0)
        n_batch = x.shape[0]
        iterate = self.leafs.unsqueeze(0).expand(n_batch, -1, -1)
        for betas, phis in zip(self.nodes_beta, self.nodes_phi):
            beta_mat = F.softmax(betas, 1).unsqueeze(0).expand(n_batch, -1, -1)
            x_mat = x.unsqueeze(2)
            sig = torch.sigmoid( self.alpha * (
                torch.matmul(beta_mat, x_mat) - phis.unsqueeze(0).expand(n_batch, -1, -1)))
            # sig = nn.Sigmoid( self.alpha * (
            #     torch.matmul(beta_mat, x_mat) - phis.unsqueeze(0).expand(n_batch, -1, -1)))    
            lefts = (1 - sig).expand_as(iterate[:, 0::2]) * \
                    iterate[:, 0::2]
            rights = sig.expand_as(iterate[:, 1::2]) * \
                     iterate[:, 1::2]
            iterate = lefts + rights
        iterate = iterate.squeeze()
        return F.softmax(iterate, dim=-1)

    def forward(self, x, deterministic = False):
        if deterministic:
            with torch.no_grad():
                return self.soft_forward(x)
        else:
            return self.soft_forward(x)
