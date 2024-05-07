import torch
import torch.nn as nn
from smile_estimator import estimate_mutual_information
from torch.nn.utils.spectral_norm import spectral_norm
import torch.nn.functional as F

class SupervenientFeatureNetwork(nn.Module):
    def __init__(
            self,
            num_atoms: int,
            feature_size: int,
            hidden_sizes: list,
            include_bias: bool = True
        ):
        super(SupervenientFeatureNetwork, self).__init__()
        self.feature_size = feature_size
        layers = []
        input_size = num_atoms
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size, bias=include_bias))
            layers.append(nn.ReLU())
            input_size = hidden_size
        layers.append(nn.Linear(input_size, feature_size, bias=include_bias))
        self.f = nn.Sequential(*layers)

    def forward(self, x):
        return self.f(x)
    


class SkipConnectionSupervenientFeatureNetwork(nn.Module):
    def __init__(
        self,
        num_atoms: int,
        feature_size: int,
        hidden_sizes: list,
        include_bias: bool = True
        ):
        super(SkipConnectionSupervenientFeatureNetwork, self).__init__()

        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one element")
        
        self.feature_size = feature_size

        # Use the first hidden layer size for the initial projection
        self.initial_projection = nn.Linear(num_atoms, hidden_sizes[0], bias=include_bias)

        # Creating the hidden layers with skip connections
        self.hidden_layers = nn.ModuleList()
        input_size = hidden_sizes[0]
        for hidden_size in hidden_sizes:
            layer = nn.Linear(input_size, hidden_size, bias=include_bias)
            self.hidden_layers.append(layer)
            input_size = hidden_size

        # Final downprojection to the feature size
        self.final_projection = nn.Linear(input_size, feature_size, bias=include_bias)

    def forward(self, x):
        # Initial projection
        x = self.initial_projection(x)

        # Passing through each hidden layer with ReLU and skip connection
        for layer in self.hidden_layers:
            identity = x
            x = layer(x)
            x = F.relu(x)
            x = torch.add(x, identity)

        # Final projection without non-linearity
        x = self.final_projection(x)
        return x


class DecoupledSmileMIEstimator(nn.Module):
    def __init__(
            self,
            feature_size: int,
            critic_output_size: int,
            hidden_sizes_1: list,
            hidden_sizes_2: list,
            clip: float,
            include_bias: bool = True,
            add_spec_norm: bool = False
        ):
        super(DecoupledSmileMIEstimator, self).__init__()

        def spec_norm(layer):
            if add_spec_norm:
                return spectral_norm(layer)
            else:
                return layer

        layers = []
        input_size = feature_size
        for hidden_size in hidden_sizes_1:
            layers.append(spec_norm(nn.Linear(input_size, hidden_size, bias=include_bias)))
            layers.append(nn.ReLU())
            input_size = hidden_size
        layers.append(spec_norm(nn.Linear(input_size, critic_output_size, bias=include_bias)))
        self.critic_1 = nn.Sequential(*layers)

        layers = []
        input_size = feature_size
        for hidden_size in hidden_sizes_2:
            layers.append(spec_norm(nn.Linear(input_size, hidden_size, bias=include_bias)))
            layers.append(nn.ReLU())
            input_size = hidden_size
        layers.append(spec_norm(nn.Linear(input_size, critic_output_size, bias=include_bias)))

        self.critic_2 = nn.Sequential(*layers)
        self.clip = clip

    def forward(self, x1, x2):
        x1_critic = self.critic_1(x1)
        x2_critic = self.critic_2(x2)

        scores = torch.matmul(x1_critic, x2_critic.t())
        MI = estimate_mutual_information('smile', scores, clip=self.clip)
        return MI
    

class DownwardSmileMIEstimator(nn.Module):
    def __init__(
            self,
            feature_size: int,
            critic_output_size: int,
            hidden_sizes_v_critic: list,
            hidden_sizes_xi_critic: list,
            clip: float,
            include_bias: bool = True,
            add_spec_norm: bool = False
        ):
        super(DownwardSmileMIEstimator, self).__init__()

        def spec_norm(layer):
            if add_spec_norm:
                return spectral_norm(layer)
            else:
                return layer

        v_encoder_layers = []
        input_size = feature_size
        for hidden_size in hidden_sizes_v_critic:
            # TODO: Understand what the fuck spectral norm actually is
            # NOTE: Spectral norm removed, stabalizes shit but seems to hamstring downward critics hardddddd
            v_encoder_layers.append(spec_norm(nn.Linear(input_size, hidden_size, bias=include_bias)))
            v_encoder_layers.append(nn.ReLU())
            input_size = hidden_size
        v_encoder_layers.append(spec_norm(nn.Linear(input_size, critic_output_size, bias=include_bias)))
        self.v_encoder = nn.Sequential(*v_encoder_layers)

        atom_encoder_layers = []
        input_size = 1 # 1 becuase this will always be the dim of a constituent part of our system
        for hidden_size in hidden_sizes_xi_critic:
            atom_encoder_layers.append(spec_norm(nn.Linear(input_size, hidden_size, bias=include_bias)))
            atom_encoder_layers.append(nn.ReLU())
            input_size = hidden_size
        atom_encoder_layers.append(spec_norm(nn.Linear(input_size, critic_output_size, bias=include_bias)))
        self.atom_encoder = nn.Sequential(*atom_encoder_layers)

        self.clip = clip
    
    def forward(self, v1, x0i):
        v1_encoded = self.v_encoder(v1)
        x0i_encoded = self.atom_encoder(x0i)

        scores = torch.matmul(v1_encoded, x0i_encoded.t())
        MI = estimate_mutual_information('smile', scores, clip=self.clip)
        return MI
    
class GeneralSmileMIEstimator(nn.Module):
    def __init__(
            self,
            x_dim: int,
            y_dim: int,
            critic_output_size: int,
            x_critics_hidden_sizes: list,
            y_critics_hidden_sizes: list,
            clip: float,
            include_bias: bool = True
        ):
        super(GeneralSmileMIEstimator, self).__init__()

        x_encoder_layers = []
        input_size = x_dim
        for hidden_size in x_critics_hidden_sizes:
            x_encoder_layers.append(nn.Linear(input_size, hidden_size, bias=include_bias))
            x_encoder_layers.append(nn.ReLU())
            input_size = hidden_size
        x_encoder_layers.append(nn.Linear(input_size, critic_output_size, bias=include_bias))
        self.x_encoder = nn.Sequential(*x_encoder_layers)

        y_encoder_layers = []
        input_size = y_dim
        for hidden_size in y_critics_hidden_sizes:
            y_encoder_layers.append(nn.Linear(input_size, hidden_size, bias=include_bias))
            y_encoder_layers.append(nn.ReLU())
            input_size = hidden_size
        y_encoder_layers.append(nn.Linear(input_size, critic_output_size, bias=include_bias))
        self.y_encoder = nn.Sequential(*y_encoder_layers)

        self.clip = clip

    def forward(self, x, y):
        x_encoded = self.x_encoder(x)
        y_encoded = self.y_encoder(y)

        scores = torch.matmul(x_encoded, y_encoded.t())
        MI = estimate_mutual_information('smile', scores, clip=self.clip)
        return MI


class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :      provides the estimation with input samples  
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim] 
    '''
    def __init__(
            self,
            v_dim,
            mu_hidden_sizes: list,
            logvar_hidden_sizes: list
        ):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        # p_logvar outputs log of variance of q(Y|X)

        # NOTE: hard coding in 1 for output dim here (and below) so that we don't have to make assumptions about the covariance matrix between the different components of y
        p_mu_layers = []
        input_size = v_dim
        for hidden_size in mu_hidden_sizes:
            p_mu_layers.append(nn.Linear(input_size, hidden_size))
            p_mu_layers.append(nn.ReLU())
            input_size = hidden_size
        p_mu_layers.append(nn.Linear(input_size, 1))
        self.p_mu = nn.Sequential(*p_mu_layers)

        p_logvar_layers = []
        input_size = v_dim
        for hidden_size in logvar_hidden_sizes:
            p_logvar_layers.append(nn.Linear(input_size, hidden_size))
            p_logvar_layers.append(nn.ReLU())
            input_size = hidden_size
        p_logvar_layers.append(nn.Linear(input_size, 1))
        p_logvar_layers.append(nn.Tanh())
        self.p_logvar = nn.Sequential(*p_logvar_layers)


    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
    
    def forward(self, x_samples, y_samples): 
        mu, logvar = self.get_mu_logvar(x_samples)
        
        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples)**2 /2./logvar.exp()  
        
        prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp() 

        return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

    def loglikeli(self, x_samples, y_samples): # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples)
        return 0.5 * (-(mu - y_samples)**2 /logvar.exp()-logvar - torch.log(torch.tensor(2 * math.pi))).sum(dim=1).mean(dim=0)
    # NOTE: y should be dim 1
    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)

    
