import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd import grad

import math

import flows

"""
Todo: Turn the entire code into conditional VAE, use label as Y
Maybe try some impainting algorithm as well. 
"""

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

# VAE class 
class CVAE(nn.Module):
    def __init__(self, feature_size, latent_size, baseline, conv=None, flow=None, exp_family=False, M=5, cFisher=True, mask_ind=-1):
        super(CVAE, self).__init__()
        self.latent_size = latent_size 
        self.baseline = baseline
        self.mask_ind = mask_ind
        if conv:
            # Mimicking the prior, recognition net structure in CVAE
            self.prior_enc, self.dec = get_conv_nets(latent_size = latent_size, **conv)
            self.recognition_enc, self.dec = get_conv_nets(latent_size = latent_size, **conv)
            self.enc1 = nn.Linear(conv['width']*8**3, latent_size)
            self.enc2 = nn.Linear(conv['width']*8**3, latent_size)

        else:
            # encoder
            self.prior_enc = nn.Sequential(nn.Linear(feature_size, 512), nn.ReLU(True), 
                                     nn.Linear(512, 256), nn.ReLU(True)) #p_theta(z|x,y_hat)
            self.recognition_enc = nn.Sequential(nn.Linear(feature_size, 512), nn.ReLU(True), 
                                     nn.Linear(512, 256), nn.ReLU(True)) #q_theta(z|x,y)
            # Conditional on Y
            self.prior_enc1 = nn.Linear(256, latent_size)
            self.prior_enc2 = nn.Linear(256, latent_size)
            
            self.recognition_enc1 = nn.Linear(256, latent_size)
            self.recognition_enc2 = nn.Linear(256, latent_size)

            # decoder
            self.dec = nn.Sequential(nn.Linear(latent_size, 256), nn.ReLU(True), 
                                     nn.Linear(256, 512), nn.ReLU(True), nn.Linear(512, feature_size))
        
        # Exp. family prior/posterior 
        self.M = M
        self.exp_coef = nn.Parameter(torch.randn(M, latent_size).normal_(0, 0.01))
        
        # Fisher/KL VAE 
        self.cFisher = cFisher
        # use exp_family model for prior
        self.exp_family = exp_family
        
        # exp. family natural parameter/ sufficient statistic
        self.natural_param = nn.Parameter(torch.randn(M*latent_size, 1).normal_(0, 0.01))

        # flow for more complicated latent
        self.flow = flow 

        # sufficient statistic 
        activation = nn.Softplus()

        self.sufficient_stat = nn.Sequential(\
                nn.Linear(  latent_size, M*latent_size), activation, \
                nn.Linear(M*latent_size, M*latent_size), activation,\
                nn.Linear(M*latent_size, M*latent_size), activation, \
                nn.Linear(M*latent_size, M*latent_size), activation, \
                nn.Linear(M*latent_size, M*latent_size))
        
    # Exp. family model     
    def dlnpz_exp(self, z, polynomial=True):
        '''
        --- returns both dz log p(z) and p(z)
        --- up to some multiplicative constant 
        '''
        if polynomial == True:
            c = self.exp_coef
            dlnpz = 0
            lnpz = 0

            for m in range(self.M):
                dlnpz += (m+1)*z**(m) * c[m,:].unsqueeze(0)
                lnpz += z**(m+1) * c[m,:].unsqueeze(0)

            pz = lnpz.sum(dim=1).exp()

            return dlnpz, pz
        else:
            Tz = self.sufficient_stat(z)
            eta = self.natural_param 
            lnpz = torch.mm(Tz, eta).sum()
            dlnpz = grad(lnpz, z, retain_graph=True)[0]
        
            return dlnpz, lnpz.exp()
    
    def baseline_concat(self, x):
        # predict y_hat as initial guess
        y_hat = self.baseline(x)
        xc = x.clone()
        # use initial guess as input label
        xc[x == self.mask_ind] = y_hat[x == self.mask_ind]
        return xc
    
    def xy_concat(self, x, y=None):
        # During training, takes in full images, as x_cat = (x,y) = x_true
        # During testing/validation this function is not called, y_hat is obtained from baseline
        xc = x.clone()
        if y != None:
            xc[x == self.mask_ind] = y[x == self.mask_ind]
        return xc
    
    def prior_encode(self, x):
        #p_theta(z|x_cat), x_cat = (x,y_hat)
        h1 = self.prior_enc(x)
        mu_z = self.prior_enc1(h1)
        logvar_z = self.prior_enc2(h1)
        
        return mu_z, logvar_z
    
    def recognition_encode(self, x):
        #q_phi(z|x_cat), x_cat = (x,y)
        h1 = self.recognition_enc(x)
        mu_z = self.recognition_enc1(h1)
        logvar_z = self.recognition_enc2(h1)
        
        return mu_z, logvar_z
    
    def decode(self, z):
        # p_theta(y|x,z)
        y = self.dec(z)
        
        return y
    
    def forward(self, x, y=None, detach=False, test=False):
        # Rewrite forward to take both prior and recognition encoder, need to reconcil its KL later. 
        xc_prior = self.baseline_concat(x) # x concat with y_guess
        
        if test and y == None:
            mu_z, logvar_z = self.prior_encode(xc_prior) # input of the encoder 
            if self.flow:
                # sample from the new latent 
                # this is a transformation of the original Gaussian
                z = self.flow.sample(noise = mu_z)
            else:
                z = mu_z
            y = self.decode(z)
            return y, None, None
        
        # if not testing, need full input for recognition encoder
        xc_recon = self.xy_concat(x,y) # x concat with y_hat
        
        # encode 
        p_mu_z, p_logvar_z = self.prior_encode(xc_prior) # input of the prior encoder
        q_mu_z, q_logvar_z = self.recognition_encode(xc_recon) # input of the recognition encoder

        # construct latent distribution
        p_std_z = (0.5*p_logvar_z).exp() # std 
        q_std_z = (0.5*q_logvar_z).exp() # std 
        p0 = torch.distributions.normal.Normal(p_mu_z, p_std_z) # dist. of epsilon N(0,1)
        q0 = torch.distributions.normal.Normal(q_mu_z, q_std_z) # dist. of epsilon N(0,1)
        
        # reparameterization trick 
        prior_z = p_mu_z + p_std_z * torch.randn_like(p_std_z) # prior_z ~ p(z|x,y_guess)
        recon_z = q_mu_z + q_std_z * torch.randn_like(q_std_z) # recon_z ~ q(z|x,y)

        if self.flow:
            # sample from the new latent 
            # this is a transformation of the original Gaussian
            z_out = self.flow.sample(noise = prior_z)

            pz0 = torch.distributions.normal.Normal(0., 1.) # prior dist. 
            KL = q0.log_prob(prior_z).sum() - pz0.log_prob(prior_z).sum() # KL[q(z|x) || p(z)]
        else:
            z_out = prior_z
        
        # decode 
        y_hat = self.decode(z_out) #p(y|x,z)

        if self.cFisher is True:

            if self.flow:
                dlnqzy = grad(self.flow.log_probs(recon_z).sum(), xc_recon, create_graph=True)[0]
                dlnqzz = grad(self.flow.log_probs(recon_z).sum(), recon_z, create_graph=True)[0]
                #dlnpzz = grad(p0.log_prob(recon_z).sum(), recon_z, create_graph=True)[0]
            else:
                dlnqzy = grad(q0.log_prob(recon_z).sum(), xc_recon, create_graph=True)[0] # d/dx log q(z|x,y)
                dlnqzz = grad(q0.log_prob(recon_z).sum(), recon_z, create_graph=True)[0] # d/dz log q(z|x,y)
                #dlnpzz = grad(p0.log_prob(recon_z).sum(), recon_z, create_graph=True)[0] # d/dz log p(z|x,y)

            stability = 0.5* dlnqzy.pow(2).sum() # stability term
            
            dlnpz = -z_out # Gaussian Prior on z
            pyxz = torch.distributions.normal.Normal(y_hat, 1.0) # p(y|x,z)
            lnpyxz = pyxz.log_prob(y) # log p(x,y|z)
            dlnpyxz = grad(lnpyxz.sum(), z_out, retain_graph=True)[0] # d/dz log p(x,y|z)
            fisher_div = 0.5*(dlnqzz - dlnpz - dlnpyxz).pow(2).sum() # Fisher div. with one sample from q(z|x)

            if self.flow:
                fisher_div += KL
            
            return y_hat, fisher_div, stability
        
        else:
            # Implement conditional VAE (Sohn et.al) here
            # KL[q(z|x,y) || p(z|x,y)], pushes prior and recon closer
            KL = torch.log(p_std_z) - torch.log(q_std_z) + (q_std_z**2 + (q_mu_z - p_mu_z)**2)/(2*p_std_z**2) 
            KL = KL.sum()
            return y_hat, KL 
    
    # the VAE loss function 
    def loss(self, y, output):
        if self.cFisher is True:
            y_hat, fisher_div, stability = output 
            MSE = 0.5*(y-y_hat).pow(2).sum()
            loss = fisher_div + MSE + stability 
        else:
            y_hat, KL = output 
            MSE = 0.5*(y-y_hat).pow(2).sum()
            # BCE = F.binary_cross_entropy(x_hat, x.detach(), reduction='sum')
            loss = KL + MSE 

        return loss / y.shape[0], MSE / y.shape[0] 

# TODO remove some of the hard coding from here
class Upsample2d(nn.Module):
    def forward(self, x):
        return F.interpolate(x, scale_factor=2)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Unflatten(nn.Module):
    def __init__(self, im_size=8):
        super(Unflatten, self).__init__()
        self.im_size = im_size

    def forward(self, x):
        return x.view(x.size(0), -1, self.im_size, self.im_size)

def get_conv_nets(latent_size, width=64, in_channels=3, fs=5, act_enc=nn.LeakyReLU(), act_dec=nn.ReLU(), n_layers=4, pooling=nn.AvgPool2d(2), tanh = True):

    padding = math.floor(fs/2)

    enc_modules = [nn.Conv2d(in_channels, width, fs, padding = padding), act_enc, pooling]
    dec_modules = [nn.Linear(latent_size, width *8*8*8), Unflatten()]

    for i in range(1, n_layers):

        if i == n_layers - 1:
            enc_modules += [nn.Conv2d(width * 2 **(i - 1), width * 2 ** i, fs, padding = padding),
                    act_enc]
        else:
            enc_modules += [nn.Conv2d(width * 2 **(i - 1), width * 2 ** i, fs, padding = padding),
                    nn.BatchNorm2d(width * 2 ** i), 
                    act_enc,
                    pooling]

    for i in range(n_layers-1, 0, -1):

        dec_modules += [Upsample2d(),
            nn.Conv2d(width * 2 ** i, width * 2 ** (i - 1), fs, padding = padding),
            nn.BatchNorm2d(width * 2 ** (i - 1)),
            act_dec]

    enc_modules.append(Flatten())
    dec_modules += [nn.Conv2d(width, in_channels, fs, padding = padding)]

    if tanh:
        dec_modules.append(nn.Tanh())

    conv_encoder = nn.Sequential( * enc_modules)
    conv_decoder = nn.Sequential( * dec_modules)

    print(conv_encoder)
    print(conv_decoder)

    return conv_encoder, conv_decoder
