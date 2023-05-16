import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from scipy.stats import kde
from torchvision.utils import make_grid
from torch.autograd import grad
import numpy as np 
import numpy.linalg as la 
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import os

from nets import CVAE
import flows
import baseline

from visualize import visualize

# set up device 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_path = '/hpc/group/tarokhlab/hy190/MV-SDE/Data/MNIST'

# plotting images 
def show(img):
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

# load data
batch_size = 512
data_name  = 'mnist'
num_quad   = 1
early_stop_patience = 20
retrain_baseline = False
cFisher = False

netsavepath = "/scratch/hy190/CFAE/cFisher_{}_data_{}_numquad_{}/".format(cFisher, data_name,num_quad)
plotsavepath = "results/cFisher_{}_data_{}_numquad_{}/".format(cFisher, data_name,num_quad)
if not os.path.exists(netsavepath):
    os.makedirs(netsavepath)
if not os.path.exists(plotsavepath):
    os.makedirs(plotsavepath)


train_loader = None
test_loader  = None
valid_loader  = None

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# optimizer
def make_optimizer(optimizer_name, model, flow = None, **kwargs):

    if flow:
        params = [{'params': flow.parameters(), 'lr': kwargs['lr'] / kwargs['flow_scale']},
                {'params': list(set(model.parameters()) - set(flow.parameters())), 'lr': kwargs['lr']}]
    else:
        params = [{'params': model.parameters(), 'lr': kwargs['lr']}]

    if optimizer_name=='Adam':
        optimizer = optim.Adam(params, betas=[0.9, 0.999])
    elif optimizer_name=='SGD':
        optimizer = optim.SGD(model.parameters(),lr=kwargs['lr'],momentum=kwargs['momentum'], weight_decay=kwargs['weight_decay'])
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(),lr=kwargs['lr'], momentum=0.9)
    else:
        raise ValueError('Not valid optimizer name')
    return optimizer

# scheduler
def make_scheduler(scheduler_name, optimizer, **kwargs):
    if scheduler_name=='MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=kwargs['milestones'],gamma=kwargs['factor'])
    elif scheduler_name=='exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=.9998)
    else:
        raise ValueError('Not valid scheduler name')
    return scheduler

# training parameters
optimizer_name = 'Adam'
scheduler_name = 'exponential'
num_epochs = 300
lr = 2e-4
device = torch.device(device)
flow = False
n_prior_update = 5
flow_scale = 1
mask_ind = -1

if data_name == 'celeba':
    conv = {'width' : 64}
else:
    conv = None

print('lr: {}'.format(lr))
print('epochs: {}'.format(num_epochs))
print('batch_size: {}'.format(batch_size))
print('n_prior_update: {}'.format(n_prior_update))
print('flow_scale: {}'.format(flow_scale))
print(conv)

# VAE
latent_size = 64 
if flow:
    flow_width  = 256
    flow_layers = 8
    modules = []
    mask = torch.arange(0, latent_size) % 2
    mask = mask.to("cuda:0" if torch.cuda.is_available() else "cpu")

    for _ in range(flow_layers):
        modules += [
            flows.LUInvertibleMM(latent_size),
            flows.CouplingLayer(
                latent_size, flow_width, mask, 0,
                s_act='tanh', t_act='relu')
        ]
        mask = 1 - mask

    flow_net = flows.FlowSequential(*modules).to("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    flow_net = None
    

import datasets
if data_name == 'mnist':
    train_loader, valid_loader, test_loader = datasets.get_mnist(batch_size = batch_size, dataset_directory = data_path, 
                                                                 num_quadrant_inputs=num_quad, mask_ind=mask_ind)
elif data_name == 'celeba':
    train_loader, valid_loader, test_loader = datasets.get_celeba(batch_size = batch_size, dataset_directory = data_path)
    
# Train baseline
datasetloaders = {'train':train_loader, 'val':valid_loader}
dataset_sizes = {'train':len(train_loader), 'val': len(valid_loader)}
if retrain_baseline:
    baseline_net = baseline.train(
        device=device,
        dataloaders=datasetloaders,
        dataset_sizes=dataset_sizes,
        learning_rate=2e-4,
        num_epochs=num_epochs,
        early_stop_patience=early_stop_patience,
        model_path='baseline_net_q{}.pt'.format(num_quad)
    )
else:
    baseline_net = baseline.baseline_reload(datasetloaders, device, model_path='baseline_net_q{}.pt'.format(num_quad)).to(device)

local_vae = CVAE(feature_size=784, 
                 latent_size=latent_size,
                 baseline=baseline_net, 
                 M=4, 
                 conv=conv, 
                 flow=flow_net,
                 mask_ind=mask_ind,
                 cFisher=cFisher, 
                 exp_family=False).to(device)

optimizer = make_optimizer(optimizer_name, local_vae, flow = flow_net, lr=lr, weight_decay=0, momentum=0.01, flow_scale=flow_scale)
scheduler = make_scheduler(scheduler_name, optimizer, milestones=[25, 50, 70, 90], factor=0.99)

# Train the VAE
loader = train_loader
data_shape = 0 
best_loss = 100000

for epoch in tqdm(range(num_epochs+1)):
    loss_epoch = 0
    mse_epoch = 0 
    for iter_, (data, _) in enumerate(loader):
        inputs = data["input"]
        masked_outputs = data["output"]
        data_shape = inputs.shape
        if len(data_shape) < 4:
            data_shape = [data_shape[0], 1, data_shape[1], data_shape[2]]
        # zero grad
        optimizer.zero_grad()

        # forward pass
        if data_name == 'mnist':
            inputs = Variable(inputs.reshape(inputs.shape[0], 784), requires_grad=True).to(device)
            masked_outputs = Variable(masked_outputs.reshape(masked_outputs.shape[0], 784), requires_grad=True).to(device)
        else:
            inputs = Variable(inputs, requires_grad=True).to(device)
            masked_outputs = Variable(masked_outputs, requires_grad=True).to(device)


        if iter_ % n_prior_update :
            output = local_vae.forward(inputs,masked_outputs,detach=False, test=False)
        else:
            output = local_vae.forward(inputs, masked_outputs)

        loss, mse = local_vae.loss(masked_outputs, output)
        loss_epoch += loss.item()
        mse_epoch += mse.item()

        plt.figure()
        show(make_grid(output[0][0:64, :].reshape(64,data_shape[1],data_shape[2],data_shape[3]).cpu().detach(), padding=0, normalize=True))
        plt.savefig('{}/train_samps.png'.format(plotsavepath))
        plt.close('all')


        # backward pass
        loss.backward()

        # update parameters
        optimizer.step()
    scheduler.step()

    # print loss at the end of every epoch
    print('Epoch : ', epoch, 
          ' | Loss VAE: {:.4f} | Loss MSE: {:.4f}'.format(loss_epoch / len(loader), mse_epoch / len(loader)), 
          ' | lr : ', optimizer.param_groups[0]['lr'])
    if epoch % 1 == 0:
        local_vae.eval()
        for iter_, (data, _) in enumerate(valid_loader):
            inputs = data["input"]
            masked_outputs = data["output"]
            data_shape = inputs.shape
            if len(data_shape) < 4:
                data_shape = [data_shape[0], 1, data_shape[1], data_shape[2]]
            # zero grad
            optimizer.zero_grad()
            # forward pass
            if data_name == 'mnist':
                inputs = Variable(inputs.reshape(inputs.shape[0], 784), requires_grad=True).to(device)
                masked_outputs = Variable(masked_outputs.reshape(masked_outputs.shape[0], 784), requires_grad=True).to(device)
            else:
                inputs = Variable(inputs, requires_grad=True).to(device)
                masked_outputs = Variable(masked_outputs, requires_grad=True).to(device)
                
            output = local_vae.forward(inputs, None)
            loss, mse = local_vae.loss(masked_outputs, output)
            loss_epoch += loss.item()
            
            
        visualize(device=device, pre_trained_baseline=local_vae.baseline, dataloader = valid_loader, 
                  pre_trained_cvae=local_vae, num_samples=10,num_images=10, data_shape=data_shape, data_name=data_name,
                  image_path='{}/epoch_{}_gen_samps.png'.format(plotsavepath, epoch))
        
        # save the best model
        if loss_epoch < best_loss:
            best_loss = loss_epoch
            torch.save(local_vae.state_dict(), '{}/best_epoch_{}_nll_{:.1f}.pt'.format(netsavepath,epoch,best_loss, num_quad))

        # save the model
        torch.save(local_vae.state_dict(), '{}/model_epoch_{}.pt'.format(netsavepath,epoch))

