import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.manifold import TSNE
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
from baseline import MaskedBCELoss, BaselineNet
from nets import CVAE


def imshow(inp, cFisher, image_path=None):
    inp = inp.cpu().numpy().transpose((1, 2, 0))
    space = np.ones((inp.shape[0], 50, inp.shape[2]))
    inp = np.concatenate([space, inp], axis=1)

    ax = plt.axes(frameon=False, xticks=[], yticks=[])
    ax.text(0, 23, 'Inputs:')
    ax.text(0, 23 + 28 + 3, 'Truth:')
    ax.text(0, 23 + (28 + 3) * 2, 'NN:')
    ax.text(0, 23 + (28 + 3) * 3, 'CFAE:' if cFisher else 'CVAE:')
    ax.imshow(inp)

    if image_path is not None:
        Path(image_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0.1)
    else:
        plt.show()

    plt.clf()


def visualize(device, pre_trained_baseline, dataloader, 
              pre_trained_cvae, num_samples, num_images, data_shape, data_name, image_path=None):
    with torch.no_grad(): 
        pre_trained_baseline.eval()
        pre_trained_cvae.eval()
        batch = next(iter(dataloader))[0]
        inputs = batch['input'].to(device)[:num_images]
        masked_outputs = batch['output'].to(device)[:num_images]

        # forward pass
        if data_name == 'mnist':
            inputs = Variable(inputs.reshape(inputs.shape[0], 784), requires_grad=False).to(device)
            masked_outputs = Variable(masked_outputs.reshape(masked_outputs.shape[0], 784), requires_grad=False).to(device)
        else:
            inputs = Variable(inputs, requires_grad=False).to(device)
            masked_outputs = Variable(masked_outputs, requires_grad=False).to(device)
        
        originals = masked_outputs.clone()
        originals[inputs != -1] = inputs[inputs != -1]
        data_shape[0] = num_samples
        # Make predictions
        baseline_preds = pre_trained_baseline(inputs)
        
        cvae_preds = torch.zeros(num_samples, num_images, data_shape[-2]*data_shape[-1]).to(device)
        for k in range(num_samples):
            cvae_preds_temp = pre_trained_cvae(inputs, None, False, test=True)[0]
            cvae_preds[k,:,:] = cvae_preds_temp

        # Predictions are only made in the pixels not masked. This completes
        # the input quadrant with the prediction for the missing quadrants, for
        # visualization purpose
        baseline_preds[masked_outputs == -1] = inputs[masked_outputs == -1]
        for i in range(cvae_preds.shape[0]):
            cvae_preds[i][masked_outputs == -1] = inputs[masked_outputs == -1]

        # adjust tensor sizes
    inputs = inputs
    inputs[inputs == -1] = 1
    inputs = inputs.view(-1, 28,28).unsqueeze(1)
    originals = originals.view(-1, 28,28).unsqueeze(1)
    baseline_preds = baseline_preds.reshape(-1, 28,28).unsqueeze(1)
    cvae_preds = cvae_preds.view(-1, 28, 28).unsqueeze(1)
    # make grids
    inputs_tensor = make_grid(inputs, nrow=num_images, padding=0)
    originals_tensor = make_grid(originals, nrow=num_images, padding=0)
    separator_tensor = torch.ones((3, 5, originals_tensor.shape[-1])).to(device)
    baseline_tensor = make_grid(baseline_preds, nrow=num_images, padding=0)
    cvae_tensor = make_grid(cvae_preds, nrow=num_images, padding=0)
    
    # add vertical and horizontal lines
    for tensor in [originals_tensor, baseline_tensor, cvae_tensor]:
        for i in range(num_images - 1):
            tensor[:, :, (i + 1) * 28] = 0.3

    for i in range(num_samples - 1):
        cvae_tensor[:, (i + 1) * 28, :] = 0.3

    # concatenate all tensors
    grid_tensor = torch.cat([inputs_tensor, separator_tensor, originals_tensor,
                             separator_tensor, baseline_tensor,
                             separator_tensor, cvae_tensor], dim=1)
    # plot tensors
    imshow(grid_tensor, pre_trained_cvae.cFisher, image_path=image_path)