import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import os 

class BaselineNet(nn.Module):
    def __init__(self, feature_size, hidden_1, hidden_2):
        super().__init__()
        self.fc1 = nn.Linear(feature_size, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, feature_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        hidden = self.relu(self.fc1(x))
        hidden = self.relu(self.fc2(hidden))
        y = torch.sigmoid(self.fc3(hidden))
        return y


class MaskedBCELoss(nn.Module):
    def __init__(self, masked_with=-1):
        super().__init__()
        self.masked_with = masked_with

    def forward(self, input, target):
        target = target.view(input.shape)
        loss = F.binary_cross_entropy(input, target, reduction='none')
        loss[target == self.masked_with] = 0
        return loss.sum()


def train(device, dataloaders, dataset_sizes, learning_rate, num_epochs,
          early_stop_patience, model_path):

    # Train baseline
    _, data = next(enumerate(dataloaders['train']))
    input = data[0]
    feature_size = input['input'][0,:].flatten().size(0)
    baseline_net = BaselineNet(feature_size, 500, 500)
    baseline_net.to(device)
    optimizer = torch.optim.Adam(baseline_net.parameters(), lr=learning_rate)
    criterion = MaskedBCELoss()
    best_loss = np.inf
    early_stop_count = 0

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                baseline_net.train()
            else:
                baseline_net.eval()

            running_loss = 0.0
            num_preds = 0

            bar = tqdm(dataloaders[phase],
                       desc='NN Epoch {} {}'.format(epoch, phase).ljust(20))
            for i, batch in enumerate(bar):
                batch = batch[0]
                inputs = batch['input'].to(device).reshape(-1, feature_size)
                outputs = batch['output'].to(device).reshape(-1, feature_size)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    preds = baseline_net(inputs)
                    loss = criterion(preds, outputs) / inputs.size(0)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()
                num_preds += 1
                if i % 10 == 0:
                    bar.set_postfix(loss='{:.2f}'.format(running_loss / num_preds),
                                    early_stop_count=early_stop_count)

            epoch_loss = running_loss / dataset_sizes[phase]
            # deep copy the model
            if phase == 'val':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(baseline_net.state_dict())
                    early_stop_count = 0
                else:
                    early_stop_count += 1

        if early_stop_count >= early_stop_patience:
            break

    baseline_net.load_state_dict(best_model_wts)
    baseline_net.eval()

    # Save model weights
    torch.save(baseline_net.state_dict(), model_path)

    return baseline_net


def baseline_reload(dataloaders, device, model_path):
    # Train baseline
    _, data = next(enumerate(dataloaders['train']))
    input = data[0]
    feature_size = input['input'][0,:].flatten().size(0)
    baseline_net = BaselineNet(feature_size, 500, 500)
    baseline_net.to(device)
    
    baseline_net.load_state_dict(torch.load(model_path))
    return baseline_net