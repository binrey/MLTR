from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary

# class CustomImageDataset(Dataset):
#     def __init__(self, X, y):
#         self.img_labels = y
#         self.imgs = X

#     def __len__(self):
#         return len(self.img_labels)

#     def __getitem__(self, idx):
#         image = self.imgs[idx]
#         label = self.img_labels[idx]
#         return image, label


class Net(nn.Module):
    def __init__(self, nf, nl, threshold=0):
        super().__init__()
        self.nf = nf
        self.nl = {8: 1, 16: 2, 32: 3, 64: 4, 128: 5}.get(nl)
        self.set_threshold(threshold)
        self.f = nn.Sequential()
        n = (1, 4)
        for i in range(self.nl):
            self.f.append(nn.Conv2d(n[0], n[1], (self.nf, 3), padding="same"))
            self.f.append(nn.BatchNorm2d(n[1]))
            self.f.append(nn.ReLU())
            self.f.append(nn.MaxPool2d((1, 2), (1, 2)))
            n = (n[1], n[1]*2)
        self.conv_valid = nn.Conv2d(n[0], n[1], (self.nf, 4), padding="valid")
        self.fc_scale = nn.Linear(2, n[1])
        self.fc = nn.Linear(n[1], 2)
        self.dropout = nn.Dropout1d(0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        p = F.relu(self.fc_scale(x[:, 0, -2:, 0]))
        x = self.f(x[:, :, :-2, :])
        # x = F.relu(self.conv_valid(x))
        # x = torch.flatten(x, 1)
        # x = self.dropout(x)
        # x = x + p
        # x = self.softmax(self.fc(x))
        return x

    def set_threshold(self, threshold):
        self.threshold = nn.Parameter(
            torch.FloatTensor([threshold]), requires_grad=False)

    def forward_thresholded(self, x):
        return (self.forward(x).squeeze() > self.threshold).cpu().numpy()


class Net2(nn.Module):
    def __init__(self, nf, nl, threshold=0):
        super().__init__()
        self.nf = nf
        self.nl = nl
        nl = {8: 1, 16: 2, 32: 3, 64: 4, 128: 5}.get(nl)
        self.set_threshold(threshold)
        self.f = nn.Sequential()
        n = (4, 4)
        for i in range(nl):
            self.f.append(nn.Conv2d(n[0], n[1], (self.nf, 3), padding="same"))
            self.f.append(nn.BatchNorm2d(n[1]))
            self.f.append(nn.ReLU())
            self.f.append(nn.MaxPool2d((1, 2), (1, 2)))
            n = (n[1], n[1]*2)
        self.conv_valid = nn.Conv2d(n[0], 2, (self.nf, 4), padding="valid")
        self.fc_scale = nn.Linear(2, 2)
        self.fc = nn.Linear(n[1], 2)
        self.dropout = nn.Dropout2d(0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # p = self.fc_scale(x[:, 0, -2:, 0])
        ma8 = nn.AvgPool2d((1, 8), (1, 1))(x)
        ma16 = nn.AvgPool2d((1, 16), (1, 1))(x)
        ma32 = nn.AvgPool2d((1, 32), (1, 1))(x)
        x = torch.concat((x[:, :, :, -self.nl:], ma8[:, :, :, -self.nl:],
                         ma16[:, :, :, -self.nl:], ma32[:, :, :, -self.nl:]), dim=1)
        # x = x[:, :, :, :self.nl]
        x = self.f(x)
        x = self.dropout(x)
        x = self.conv_valid(x)
        x = torch.flatten(x, 1)
        x = self.softmax(x)
        return x

    def set_threshold(self, threshold):
        self.threshold = nn.Parameter(
            torch.FloatTensor([threshold]), requires_grad=False)

    def forward_thresholded(self, x):
        return (self.forward(x).squeeze() > self.threshold).cpu().numpy()


def train(X_train, y_train, X_test, y_test, batch_size=1, epochs=4, calc_test=True, device="cuda"):
    trainloader = torch.utils.data.DataLoader(CustomImageDataset(X_train, y_train),
                                              batch_size=batch_size,
                                              shuffle=True
                                              )

    model = Net2(X_train.shape[2], 32).to(device).float()  # 32-3, 16-2, 8-1
    if calc_test:
        y_test_tensor = torch.tensor(y_test).float().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(
        weight=10000/torch.tensor(y_train).float().to(device).sum(0))
    loss_hist = np.zeros((epochs, 2))
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0
        running_roc_train = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            labels = labels.to(device)
            outputs = model(inputs.float().to(device))
            # roc_train = 0
            # if calc_test:
            #     roc_train = roc_auc_score(labels[:, 0].cpu().numpy(), outputs[:, 0].cpu().detach().numpy())
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            loss = criterion(labels, outputs)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            # running_roc_train += roc_train

        loss_test = 0
        if calc_test:
            assert X_test.shape[0] > 0, "No test data"
            test_out = model(X_test)
            roc_test = roc_auc_score(
                y_test[:, 0], test_out[:, 0].cpu().detach().numpy())
            loss_test = criterion(
                y_test_tensor, test_out).detach().cpu().numpy()
            print(f"{epoch + 1:03d} loss train: {running_loss/(i+1):.4f} | test: {loss_test:.4f}   ROC train: {running_roc_train/(i+1):.4f} | test: {roc_test:.4f}")
            loss_hist[epoch] = np.array([running_loss/(i+1), loss_test/3])
        running_loss = 0.0
        # if loss_hist[epoch, 0] <= loss_hist[epoch, 1]:
        #     break
    return model, loss_hist[:epoch+1]


class E2EModel(nn.Module):
    def __init__(self, n_indicators, n_features, nh, cls_head=False):
        super(E2EModel, self).__init__()
        self.nh = nh
        self.ni = n_indicators
        self.nf = n_features
        self.train_info = {}

        nout = 3 if cls_head else 1
        # self.features_merge = nn.Conv1d(in_channels=self.nf, out_channels=self.nf, kernel_size=self.ni, stride=self.ni)
        self.features_merge = nn.Linear(self.ni, 1)
        self.fc_features_in = nn.Linear(self.nf, nh)
        self.fc_out_prev_in = nn.Linear(1, nh)
        self.fc_hid = nn.Linear(nh, nh)
        self.fc_out = nn.Linear(nh*self.ni, nout)

        self.norm_hid = nn.LayerNorm(nh)
        self.norm_in = nn.LayerNorm(self.nf)

        self.norm_out = nn.LayerNorm(nout)
        self.relu = nn.ReLU()
        self.out_func = self.cls_head if cls_head else nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)
        self.states = torch.tensor([-1, 0, 1])
        self.flatten = nn.Flatten()

    def cls_head(self, x):
        x = self.softmax(x)
        return (self.states*x).sum(dim=1)

    def forward(self, x):
        # x = self.norm_in(x)
        # x = x.permute(0, 2, 1)
        # x = self.features_merge(x)
        # x = x.permute(0, 2, 1)
        # x = x[:, 0, :]
        x = self.fc_features_in(x)
        x = self.relu(x)
        
        x = self.dropout(x)         
        x = self.fc_hid(x)
        x = self.relu(x)
 
        x = self.flatten(x)                
        x = self.fc_out(x)
        output = self.out_func(x)
        
        return output


class E2EModelConv(nn.Module):
    def __init__(self, n_indicators, n_features, nh, cls_head=False):
        super(E2EModelConv, self).__init__()
        self.nh = nh
        self.ni = n_indicators
        self.nf = n_features
        self.train_info = {}

        nout = 3 if cls_head else 1
        self.features_merge = nn.Conv1d(in_channels=self.nf, out_channels=self.nh, kernel_size=self.ni, stride=self.ni)
        self.fc_features_in = nn.Linear(self.nf, nh)
        self.fc_out_prev_in = nn.Linear(1, nh)
        self.fc_hid = nn.Linear(nh, nh)
        self.fc_out = nn.Linear(nh, nout)

        self.norm_hid = nn.LayerNorm(nh)
        self.norm_in = nn.LayerNorm(self.nf)

        self.norm_out = nn.LayerNorm(nout)
        self.relu = nn.ReLU()
        self.out_func = self.cls_head if cls_head else nn.Tanh()
        self.dropout = nn.Dropout(0.25)
        self.softmax = nn.Softmax(dim=1)
        self.states = torch.tensor([-1, 0, 1])
        self.flatten = nn.Flatten()

    def cls_head(self, x):
        x = self.softmax(x)
        return (self.states*x).sum(dim=1)

    def forward(self, x):
        # x = self.norm_in(x)
        x = x.permute(0, 2, 1)
        x = self.features_merge(x)
        x = x.permute(0, 2, 1)
        # x = self.fc_features_in(x)
        x = self.relu(x)
        
        x = self.fc_hid(x)
        x = self.relu(x)
        x = self.dropout(x)      
 
        x = self.flatten(x)                
        x = self.fc_out(x)
        output = self.out_func(x)
        
        return output
    
@dataclass
class SeqOutput:
    model_ans: torch.Tensor
    price: torch.Tensor
    fees: torch.Tensor
    
    @property
    def profits(self):
        return self.model_ans[:-1] * (self.price[1:] - self.price[:-1])
    
    @property
    def profits_with_fees(self):
        return self.profits - self.fees
    
    @property
    def profits_with_fees_relative(self):
        return self.profits_with_fees / self.price[:-1]
    
    def sum_profit(self, include_fees=True):
        profit = self.profits.sum(dim=0)
        if include_fees:
            profit -= self.sum_fees()
        return profit
      
    def sum_profit_relative(self):
        return self.profits_with_fees_relative.sum(dim=0)
    
    def sum_fees(self):
        return self.fees.sum(dim=0)
    
    def profit_curve(self, relative=True, include_fees=True):
        if relative and include_fees:
            profit_curve = self.profits_with_fees_relative.cumsum(dim=0)
        if not relative:
            profit_curve = self.profits.cumsum(dim=0)
            if include_fees:
                profit_curve -= self.fees.cumsum(dim=0)
        return torch.concat([torch.Tensor([0]), profit_curve.cpu()])



if __name__ == "__main__":
    from time import time

    import numpy as np
    ni = 2
    nf = 64
    
    torch.manual_seed(0)
    device = torch.device("cpu")
    f = torch.randn(10000, ni, nf).to(device)

    model = E2EModel(ni, nf, 8)
    model.to(device)
    out = model(f)
    summary(model, (1, ni, nf), device="cpu")
    print(out.shape, out, "\n")
    t0 = time()
    for i in range(100):
        out = model(f)
    print(time()-t0)
