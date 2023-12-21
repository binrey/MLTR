import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from dataloading import CustomImageDataset
from sklearn.metrics import roc_auc_score


class Net(nn.Module):
    def __init__(self, nf, nl, threshold=0):
        super().__init__()
        self.nf = nf
        self.nl = {8:1, 16:2, 32:3, 64:4, 128:5}.get(nl)
        self.set_threshold(threshold)
        self.f = nn.Sequential()
        n = (1, 4)
        for i in range(self.nl):
            self.f.append(nn.Conv2d(n[0], n[1], (self.nf, 3), padding="same"))
            # self.f.append(nn.BatchNorm2d(n[1]))
            self.f.append(nn.ReLU())
            self.f.append(nn.MaxPool2d((1, 2), (1, 2)))
            n = (n[1], n[1]*2)
        self.conv_valid = nn.Conv2d(n[0], n[1], (self.nf, 4), padding="valid")
        self.fc_scale = nn.Linear(2, 2)
        self.fc = nn.Linear(n[1], 3)
        self.dropout = nn.Dropout1d(0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # p = self.fc_scale(x[:, 0, -2:, 0])
        x = self.f(x[:, :, :, :])
        x = F.relu(self.conv_valid(x))
        x = torch.flatten(x, 1)
        # x = self.dropout(x)
        # x = torch.concat([x, p], 1)
        x = self.softmax(self.fc(x))
        return x
        # return self.c3(self.c2(self.c1(x)))
    
    def set_threshold(self, threshold):
        self.threshold = nn.Parameter(torch.FloatTensor([threshold]), requires_grad=False)
    
    def predict(self, x):
        return (self.forward(x).argmax(1)).cpu().numpy()
    

def train(X_train, y_train, X_test, y_test, batch_size=1, epochs=4, calc_test=True, device="cuda"):
    y_train /= y_train.sum(0)
    y_test /= y_test.sum(0)
    trainloader = torch.utils.data.DataLoader(CustomImageDataset(X_train, y_train), 
                                              batch_size=batch_size, 
                                              shuffle=True
                                              )
    # costs = torch.FloatTensor([[1, 1/2, 1/3]]*y_train.shape[0]).to(device)
    model = Net(X_train.shape[2], X_train.shape[3]).to(device) #32-3, 16-2, 8-1
    if calc_test:
        y_test_tensor = torch.tensor(y_test).float().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # criterion = nn.CrossEntropyLoss(weight=100/torch.tensor(y_train).float().to(device).sum(0))
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
            loss = -((outputs*labels)).sum() - outputs.std(0).mean()*3
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            # running_roc_train += roc_train
            
        loss_test = -(model(X_test).cpu().detach().numpy()*y_test**2).sum()
        print(f"{epoch + 1:03d} loss train: {running_loss/(i+1):.4f} | test: {loss_test:.4f}")
        # if calc_test:
        #     test_out = model(X_test)
        #     roc_test = roc_auc_score(y_test[:, 0], test_out[:, 0].cpu().detach().numpy())
        #     loss_test = criterion(y_test_tensor, test_out)
        #     print(f"{epoch + 1:03d} loss train: {running_loss/(i+1):.4f} | test: {loss_test:.4f}   ROC train: {running_roc_train/(i+1):.4f} | test: {roc_test:.4f}")
        running_loss = 0.0
    return model
    
    
if __name__ == "__main__":
    import numpy as np
    torch.manual_seed(0)
    device = "cpu"
    model = Net(4, 64)
    model.eval()
    print(summary(model, (10, 1, 4, 64)))
    model.to(device)
    x = torch.tensor(np.zeros((10, 1, 4, 64))).float().to(device)
    print(model(x))
    
    
    def train(X_train, y_train, X_test, y_test, batch_size=1, epochs=4, calc_test=True, device="cuda"):
        trainloader = torch.utils.data.DataLoader(CustomImageDataset(X_train, y_train), 
                                                batch_size=batch_size, 
                                                shuffle=True
                                                )
        
        model = Net(X_train.shape[2]-2, X_train.shape[3]).to(device) #32-3, 16-2, 8-1
        if calc_test:
            y_test_tensor = torch.tensor(y_test).float().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.1)
        criterion = nn.MSELoss()#weight=torch.tensor(y_train).float().to(device).sum(0)[[1, 0]])
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0
            running_roc_train = 0
            
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                labels = labels.to(device)
                
                outputs = model(inputs.float().to(device))
                roc_train = 0
                if calc_test:
                    roc_train = roc_auc_score(labels[:, 0].cpu().numpy()>0, outputs[:, 0].cpu().detach().numpy())
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                loss = -((outputs[:, :1]*labels).sum() - labels.sum()*outputs[:, :1].mean())
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
                running_roc_train += roc_train
            if calc_test:
                test_out = model(X_test)
                roc_test = roc_auc_score(y_test[:, 0]>0, test_out[:, 0].cpu().detach().numpy())
                loss_test = -((test_out[:, :1]*y_test_tensor).sum() - y_test_tensor.sum()*test_out[:, :1].mean())
                print(f"{epoch + 1:03d} loss train: {running_loss/(i+1):.4f} | test: {loss_test:.4f}   ROC train: {running_roc_train/(i+1):.4f} | test: {roc_test:.4f}")
            running_loss = 0.0
        return model