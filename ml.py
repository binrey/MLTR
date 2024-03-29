import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchinfo import summary
from dataloading import CustomImageDataset
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset
import numpy as np


class CustomImageDataset(Dataset):
    def __init__(self, X, y):
        self.img_labels = y
        self.imgs = X

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.imgs[idx]
        label = self.img_labels[idx]
        return image, label


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
        self.threshold = nn.Parameter(torch.FloatTensor([threshold]), requires_grad=False)
    
    def forward_thresholded(self, x):
        return (self.forward(x).squeeze() > self.threshold).cpu().numpy()
    
    
class Net2(nn.Module):
    def __init__(self, nf, nl, threshold=0):
        super().__init__()
        self.nf = nf
        self.nl = nl
        nl = {8:1, 16:2, 32:3, 64:4, 128:5}.get(nl)
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
        x = torch.concat((x[:, :, :, -self.nl:], ma8[:, :, :, -self.nl:], ma16[:, :, :, -self.nl:], ma32[:, :, :, -self.nl:]), dim=1)
        #x = x[:, :, :, :self.nl]
        x = self.f(x)
        x = self.dropout(x)
        x = self.conv_valid(x)
        x = torch.flatten(x, 1)
        x = self.softmax(x)
        return x
    
    def set_threshold(self, threshold):
        self.threshold = nn.Parameter(torch.FloatTensor([threshold]), requires_grad=False)
    
    def forward_thresholded(self, x):
        return (self.forward(x).squeeze() > self.threshold).cpu().numpy()
    
def train(X_train, y_train, X_test, y_test, batch_size=1, epochs=4, calc_test=True, device="cuda"):
    trainloader = torch.utils.data.DataLoader(CustomImageDataset(X_train, y_train), 
                                              batch_size=batch_size, 
                                              shuffle=True
                                              )
    
    model = Net2(X_train.shape[2], 32).to(device).float() #32-3, 16-2, 8-1
    if calc_test:
        y_test_tensor = torch.tensor(y_test).float().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(weight=10000/torch.tensor(y_train).float().to(device).sum(0))
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
            roc_test = roc_auc_score(y_test[:, 0], test_out[:, 0].cpu().detach().numpy())
            loss_test = criterion(y_test_tensor, test_out).detach().cpu().numpy()
            print(f"{epoch + 1:03d} loss train: {running_loss/(i+1):.4f} | test: {loss_test:.4f}   ROC train: {running_roc_train/(i+1):.4f} | test: {roc_test:.4f}")
            loss_hist[epoch] = np.array([running_loss/(i+1), loss_test/3])
        running_loss = 0.0
        # if loss_hist[epoch, 0] <= loss_hist[epoch, 1]:
        #     break  
    return model, loss_hist[:epoch+1]
    

if __name__ == "__main__":
    import numpy as np
    torch.manual_seed(0)
    device = "cpu"
    model = Net2(4, 32)
    model.eval()
    print(summary(model, (10, 1, 4, 64)))
    # model.to(device)
    # x = torch.tensor(np.zeros((1, 1, 6, 64))).float().to(device)
    # with torch.no_grad():
    #     for _ in range(3):
    #         print(model(x))