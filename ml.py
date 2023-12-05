import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from dataloading import CustomImageDataset


class Net(nn.Module):
    def __init__(self, nf, nl, threshold=0):
        super().__init__()
        self.nf = nf
        self.nl = {8:1, 16:2, 32:3, 64:4, 128:5}.get(nl)
        self.set_threshold(threshold)
        self.convs = nn.ModuleList()
        n = (1, 4)
        for i in range(self.nl):
            self.convs.append(nn.Conv2d(n[0], n[1], (self.nf, 3), padding="same"))
            n = (n[1], n[1]*2)
        self.conv_valid = nn.Conv2d(n[0], n[1], (self.nf, 4), padding="valid")
        self.pool = nn.AvgPool2d((1, 2), (1, 2))
        self.fc = nn.Linear(n[1], 2)
        self.dropout = nn.Dropout1d(0.75)
        self.softmax = nn.Softmax()
        # self.c1 = nn.Conv2d(1, 4,  (3, 3), padding=1, padding_mode="zeros")
        # self.c2 = nn.Conv2d(4, 8,  (3, 3), padding=1, padding_mode="zeros")
        # self.c3 = nn.Conv2d(8, 16, (3, 3), padding=1, padding_mode="zeros")
        # self.convs = [self.c1, self.c2, self.c3]

    def forward(self, x):
        for conv in self.convs: 
            x = self.pool(F.relu(conv(x)))
        x = F.relu(self.conv_valid(x))
        x = torch.flatten(x, 1)
        # x = self.dropout(x)
        x = self.softmax(self.fc(x))
        return x
        # return self.c3(self.c2(self.c1(x)))
    
    def set_threshold(self, threshold):
        self.threshold = nn.Parameter(torch.FloatTensor([threshold]), requires_grad=False)
    
    def forward_thresholded(self, x):
        return (self.forward(x).squeeze() > self.threshold).cpu().numpy()
    

def train(X_train, y_train, X_test, y_test, batch_size=1, calc_test=True, device="cuda"):
    # class_sample_count = [10, 1, 20, 3, 4] # dataset has 10 class-1 samples, 1 class-2 samples, etc.
    # weights = 1 / torch.Tensor(class_sample_count)
    # weights = weights.double()
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, batch_size)
    trainloader = torch.utils.data.DataLoader(CustomImageDataset(X_train, y_train), 
                                              batch_size=batch_size, 
                                              shuffle=True,
                                              )
    
    model = Net(X_train.shape[2], X_train.shape[3]).to(device) #32-3, 16-2, 8-1
    # print(summary(model, (1, 6, 32)))
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            labels = labels.to(device)
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(labels.sum(0)[[1, 0]]))
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.float().to(device))
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        if calc_test:
            loss_test = criterion(model(torch.tensor(X_test).float().to(device)), 
                                        torch.tensor(y_test).float().to(device))
            print(f"[{epoch + 1:03d}, {i + 1:5d}] loss train: {running_loss / (i + 1):.4f} | test: {loss_test / (i + 1):.4f}")
        running_loss = 0.0
    return model
    
    
if __name__ == "__main__":
    import numpy as np
    device = "cpu"
    # model = Net(7, 32).to(device).double()
    # model.eval()
    # # print(summary(model, (10, 1, 7, 32)))
    # x = torch.tensor(np.zeros((1, 1, 7, 32))).double().to(device)
    # model(x)
    # with torch.no_grad():
    #     for _ in range(10):
    #         print(model(x))
            
            
            
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(1, 4,  (3, 3), padding=1, padding_mode="zeros")
            self.c2 = nn.Conv2d(4, 8,  (3, 3), padding=1, padding_mode="zeros")
            self.c3 = nn.Conv2d(8, 16,  (3, 3), padding=1, padding_mode="zeros")
            self.c4 = nn.Conv2d(16, 32,  (3, 3), padding=1, padding_mode="zeros")
            self.convs = [self.c1, self.c2, self.c3, self.c4]

        def forward(self, x):
            for conv in self.convs: 
                x = conv(x)
            return x

    model = Net().to(device)
    # print(summary(model, (10, 1, 32, 32)))
    x = torch.tensor(np.zeros((1, 1, 32, 32))).float().to(device)
    model(x)
    with torch.no_grad():
        for _ in range(10):
            print(model(x).cpu().sum())