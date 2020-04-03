import torch.nn as nn #for layers
import torch.nn.functional as F #activation functions
import torch.optim as optim ## optimizer
import torch as T

class LinearClassification(nn.Module):
    def __init__(self, lr, n_classes, input_dims):
        super(LinearClassification, self).__init__()

        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128,256)
        self.fc3 = nn.Linear(256, n_classes)
        self.optimizer = optim.Adam(self.parameters(), lr=lr) #self.parameters() comes from nn.Module
        self.loss = nn.CrossEntropyLoss() #nn.MSELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)# send sself to device


    def forward(self, data):
        layer1 = F.sigmoid(self.fc1(data))
        layer2 = F.sigmoid(self.fc2(layer1))
        layer3 = self.fc3(layer2)

        return layer3


    def learn(self, data, labels):
        self.optimizer.zero_grad()
        data = T.tensor(data).to(self.device) #tensor preserves datatype, Tensor refreshes it
        labels = T.tensor(labels).to(self.device)

        predictions = self.forward(data)

        cost = self.loss(predictions, labels)

        cost.backward()
        self.optimizer.step()
