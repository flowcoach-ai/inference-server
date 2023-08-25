from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim * 2)
        self.fc2 = nn.Linear(input_dim * 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, int(hidden_dim / 2))
        self.fc4 = nn.Linear(int(hidden_dim / 2), int(hidden_dim / 4))
        self.fc5 = nn.Linear(int(hidden_dim / 4), int(hidden_dim / 8))
        self.fc6 = nn.Linear(int(hidden_dim / 8), num_classes)

    def forward(self, x_in):
        z = F.relu(self.fc1(x_in))  # ReLU activation function added!
        z = F.relu(self.fc2(z))
        z = F.relu(self.fc3(z))
        z = F.relu(self.fc4(z))
        z = F.relu(self.fc5(z))
        z = self.fc6(z)
        return z
