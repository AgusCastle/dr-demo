import torch
import torch.nn as nn


class SparseFusion(nn.Module):
    def __init__(self, n_classes, device) -> None:
        super(SparseFusion, self).__init__()
        W = torch.randn(9, 5, dtype=torch.float32)
        W = W.to(device)
        self.W = nn.Parameter(W)

        self.i = torch.eye(n_classes)
        self.i = self.i.to(device)

        W2 = torch.randn(5, dtype=torch.float32)
        W2 = W2.to(device)
        self.W2 = nn.Parameter(W2)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.matmul(x, self.W)
        x = torch.mul(x, self.i)
        x = torch.diagonal(x, dim1= -1, dim2= -2)
        x = torch.mul(x, self.W2)
        #return self.softmax((torch.diagonal((x @ self.W) * self.i, dim1= -1, dim2=-2)) * self.W2)
        return self.softmax(x)