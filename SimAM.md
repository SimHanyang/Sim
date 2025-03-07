import torch
import torch.nn as nn
 
class SimAM(nn.Module):
    def __init__(self, lamda=1e-5):
        super().__init__()
        self.lamda = lamda
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
                b, c, h, w = x.shape
               n = h * w - 1
              mean = torch.mean(x, dim=[-2,-1], keepdim=True)
              var = torch.sum(torch.pow((x - mean), 2), dim=[-2, -1], keepdim=True) / n
                e_t = torch.pow((x - mean), 2) / (4 * (var + self.lamda)) + 0.5
               out = self.sigmoid(e_t) * x
        return out
 
if __name__ == "__main__":
       layer = SimAM(lamda=1e-5)
    x = torch.randn((2, 3, 224, 224))
    output = layer(x)
print("Output shape:", output.shape)
