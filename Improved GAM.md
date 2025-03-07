import torch
import torch.nn as nn

class GAM_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, rate=4):
        super(GAM_Attention, self).__init__()

        # Channel Attention
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, in_channels // rate),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // rate, in_channels)
        )

        # Replace the original 7x7 convolutions with three consecutive 3x3 convolutions
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // rate, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // rate),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels // rate, in_channels // rate, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // rate),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels // rate, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        # Apply channel attention
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)

        # Multiply the input by the channel attention
        x = x * x_channel_att

        # Apply spatial attention
        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att

        return out

if __name__ == '__main__':
    x = torch.randn(1, 64, 32, 48)
    b, c, h, w = x.shape
    net = GAM_Attention(in_channels=c, out_channels=c)
    y = net(x)
    print(y.shape)

