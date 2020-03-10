import torch
import torch.nn as nn


class G1(nn.Module):  # expected input: randn(., 100, 1, 1)
    def __init__(self):
        super(G1, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

    def get_noise_batch(self, batch_labels, spiked=False):
        spike_val = 10
        noise = torch.randn(len(batch_labels), 100, 1, 1)
        if spiked:
            for i, label in enumerate(batch_labels):
                noise[i, label.to(int).item(), 0, 0] = spike_val
        return noise

class G2(nn.Module):  # expected input: randn(., 3, 8, 64)
    def __init__(self):
        super(G2, self).__init__()
        self.input_channels = 3
        self.input_height = 8
        self.input_width = 64
        self.main = nn.Sequential(
            nn.Conv2d(3, 24, (8, 1), (1, 1), 0, bias=False),
            nn.Conv2d(24, 24, (1, 3), (1, 1), 0, bias=False),
            nn.Conv2d(24, 48, (1, 5), (1, 2), 0, bias=False),
            nn.Conv2d(48, 96, (1, 10), (1, 2), 0, bias=False),
            nn.Conv2d(96, 100, (1, 10), (1, 1), 0, bias=False),
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

    def get_noise_batch(self, batch_labels, spiked=True):
        if self.input_width <= max(batch_labels):
            raise ('Error while creating noise_batch: Number of classes exceeds generator input height.')

        noise = torch.randn(len(batch_labels), self.input_channels, self.input_height, self.input_width)
        if spiked:
            for i in range(len(batch_labels)):
                noise[i, :, :, batch_labels[i].to(int).item()] = torch.ones(self.input_height)
        return noise


class G2a(nn.Module):    # expected input: randn(., 3, 10, 64)
    def __init__(self):
        super(G2a, self).__init__()
        self.input_channels = 3
        self.input_height = 10
        self.input_width = 64
        self.main = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=(3, 1), stride=(2, 1), bias=False),
            nn.Conv2d(8, 24, kernel_size=(3, 1), stride=(2, 1), bias=False),
            nn.Conv2d(24, 100, kernel_size=(1, 64), stride=(1, 1), bias=False),
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

    def get_noise_batch(self, batch_labels, spiked=True):
        spike_val = 10
        if self.input_width <= max(batch_labels):
            raise ('Error while creating noise_batch: Number of classes exceeds generator input height.')

        noise = torch.randn(len(batch_labels), self.input_channels, self.input_height, self.input_width)
        if spiked:
            for i in range(len(batch_labels)):
                noise[i, :, :, batch_labels[i].to(int).item()] = spike_val * torch.ones(self.input_height)
        return noise


class G3(nn.Module):
    def __init__(self, n_classes):
        super(G3, self).__init__()
        self.n_classes = n_classes
        self.input_channels = 100
        self.input_height = 1
        self.input_width = 1
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

    def get_noise_batch(self, batch_labels, spiked=True):
        noise = torch.randn(len(batch_labels), self.input_channels, self.input_height, self.input_width)
        if spiked:
            spike_width = (self.input_channels / self.n_classes).__int__()
            for i in range(len(batch_labels)):
                idx_start = batch_labels[i].to(int).item() * spike_width
                idx_end = (batch_labels[i].to(int).item() + 1) * spike_width
                noise[i, idx_start:idx_end, 0, 0] = torch.ones(spike_width)
        return noise
