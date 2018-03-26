import torch
import librosa
import numpy as np


def get_power_loss_torch(y, y1, n_fft=512, hop_length=256, cuda=True):
    batch = y.size(0)
    x = y.view(batch, -1)
    x1 = y1.view(batch, -1)
    s = torch.stft(x, n_fft, hop_length)
    s1 = torch.stft(x1, n_fft, hop_length)
    ss = torch.log(torch.sqrt(s[:, :, :, 0] ** 2 + s[:, :, :, 1] ** 2)) - torch.log(torch.sqrt(s1[:, :, :, 0] ** 2 + s1[:, :, :, 1] ** 2))
    return torch.mean(ss ** 2)


def to_tensor(x):
    x = torch.from_numpy(x).float()
    return x.view(1, -1)


y, sr = librosa.load(librosa.util.example_audio_file())
y = y[2000:10000]
y1 = y + np.random.randn(*y.shape)*0.0001
y, y1 = to_tensor(y), to_tensor(y1)

loss = get_power_loss_torch(y, y1)
print(loss)
