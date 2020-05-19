import sys
import numpy as np
import librosa
import torch
from torchvision import transforms


class AudioRead():
    def __init__(self, sr=16000, offset=0.0, duration=None):
        self.sr = sr
        self.offset = offset
        self.duration = duration

    def __call__(self, x):
        y, _ = librosa.load(x, sr=self.sr, duration=self.duration,
                            offset=self.offset)
        return y


class Zscore():
    def __init__(self, divide_sigma=True):
        self.divide_sigma = divide_sigma

    def __call__(self, x):
        assert isinstance(x, np.ndarray)
        assert x.ndim <= 2
        x -= x.mean(axis=0)
        if self.divide_sigma:
            x /= x.std(axis=0)
        return x


class Spectrogram():
    def __init__(self, sr=16000, n_fft=1024, hop_size=160, n_band=128, fmax=10000, fmin=100,
                 spec_type='melspec'):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.n_band = n_band
        self.fmax = fmax
        self.fmin = fmin
        self.spec_type = spec_type

    def __call__(self, x):
        return spectrogram(x, self.sr, self.n_fft, self.hop_size, self.n_band, self.fmax, self.fmin,
                           self.spec_type)


def spectrogram(x, sr, n_fft, hop_size, n_band, fmax=10000, fmin=100, spec_type='melspec'):
    if spec_type == 'stft':
        S = librosa.core.stft(y=x, n_fft=n_fft, hop_length=hop_size)
        S = np.abs(S)
    elif spec_type == 'melspec':
        S = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=n_fft,
                                           hop_length=hop_size, n_mels=n_band, fmin=fmin, fmax=fmax)
    return S


class LogCompress():
    def __init__(self):
        self.factor = factor

    def __call__(self, x):
        return np.log(sys.float_info.epsilon + x)


class ChunkSpec():
    def __init__(self, duration=0.2, sr=16000, hop_size=160):
        self.context_size = int(sr * duration) // hop_size

    def __call__(self, x):
        assert isinstance(x, torch.Tensor)
        assert x.dim() == 2
        # MAKE SURE THE LAST DIM. IS TIME
        if x.size(1) % self.context_size != 0:
            y = torch.split(x, self.context_size, dim=1)[:-1]
        else:
            y = torch.split(x, self.context_size, dim=1)

        return torch.stack(y)


class Clipping():
    def __init__(self, clip_val=-100):
        self.clip_val = clip_val

    def __call__(self, x):
        x[x < self.clip_val] = self.clip_val
        return x


class TransposeNumpy():
    def __call__(self, x):
        assert isinstance(x, np.ndarray)
        return x.T


class ToTensor():
    def __call__(self, x):
        return totensor(x)


def totensor(x):
    if isinstance(x, list):
        y = [torch.from_numpy(y_i).type('torch.FloatTensor') for y_i in x]
    else:
        y = torch.from_numpy(x).type('torch.FloatTensor')
    return y


class Loadnp():
    def __call__(self, x):
        return np.load(x)


class LoadTensor():
    def __call__(self, x):
        return torch.load(x)


class MinMaxNorm():
    def __init__(self, min_val=0, max_val=1):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, x):
        if isinstance(x, list):
            return [minmaxNorm(x_i, min_val=self.min_val, max_val=self.max_val) for x_i in x]
        else:
            return minmaxNorm(x, min_val=self.min_val, max_val=self.max_val)


def minmaxNorm(x, min_val=0, max_val=1):
    x -= x.mean()
    x_min = x.min()
    x_max = x.max()
    nom = x - x_min
    den = x_max - x_min

    if abs(den) > 1e-4:
            return (max_val - min_val) * (nom / den) + min_val
    else:
        return nom


class PickFirstChunk:
    def __call__(self, x):
        return x[0, :, :].unsqueeze(0)


class PitchShift():
    def __init__(self, n_steps, sr=22050):
        self.n_steps = n_steps
        self.sr = sr

    def __call__(self, x):
        return librosa.effects.pitch_shift(x, sr=self.sr, n_steps=self.n_steps)
