# coding: utf-8
"""
Synthesis waveform from trained WaveNet.

usage: synthesis.py [options] <checkpoint> <dst_dir>

options:
    --hparams=<parmas>                Hyper parameters [default: ].
    --length=<T>                      Steps to generate [default: 32000].
    --initial-value=<n>               Initial value for the WaveNet decoder.
    --conditional=<p>                 Conditional features path.
    --file-name-suffix=<s>            File name suffix [default: ].
    --speaker-id=<id>                 Speaker ID (for multi-speaker model).
    --output-html                     Output html for blog post.
    -h, --help               Show help message.
"""
from docopt import docopt

import sys
import os
from os.path import dirname, join, basename, splitext
import torch
from torch.autograd import Variable
import numpy as np
from nnmnkwii import preprocessing as P
from keras.utils import np_utils
from tqdm import tqdm
import librosa

from wavenet_vocoder.util import is_mulaw_quantize, is_mulaw, is_raw

import audio
from hparams import hparams

use_cuda = torch.cuda.is_available()


def _to_numpy(x):
    # this is ugly
    if x is None:
        return None
    if isinstance(x, np.ndarray) or np.isscalar(x):
        return x
    # remove batch axis
    if x.dim() == 3:
        x = x.squeeze(0)
    return x.numpy()


def extract_mel_condition(wav_file_path, sample_rate=hparams.sample_rate):
    wav, sr = librosa.load(wav_file_path, sample_rate)
    c = audio.melspectrogram(wav)
    return wav,c


def wavegen(model, length=None, c=None, g=None, initial_value=None, fast=False, tqdm=tqdm,current_gpu=1):
    """Generate waveform samples by WaveNet.

    Args:
        model (nn.Module) : WaveNet decoder
        length (int): Time steps to generate. If conditinlal features are given,
          then this is determined by the feature size.
        c (numpy.ndarray): Conditional features, of shape T x C
        g (scaler): Speaker ID
        initial_value (int) : initial_value for the WaveNet decoder.
        fast (Bool): Whether to remove weight normalization or not.
        tqdm (lambda): tqdm

    Returns:
        numpy.ndarray : Generated waveform samples
    """
    from train import sanity_check
    sanity_check(model, c, g)

    if use_cuda:
        model = model.cuda(current_gpu)
    model.eval()
    T = c.size(-1)
    u = Variable(torch.zeros(1,1,length).uniform_(1e-5, 1 - 1e-5), requires_grad=False).cuda(current_gpu)
    z = torch.log(u) - torch.log(1 - u)
    predict, mu, scale = model(z, c=c, g=g, softmax=False)
    wave = predict.data.cpu().numpy()
    return wave




if __name__ == "__main__":
    # args = docopt(__doc__)
    args = {
        '--file-name-suffix': '',
        '--output-html': '',
        '--speaker-id': None,
        '--length': '24000',
        '--hparams': "cin_channels=80,gin_channels=-1",
        '--initial-value': None,
        '--conditional': './data/ljspeech/ljspeech-mel-02183.npy',
        '--gpu_index': 1
    }
    print("Command line args:\n", args)
    # checkpoint_path = args["<checkpoint>"]
    checkpoint_path = './checkpoints_student/checkpoint_step000593000.pth'
    # dst_dir = args["<dst_dir>"]
    dst_dir = './generate'
    # length = int(args["--length"])
    length = 32000
    initial_value = args["--initial-value"]
    initial_value = None if initial_value is None else float(initial_value)
    conditional_path = args["--conditional"]
    file_name_suffix = args["--file-name-suffix"]
    output_html = args["--output-html"]
    speaker_id = args["--speaker-id"]
    speaker_id = None if speaker_id is None else int(speaker_id)
    current_gpu = args['--gpu_index'] if not args['--gpu_index'] else 1
    # Override hyper parameters
    hparams.parse(args["--hparams"])
    assert hparams.name == "wavenet_vocoder"
    os.makedirs(dst_dir, exist_ok=True)
    # Presets
    if hparams.preset is not None and hparams.preset != "":
        preset = hparams.presets[hparams.preset]
        import json

        hparams.parse_json(json.dumps(preset))
        print("Override hyper parameters with preset \"{}\": {}".format(
            hparams.preset, json.dumps(preset, indent=4)))
    # Load conditional features
    if conditional_path is not None:
        c = np.load(conditional_path)
        wave_path = conditional_path.replace('mel','audio')
        wav_target = np.load(wave_path)
        length = wav_target.shape[0]
        # x,c = audio.adjast_time_resolution(wav_target,c)
        T,C = c.shape
        c = torch.from_numpy(c.transpose().reshape(1,C,T))
        c = Variable(c,requires_grad=False).cuda(current_gpu) if use_cuda else Variable(c,requires_grad=False)
    else:
        c = None
        raise Exception("condition can't be null")

    from train_student import build_model
    import matplotlib.pyplot as plt

    # Model
    model = build_model('student')
    model.gpu = current_gpu

    # Load checkpoint
    print("Load checkpoint from {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    checkpoint_name = splitext(basename(checkpoint_path))[0]
    dst_gen_path = join(dst_dir, "{}{}_gen.wav".format(checkpoint_name, file_name_suffix))
    dst_tgt_path = join(dst_dir, "{}{}_tgt.wav".format(checkpoint_name, file_name_suffix))
    dst_img_path = join(dst_dir, "{}{}.png".format(checkpoint_name, file_name_suffix))

    # DO generate
    waveform = wavegen(model, length=length, c=c, g=speaker_id, initial_value=initial_value, fast=True)
    wave_gen = waveform.reshape(-1)
    # save
    librosa.output.write_wav(dst_gen_path, wave_gen, sr=hparams.sample_rate)
    librosa.output.write_wav(dst_tgt_path, wav_target, sr=hparams.sample_rate)
    plt.figure(figsize=(16, 6))
    plt.subplot(2, 1, 1)
    plt.title('generate')
    librosa.display.waveplot(wave_gen, sr=hparams.sample_rate)
    plt.subplot(2, 1, 2)
    plt.title('target')
    librosa.display.waveplot(wav_target, sr=hparams.sample_rate)
    plt.tight_layout()
    plt.savefig(dst_img_path, format="png")
    plt.close()
    print("Finished! Check out {} for generated audio samples.".format(dst_dir))
    sys.exit(0)
