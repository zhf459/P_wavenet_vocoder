"""Trainining script for WaveNet vocoder

usage: train.py [options]

options:
    --data-root=<dir>            Directory contains preprocessed features.
    --checkpoint-dir=<dir>       Directory where to save model checkpoints [default: checkpoints].
    --hparams=<parmas>           Hyper parameters [default: ].
    --checkpoint=<path>          Restore model from checkpoint path if given.
    --restore-parts=<path>       Restore part of the model.
    --log-event-path=<name>      Log event path.
    --reset-optimizer            Reset optimizer.
    --speaker-id=<N>             Use specific speaker of data in case for multi-speaker datasets.
    -h, --help                   Show this help message and exit
"""
from docopt import docopt

import sys
from os.path import dirname, join
from tqdm import tqdm, trange
from datetime import datetime

from wavenet_vocoder import builder
import lrschedule

import torch
from torch.utils import data as data_utils
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
from torch.utils.data.sampler import Sampler
import numpy as np

from nnmnkwii import preprocessing as P
from nnmnkwii.datasets import FileSourceDataset, FileDataSource

from os.path import join, expanduser
import random
import librosa.display
from matplotlib import pyplot as plt
import sys
import os

from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from tensorboardX import SummaryWriter
from matplotlib import cm
from warnings import warn
from train_utils import PartialyRandomizedSimilarTimeLengthSampler, PyTorchDataset
from train_utils import *
from wavenet_vocoder.student_wavenet import StudentWaveNet
from wavenet_vocoder.util import is_mulaw_quantize, is_mulaw, is_raw, is_scalar_input
from wavenet_vocoder.mixture import discretized_mix_logistic_loss, probs_logistic
from wavenet_vocoder.mixture import sample_from_discretized_mix_logistic
import pickle
import audio
from hparams import hparams, hparams_debug_string
from scipy.stats import logistic

fs = hparams.sample_rate

global_step = 0
global_test_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False


# https://discuss.pytorch.org/t/how-to-apply-exponential-moving-average-decay-for-variables/10856/4
# https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
class ExponentialMovingAverage(object):
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def update(self, name, x):
        assert name in self.shadow
        update_delta = self.shadow[name] - x
        self.shadow[name] -= (1.0 - self.decay) * update_delta


def clone_as_averaged_model(model, ema):
    assert ema is not None
    averaged_model = build_model()
    if use_cuda:
        averaged_model = averaged_model.cuda()
    averaged_model.load_state_dict(model.state_dict())
    for name, param in averaged_model.named_parameters():
        if name in ema.shadow:
            param.data = ema.shadow[name].clone()
    return averaged_model


def ensure_divisible(length, divisible_by=256, lower=True):
    if length % divisible_by == 0:
        return length
    if lower:
        return length - length % divisible_by
    else:
        return length + (divisible_by - length % divisible_by)


def assert_ready_for_upsampling(x, c):
    assert len(x) % len(c) == 0 and len(x) // len(c) == audio.get_hop_size()


def time_string():
    return datetime.now().strftime('%Y-%m-%d %H:%M')


def get_power_loss(sample_, x_):
    batch = sample_.shape[0]
    T = sample_.shape[1]
    s = 0
    for i in range(batch):
        ss = np.abs(librosa.stft(sample_[i][0])) ** 2 - np.abs(librosa.stft(x_[i][0])) ** 2
        s += np.sum((ss / ss.shape[0]) ** 2)
    return s / batch


def to_numpy(x):
    return x.cpu().data.numpy()


def eval_model(global_step, writer, model, y, c, g, input_lengths, eval_dir, ema=None):
    if ema is not None:
        print("Using averaged model for evaluation")
        model = clone_as_averaged_model(model, ema)

    model.eval()
    idx = np.random.randint(0, len(y))
    length = input_lengths[idx].data.cpu().numpy()[0]

    # (T,)
    y_target = y[idx].view(-1).data.cpu().numpy()[:length]

    if c is not None:
        c = c[idx, :, :length].unsqueeze(0)
        assert c.dim() == 3
        print("Shape of local conditioning features: {}".format(c.size()))
    if g is not None:
        # TODO: test
        g = g[idx]
        print("Shape of global conditioning features: {}".format(g.size()))

    # Dummy silence
    if is_mulaw_quantize(hparams.input_type):
        initial_value = P.mulaw_quantize(0, hparams.quantize_channels)
    elif is_mulaw(hparams.input_type):
        initial_value = P.mulaw(0.0, hparams.quantize_channels)
    else:
        initial_value = 0.0
    print("Intial value:", initial_value)

    # (C,)
    if is_mulaw_quantize(hparams.input_type):
        initial_input = np_utils.to_categorical(
            initial_value, num_classes=hparams.quantize_channels).astype(np.float32)
        initial_input = Variable(torch.from_numpy(initial_input)).view(
            1, 1, hparams.quantize_channels)
    else:
        initial_input = Variable(torch.zeros(1, 1, 1).fill_(initial_value))
    initial_input = initial_input.cuda() if use_cuda else initial_input
    y_hat = model.incremental_forward(
        initial_input, c=c, g=g, T=length, tqdm=tqdm, softmax=True, quantize=True,
        log_scale_min=hparams.log_scale_min)

    if is_mulaw_quantize(hparams.input_type):
        y_hat = y_hat.max(1)[1].view(-1).long().cpu().data.numpy()
        y_hat = P.inv_mulaw_quantize(y_hat, hparams.quantize_channels)
        y_target = P.inv_mulaw_quantize(y_target, hparams.quantize_channels)
    elif is_mulaw(hparams.input_type):
        y_hat = P.inv_mulaw(y_hat.view(-1).cpu().data.numpy(), hparams.quantize_channels)
        y_target = P.inv_mulaw(y_target, hparams.quantize_channels)
    else:
        y_hat = y_hat.view(-1).cpu().data.numpy()

    # Save audio
    os.makedirs(eval_dir, exist_ok=True)
    path = join(eval_dir, "step{:09d}_predicted.wav".format(global_step))
    librosa.output.write_wav(path, y_hat, sr=hparams.sample_rate)
    path = join(eval_dir, "step{:09d}_target.wav".format(global_step))
    librosa.output.write_wav(path, y_target, sr=hparams.sample_rate)

    # save figure
    path = join(eval_dir, "step{:09d}_waveplots.png".format(global_step))
    save_waveplot(path, y_hat, y_target)


# save sample from
def save_states(global_step, writer, y_hat, y, y_student, input_lengths, checkpoint_dir=None):
    print("Save intermediate states at step {}".format(global_step))
    idx = np.random.randint(0, len(y_hat))
    length = input_lengths[idx].data.cpu().numpy()[0]

    # (B, C, T)
    if y_hat.dim() == 4:
        y_hat = y_hat.squeeze(-1)

    if is_mulaw_quantize(hparams.input_type):
        # (B, T)
        y_hat = F.softmax(y_hat, dim=1).max(1)[1]

        # (T,)
        y_hat = y_hat[idx].data.cpu().long().numpy()
        y = y[idx].view(-1).data.cpu().long().numpy()

        y_hat = P.inv_mulaw_quantize(y_hat, hparams.quantize_channels)
        y = P.inv_mulaw_quantize(y, hparams.quantize_channels)
    else:
        # (B, T)
        y_hat = sample_from_discretized_mix_logistic(
            y_hat, log_scale_min=hparams.log_scale_min)
        # (T,)
        y_hat = y_hat[idx].view(-1).data.cpu().numpy()
        y = y[idx].view(-1).data.cpu().numpy()

        if is_mulaw(hparams.input_type):
            y_hat = P.inv_mulaw(y_hat, hparams.quantize_channels)
            y = P.inv_mulaw(y, hparams.quantize_channels)

    # Mask by length
    y_hat[length:] = 0
    y[length:] = 0
    y_student = y_student.cpu().data.numpy()
    # Save audio
    audio_dir = join(checkpoint_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    path = join(audio_dir, "step{:09d}_teacher.wav".format(global_step))
    librosa.output.write_wav(path, y_hat, sr=hparams.sample_rate)
    path = join(audio_dir, "step{:09d}_target.wav".format(global_step))
    librosa.output.write_wav(path, y, sr=hparams.sample_rate)
    path = join(audio_dir, "step{:09d}_student.wav".format(global_step))
    librosa.output.write_wav(path, y_student, sr=hparams.sample_rate)


def __train_step(phase, epoch, global_step, global_test_step,
                 teacher, student, optimizer, writer,
                 x, y, c, g, input_lengths,
                 checkpoint_dir, eval_dir=None, do_eval=False, ema=None):
    sanity_check(teacher, c, g)  # ???
    sanity_check(student, c, g)  # ???

    # x : (B, C, T)
    # y : (B, T, 1)
    # c : (B, C, T)
    # g : (B,)
    train = (phase == "train")
    clip_thresh = hparams.clip_thresh
    if train:
        student.train()
        step = global_step
    else:
        student.eval()
        step = global_test_step

    # Learning rate schedule
    current_lr = hparams.initial_learning_rate
    if train and hparams.lr_schedule is not None:
        lr_schedule_f = getattr(lrschedule, hparams.lr_schedule)
        current_lr = lr_schedule_f(
            hparams.initial_learning_rate, step, **hparams.lr_schedule_kwargs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
    optimizer.zero_grad()

    # Prepare data
    x, y = Variable(x), Variable(y, requires_grad=False)
    c = Variable(c) if c is not None else None
    g = Variable(g) if g is not None else None
    input_lengths = Variable(input_lengths)
    if use_cuda:
        x, y = x.cuda(), y.cuda()
        input_lengths = input_lengths.cuda()
        c = c.cuda() if c is not None else None
        g = g.cuda() if g is not None else None

    # (B, T, 1)
    mask = sequence_mask(input_lengths, max_len=x.size(-1)).unsqueeze(-1)
    mask = mask[:, :, :]
    mask.expand_as(y)
    # apply the student model with stacked iaf layers and return mu,scale
    mu, scale = student(x, c=c, g=g, softmax=False)
    m, s = mu, scale
    mu, scale = to_numpy(mu), to_numpy(scale)
    cross_entropy_loss_sum = 0
    hpts = 0
    h_s = 0
    sample_T = 5  # Multi-sample from student
    for i in range(sample_T):
        student_predict = np.random.logistic(mu, scale)
        p_s = probs_logistic(mu, scale, student_predict)
        p_s = Variable(torch.from_numpy(p_s).float()).cuda().permute(0, 2, 1)
        # input student sample to evaluate probability
        student_predict = Variable(torch.from_numpy(student_predict).float()).cuda()
        y_hat = teacher(student_predict)  # y_hat: (B x C x T) teacher: 10-mixture-logistic
        student_predict = student_predict.permute(0, 2, 1)
        teacher_log_p = discretized_mix_logistic_loss(y_hat, student_predict, reduce=False)  # log(Pt)
        h_pt_ps = torch.sum(teacher_log_p * p_s * mask) / mask.sum()
        hpts += torch.sum(teacher_log_p) / mask.sum()
        a = s.permute(0, 2, 1)
        h_ps = torch.sum((torch.log(a) + 2) * mask) / mask.sum()
        h_s += h_ps
        # ss = h_pt_ps - h_ps
        cross_entropy_loss_sum += ((y - student_predict) ** 2).mean()
    # average over sample_T
    loss = hpts / sample_T - h_s / sample_T
    cross_entropy_loss = cross_entropy_loss_sum / sample_T
    power_loss = get_power_loss(to_numpy(student_predict), to_numpy(y))
    # print(power_loss )
    # loss = cross_entropy_loss  # + power_loss
    rs = loss.cpu().data.numpy()
    cross_entropy_loss = to_numpy(cross_entropy_loss) / sample_T
    if rs == np.isinf(rs):
        print('inf detected')
    else:
        print(rs, np.max(scale), np.max(mu), cross_entropy_loss)
    # loss = Variable(cross_entropy_loss., requires_grad=True).cuda()
    if train and step > 0 and step % hparams.checkpoint_interval == 0:
        save_states(step, writer, y_hat, y, student_predict, input_lengths, checkpoint_dir)
        save_checkpoint(student, optimizer, step, checkpoint_dir, epoch, ema)
    if do_eval:
        # NOTE: use train step (i.e., global_step) for filename
        # eval_model(global_step, writer, model, y, c, g, input_lengths, eval_dir, ema)
        eval_model(global_step, writer, student, y, c, g, input_lengths, eval_dir, ema)

    # Update
    if train:
        loss.backward()
        if clip_thresh > 0:
            grad_norm = torch.nn.utils.clip_grad_norm(student.parameters(), clip_thresh)
        optimizer.step()
        # update moving average
        if ema is not None:
            for name, param in student.named_parameters():
                if name in ema.shadow:
                    ema.update(name, param.data)

    # Logs
    writer.add_scalar("{} loss".format(phase), float(loss.data[0]), step)
    if train:
        if clip_thresh > 0:
            writer.add_scalar("gradient norm", grad_norm, step)
        writer.add_scalar("learning rate", current_lr, step)

    return loss.data[0]


def train_loop(student, teacher, data_loaders, optimizer, writer, checkpoint_dir=None):
    if use_cuda:
        student = student.cuda()
        teacher = teacher.cuda()

    if hparams.exponential_moving_average:
        ema = ExponentialMovingAverage(hparams.ema_decay)
        for name, param in student.named_parameters():
            if param.requires_grad:
                ema.register(name, param.data)
    else:
        ema = None

    global global_step, global_epoch, global_test_step
    while global_epoch < hparams.nepochs:
        for phase, data_loader in data_loaders.items():
            train = (phase == "train")
            running_loss = 0.
            test_evaluated = False
            for step, (x, y, c, g, input_lengths) in tqdm(enumerate(data_loader)):
                # Whether to save eval (i.e., online decoding) result
                do_eval = False
                eval_dir = join(checkpoint_dir, "{}_eval".format(phase))
                # Do eval per eval_interval for train
                if train and global_step > 0 \
                        and global_step % hparams.train_eval_interval == 0:
                    do_eval = True
                # Do eval for test
                # NOTE: Decoding WaveNet is quite time consuming, so
                # do only once in a single epoch for testset
                if not train and not test_evaluated \
                        and global_epoch % hparams.test_eval_epoch_interval == 0:
                    do_eval = True
                    test_evaluated = True
                if do_eval:
                    print("[{}] Eval at train step {}".format(phase, global_step))

                # Do step
                do_eval = False
                running_loss += __train_step(
                    phase, global_epoch, global_step, global_test_step, teacher, student,
                    optimizer, writer, x, y, c, g, input_lengths,
                    checkpoint_dir, eval_dir, do_eval, ema)

                # update global state
                if train:
                    global_step += 1
                else:
                    global_test_step += 1

            # log per epoch
            averaged_loss = running_loss / len(data_loader)
            writer.add_scalar("{} loss (per epoch)".format(phase),
                              averaged_loss, global_epoch)
            print("[{}] Loss: {}".format(phase, running_loss / len(data_loader)))

        global_epoch += 1


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch, ema=None):
    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    global global_test_step
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
        "global_test_step": global_test_step,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

    if ema is not None:
        averaged_model = clone_as_averaged_model(model, ema)
        checkpoint_path = join(
            checkpoint_dir, "checkpoint_step{:09d}_ema.pth".format(global_step))
        torch.save({
            "state_dict": averaged_model.state_dict(),
            "optimizer": optimizer_state,
            "global_step": step,
            "global_epoch": epoch,
            "global_test_step": global_test_step,
        }, checkpoint_path)
        print("Saved averaged checkpoint:", checkpoint_path)


def build_model(name='teacher'):
    if is_mulaw_quantize(hparams.input_type):
        if hparams.out_channels != hparams.quantize_channels:
            raise RuntimeError(
                "out_channels must equal to quantize_chennels if input_type is 'mulaw-quantize'")
    if hparams.upsample_conditional_features and hparams.cin_channels < 0:
        s = "Upsample conv layers were specified while local conditioning disabled. "
        s += "Notice that upsample conv layers will never be used."
        warn(s)
    if name == 'teacher':
        return getattr(builder, hparams.builder)(
            out_channels=hparams.out_channels,
            layers=hparams.layers,
            stacks=hparams.stacks,
            residual_channels=hparams.residual_channels,
            gate_channels=hparams.gate_channels,
            skip_out_channels=hparams.skip_out_channels,
            cin_channels=hparams.cin_channels,
            gin_channels=hparams.gin_channels,
            weight_normalization=hparams.weight_normalization,
            n_speakers=hparams.n_speakers,
            dropout=hparams.dropout,
            kernel_size=hparams.kernel_size,
            upsample_conditional_features=hparams.upsample_conditional_features,
            upsample_scales=hparams.upsample_scales,
            freq_axis_kernel_size=hparams.freq_axis_kernel_size,
            scalar_input=is_scalar_input(hparams.input_type),
        )
    else:
        # for now all the parameters are default
        return StudentWaveNet(
            cin_channels=hparams.cin_channels,
            gin_channels=hparams.gin_channels,
            upsample_conditional_features=hparams.upsample_conditional_features,
            upsample_scales=hparams.upsample_scales,
        )


def load_checkpoint(path, model, optimizer, reset_optimizer):
    global global_step
    global global_epoch
    global global_test_step

    print("Load checkpoint from: {}".format(path))
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]
    global_test_step = checkpoint.get("global_test_step", 0)

    return model


def get_data_loaders(data_root, speaker_id, test_shuffle=True):
    fname = './data/dataloader_pkl/train.pkl'
    if os.path.exists(fname):
        f = open(fname, 'rb')
        data_loaders = pickle.load(f)
        return data_loaders
    else:
        data_loaders = {}
        local_conditioning = hparams.cin_channels > 0
        for phase in ["train", "test"]:
            train = phase == "train"
            X = FileSourceDataset(RawAudioDataSource(data_root, speaker_id=speaker_id,
                                                     train=train,
                                                     test_size=hparams.test_size,
                                                     test_num_samples=hparams.test_num_samples,
                                                     random_state=hparams.random_state))
            if local_conditioning:
                Mel = FileSourceDataset(MelSpecDataSource(data_root, speaker_id=speaker_id,
                                                          train=train,
                                                          test_size=hparams.test_size,
                                                          test_num_samples=hparams.test_num_samples,
                                                          random_state=hparams.random_state))
                assert len(X) == len(Mel)
                print("Local conditioning enabled. Shape of a sample: {}.".format(
                    Mel[0].shape))
            else:
                Mel = None
            print("[{}]: length of the dataset is {}".format(phase, len(X)))

            if train:
                lengths = np.array(X.file_data_source.lengths)
                # Prepare sampler
                sampler = PartialyRandomizedSimilarTimeLengthSampler(
                    lengths, batch_size=hparams.batch_size)
                shuffle = False
            else:
                sampler = None
                shuffle = test_shuffle

            dataset = PyTorchDataset(X, Mel)
            data_loader = data_utils.DataLoader(
                dataset, batch_size=hparams.batch_size,
                num_workers=hparams.num_workers, sampler=sampler, shuffle=shuffle,
                collate_fn=collate_fn, pin_memory=hparams.pin_memory)

            speaker_ids = {}
            for idx, (x, c, g) in enumerate(dataset):
                if g is not None:
                    try:
                        speaker_ids[g] += 1
                    except KeyError:
                        speaker_ids[g] = 1
            if len(speaker_ids) > 0:
                print("Speaker stats:", speaker_ids)

            data_loaders[phase] = data_loader
        with open(fname, 'wb') as f:
            pickle.dump(data_loaders, f)
        return data_loaders


if __name__ == "__main__":
    # args = docopt(__doc__)
    args = {
        "--checkpoint-dir": 'checkpoints_student',
        "--checkpoint_teacher": './checkpoints_teacher/checkpoint_step000700000.pth',  # the pre-trained teacher model
        "--checkpoint_student": None,  # 是否加载
        "--checkpoint": None,
        "--restore-parts": None,
        "--data-root": './data/ljspeech',  # dataset
        "--log-event-path": None,  # if continue training, reload the checkpoint
        "--speaker-id": None,
        "--reset-optimizer": None,
        "--hparams": "cin_channels=80,gin_channels=-1"

    }
    print("Command line args:\n", args)
    checkpoint_dir = args["--checkpoint-dir"]
    # checkpoint_path = args["--log-event-teacher-path"]
    checkpoint_teacher_path = args["--checkpoint_teacher"]
    checkpoint_student_path = args["--checkpoint_student"]
    checkpoint_restore_parts = args["--restore-parts"]
    speaker_id = args["--speaker-id"]
    speaker_id = int(speaker_id) if speaker_id is not None else None

    data_root = args["--data-root"]
    if data_root is None:
        data_root = join(dirname(__file__), "data", "ljspeech")

    log_event_path = args["--log-event-path"]
    reset_optimizer = args["--reset-optimizer"]

    # Override hyper parameters
    hparams.parse(args["--hparams"])
    print(hparams_debug_string())
    assert hparams.name == "wavenet_vocoder"

    # Presets
    if hparams.preset is not None and hparams.preset != "":
        preset = hparams.presets[hparams.preset]
        import json

        hparams.parse_json(json.dumps(preset))
        print("Override hyper parameters with preset \"{}\": {}".format(
            hparams.preset, json.dumps(preset, indent=4)))

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Dataloader setup
    data_loaders = get_data_loaders(data_root, speaker_id, test_shuffle=True)

    # Model
    student_model = build_model(name='student')
    teacher_model = build_model(name='teacher')

    print(teacher_model)
    if use_cuda:
        teacher_model = teacher_model.cuda()
        student_model = student_model.cuda()

    receptive_field = teacher_model.receptive_field
    print("Receptive field (samples / ms): {} / {}".format(
        receptive_field, receptive_field / fs * 1000))
    # teacher和student share 大部分的参数
    optimizer = optim.Adam(student_model.parameters(),
                           lr=hparams.initial_learning_rate,
                           betas=(hparams.adam_beta1, hparams.adam_beta2),
                           eps=hparams.adam_eps,
                           weight_decay=hparams.weight_decay)

    # load teacher model first
    restore_parts(checkpoint_teacher_path, teacher_model)
    teacher_model.eval()  # the teacher use eval not to train the parameters
    # 是否加载继续训练
    # if checkpoint_restore_parts is not None:
    #     restore_parts(checkpoint_restore_parts, student_model)

    # Load checkpoints
    if checkpoint_student_path is not None:
        load_checkpoint(checkpoint_student_path, student_model, optimizer, reset_optimizer)

    # Setup summary writer for tensorboard
    if log_event_path is None:
        log_event_path = "log/run-test" + str(datetime.now()).replace(" ", "_")
    print("Los event path: {}".format(log_event_path))
    writer = SummaryWriter(log_dir=log_event_path)

    # Train!
    try:
        train_loop(student=student_model,
                   teacher=teacher_model,
                   data_loaders=data_loaders,
                   optimizer=optimizer,
                   writer=writer,
                   checkpoint_dir=checkpoint_dir)
    except KeyboardInterrupt:
        save_checkpoint(student_model, optimizer, global_step, checkpoint_dir, global_epoch)

    print("Finished")
    sys.exit(0)