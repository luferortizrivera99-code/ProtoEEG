"""Per-channel SeqSleepNet epoch encoder, adapted from PhysioEx v2.0.2.

Sources (MIT license):
https://github.com/guidogagl/physioex/blob/v2.0.2/physioex/models/seqsleepnet.py
https://github.com/guidogagl/physioex/blob/v2.0.2/physioex/models/protosleepnet.py
Copyright (c) PhysioEx contributors. See LICENSE_PHYSIOEX alongside this file.

The filterbank/LSTM/attention operations are preserved. The nested upstream
encoder is exposed as a standalone class with explicit constructor defaults.
No ChannelMixer, inter-epoch encoder, prototypes or classifier are included.
"""
import torch
from torch import nn

class LearnableFilterbank(nn.Module):
    def __init__(
        self,
        in_chan: int = 2,
        F: int = 129,
        nfilt: int = 32,
        nfft: int = 256,
        sf: int = 100,
        lowfreq: int = 0,
        highfreq: int = 50,
    ):
        super().__init__()
        self.F, self.D = F, nfilt

        S = torch.zeros((in_chan, F, nfilt), dtype=torch.float32)

        for i in range(in_chan):
            S[i] = self.lin_tri_filter_shape(nfilt, nfft, sf, lowfreq, highfreq)

        W = torch.zeros((in_chan, F, nfilt), dtype=torch.float32)

        self.W = nn.Parameter(W, requires_grad=True)
        self.S = nn.Parameter(S, requires_grad=False)

        nn.init.normal_(self.W)

    def forward(self, x):
        Wfb = torch.mul(torch.sigmoid(self.W), self.S)

        return torch.matmul(x, Wfb)

    def lin_tri_filter_shape(
        self, nfilt=20, nfft=512, sf=16000, lowfreq=0, highfreq=None
    ):
        highfreq = highfreq or sf / 2
        assert highfreq <= sf / 2, "highfreq is greater than sf/2"

        hzpoints = torch.linspace(lowfreq, highfreq, nfilt + 2)
        bin = torch.floor((nfft + 1) * hzpoints / sf)

        fbank = torch.zeros([nfilt, nfft // 2 + 1])
        for j in range(0, nfilt):
            for i in range(int(bin[j]), int(bin[j + 1])):
                fbank[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
            for i in range(int(bin[j + 1]), int(bin[j + 2])):
                fbank[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])
        fbank = torch.transpose(fbank, 0, 1)
        return fbank.float()

class AttentionLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        attention_size: int = 32,
        time_major: bool = False,
    ):
        super().__init__()

        W_omega = torch.zeros((hidden_size, attention_size), dtype=torch.float32)
        b_omega = torch.zeros((attention_size), dtype=torch.float32)
        u_omega = torch.zeros((attention_size), dtype=torch.float32)

        self.W_omega = nn.Parameter(W_omega)
        self.b_omega = nn.Parameter(b_omega)
        self.u_omega = nn.Parameter(u_omega)

        nn.init.normal_(self.W_omega, std=0.1)
        nn.init.normal_(self.b_omega, std=0.1)
        nn.init.normal_(self.u_omega, std=0.1)

    def forward(self, x, r_alphas=False):
        batch_size, sequence_length, hidden_size = x.size()

        v = torch.tanh(
            torch.matmul(
                torch.reshape(x, [batch_size * sequence_length, hidden_size]),
                self.W_omega,
            )
            + torch.reshape(self.b_omega, [1, -1])
        )
        vu = torch.matmul(v, torch.reshape(self.u_omega, [-1, 1]))
        exps = torch.reshape(torch.exp(vu), [-1, sequence_length])
        alphas = exps / torch.reshape(torch.sum(exps, 1), [-1, 1])

        output = torch.sum(
            x * torch.reshape(alphas, [batch_size, sequence_length, 1]), 1
        )
        if r_alphas:
            return output, alphas
        return output

class SeqEpochEncoder(nn.Module):
            def __init__(self, F=129, D=32, nfft=256, fs=100, lowfreq=0, highfreq=50,
                 seqnhidden1=64, seqnlayer1=4, attentionsize=32):
                super().__init__()
                self.filterbank = LearnableFilterbank(
                    in_chan=1, F=F, nfilt=D, nfft=nfft, sf=fs,
                    lowfreq=lowfreq, highfreq=highfreq,
                )
                self.lstm = nn.LSTM(
                    D, seqnhidden1, num_layers=seqnlayer1,
                    batch_first=True, bidirectional=True,
                )
                self.attention = AttentionLayer(2 * seqnhidden1, attentionsize)

            def forward(self, x):
                x = self.filterbank(x)
                x = x.permute(0, 2, 1, 3)
                x = x.reshape(x.shape[0], x.shape[1], -1)
                x, _ = self.lstm(x)
                x = self.attention(x)
                return x

def encode_channels(encoder, spectrograms):
    """(B, 3, 29, 129) -> (B, 128, 1, 3), with shared independent encoding.

    Pass an encoder in eval mode for reproducible inference. No normalization,
    sensor weighting or channel mixing is performed here.
    """
    if spectrograms.ndim != 4 or tuple(spectrograms.shape[1:]) != (3, 29, 129):
        raise ValueError("Expected spectrograms with shape (B, 3, 29, 129)")
    batch = spectrograms.shape[0]
    h = encoder(spectrograms.reshape(batch * 3, 1, 29, 129))
    return h.reshape(batch, 3, -1).permute(0, 2, 1).unsqueeze(2).contiguous()


def load_epoch_encoder_weights(encoder, checkpoint):
    """Load only encoder weights from an upstream state_dict, strictly.

    Supports a full model's epoch_encoder.* keys or a standalone encoder dict.
    Does not accept pickled model objects or silently skip incompatible layers.
    """
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a state_dict checkpoint")
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict):
        raise ValueError("Expected a state_dict mapping")
    selected = {k[len("epoch_encoder."):]: v for k, v in state.items()
                if k.startswith("epoch_encoder.")}
    if not selected:
        selected = state
    encoder.load_state_dict(selected, strict=True)
    return encoder
