import pytest
import torch

from protopnet.pretrained.sleep_seqepoch_features import (
    SeqEpochEncoder, encode_channels, load_epoch_encoder_weights,
)


def test_independent_channels_and_checkpoint_loading():
    torch.manual_seed(42)
    encoder = SeqEpochEncoder().eval()
    x = torch.randn(2, 3, 29, 129)
    with torch.no_grad():
        h = encode_channels(encoder, x)
        assert h.shape == (2, 128, 1, 3)
        assert torch.isfinite(h).all()
        changed = x.clone()
        changed[:, 0] += 3
        h_changed = encode_channels(encoder, changed)
        torch.testing.assert_close(h[..., 1:], h_changed[..., 1:])
        assert not torch.allclose(h[..., 0], h_changed[..., 0])
        perm = [2, 0, 1]
        torch.testing.assert_close(encode_channels(encoder, x[:, perm]), h[..., perm])

    state = {'epoch_encoder.' + k: v for k, v in encoder.state_dict().items()}
    state['classifier.weight'] = torch.randn(5, 128)
    restored = load_epoch_encoder_weights(SeqEpochEncoder(), state).eval()
    with torch.no_grad():
        torch.testing.assert_close(encode_channels(restored, x), h)
    del state['epoch_encoder.filterbank.W']
    with pytest.raises(RuntimeError, match='Missing key'):
        load_epoch_encoder_weights(restored, state)


def test_gradients_and_invalid_shape():
    encoder = SeqEpochEncoder()
    x = torch.randn(1, 3, 29, 129)
    encode_channels(encoder, x).square().mean().backward()
    for parameter in encoder.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
    with pytest.raises(ValueError, match='Expected spectrograms'):
        encode_channels(encoder, torch.randn(1, 3, 3000))
