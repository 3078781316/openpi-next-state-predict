import torch

from openpi.models.utils.frequency_rvq_tokenizer import FrequencyRVQConfig
from openpi.models.utils.frequency_rvq_tokenizer import FrequencyRVQTokenizerModel
from openpi.models.utils.frequency_rvq_tokens import FrequencyRVQTokenLayout


def make_small_model() -> FrequencyRVQTokenizerModel:
    return FrequencyRVQTokenizerModel(
        FrequencyRVQConfig(
            action_horizon=8,
            action_dim=3,
            low_frequency_bins=2,
            latent_dim=8,
            hidden_dim=16,
            low_codebook_size=8,
            high_codebook_size=4,
            num_residual_stages=2,
            kmeans_init=False,
        )
    )


def test_dct_round_trip():
    model = make_small_model()
    actions = torch.randn(5, 8, 3)
    torch.testing.assert_close(model.idct(model.dct(actions)), actions, atol=1e-5, rtol=1e-5)


def test_encode_decode_shapes():
    model = make_small_model().eval()
    actions = torch.randn(5, 8, 3)
    code_ids = model.encode(actions)
    assert code_ids.shape == (5, 3)
    assert model.decode(code_ids).shape == actions.shape
    assert model.decode(code_ids, num_residual_stages=0).shape == actions.shape


def test_action_token_layout_round_trip():
    layout = FrequencyRVQTokenLayout()
    vocab_size = 257_152
    for stage, size in enumerate(layout.codebook_sizes):
        for code_id in (0, size - 1):
            local_id = layout.local_id(stage, code_id)
            token_id = layout.paligemma_id(local_id, vocab_size)
            assert layout.local_id_from_paligemma(token_id, vocab_size) == local_id
            low, high = layout.paligemma_range(stage, vocab_size)
            assert low <= token_id <= high
