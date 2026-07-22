"""Shared action-token layout for the frequency-RVQ policy."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class FrequencyRVQTokenLayout:
    """Maps four codebooks into disjoint local and PaliGemma token ranges."""

    codebook_sizes: tuple[int, ...] = (512, 256, 256, 256)
    paligemma_special_tokens: int = 128

    @property
    def offsets(self) -> tuple[int, ...]:
        offsets = [0]
        for size in self.codebook_sizes[:-1]:
            offsets.append(offsets[-1] + size)
        return tuple(offsets)

    @property
    def num_stages(self) -> int:
        return len(self.codebook_sizes)

    @property
    def vocab_size(self) -> int:
        return sum(self.codebook_sizes)

    def local_id(self, stage: int, code_id: int) -> int:
        if not 0 <= stage < self.num_stages:
            raise ValueError(f"Invalid stage {stage}")
        if not 0 <= code_id < self.codebook_sizes[stage]:
            raise ValueError(f"Code {code_id} is invalid for stage {stage}")
        return self.offsets[stage] + code_id

    def paligemma_id(self, local_id: int, paligemma_vocab_size: int) -> int:
        if not 0 <= local_id < self.vocab_size:
            raise ValueError(f"Invalid local action token {local_id}")
        return paligemma_vocab_size - 1 - self.paligemma_special_tokens - local_id

    def local_id_from_paligemma(self, token_id: int, paligemma_vocab_size: int) -> int:
        local_id = paligemma_vocab_size - 1 - self.paligemma_special_tokens - token_id
        if not 0 <= local_id < self.vocab_size:
            raise ValueError(f"PaliGemma token {token_id} is not a frequency-RVQ action token")
        return local_id

    def paligemma_range(self, stage: int, paligemma_vocab_size: int) -> tuple[int, int]:
        """Returns the inclusive ascending token-ID range for a stage."""
        offset = self.offsets[stage]
        size = self.codebook_sizes[stage]
        high = self.paligemma_id(offset, paligemma_vocab_size)
        low = self.paligemma_id(offset + size - 1, paligemma_vocab_size)
        return low, high


DEFAULT_FREQUENCY_RVQ_LAYOUT = FrequencyRVQTokenLayout()
