# Frequency-RVQ action tokenizer

This implementation replaces FAST action compression with four coarse-to-fine codes:

```text
normalized action [16, 7]
  -> orthonormal DCT-II
  -> low coefficients [4, 7] -> VQ-512 -> LOW
  -> high coefficients [12, 7] -> 3-stage RVQ-256 -> RES1, RES2, RES3
  -> frozen decoders + IDCT -> normalized action [16, 7]
```

The tokenizer is trained in PyTorch. `Pi0FrequencyRVQ` remains a JAX model and is parameter-tree compatible with
`pi0_fast_base`. The tokenizer is frozen while the VLA is trained.

## 1. Prepare data

The output directory in this example matches the normalization-assets path in the provided training configs.

```bash
uv run scripts/prepare_frequency_rvq_data.py \
  --repo-id physical-intelligence/libero \
  --output-dir data/frequency_rvq_libero \
  --action-horizon 16 \
  --stride 2
```

Complete episodes are split before windows are extracted. The script writes:

```text
data/frequency_rvq_libero/
  train.npz
  validation.npz
  test.npz
  norm_stats.npz
  norm_stats.json
  metadata.json
```

`norm_stats.json` is directly consumable by OpenPI. The VLA config must use the same statistics because the frozen
tokenizer expects actions normalized with these train-split quantiles.

## 2. Train the tokenizer

```bash
uv run scripts/train_frequency_rvq_tokenizer.py \
  --data-dir data/frequency_rvq_libero \
  --output-dir checkpoints/frequency_rvq_libero \
  --epochs 100 \
  --pretrain-epochs 5 \
  --batch-size 512
```

The first five epochs train the low/high autoencoders without quantization. The first quantized batch initializes
each codebook with K-means; subsequent updates use EMA. `best/` is selected using validation action L1.

The validation log reports reconstruction using zero, one, two, and three high-frequency residual stages. Verify
that adding stages generally reduces the reconstruction error before training the VLA.

## 3. Optional offline action encoding

```bash
uv run scripts/encode_frequency_rvq_actions.py \
  --data-dir data/frequency_rvq_libero \
  --tokenizer-path checkpoints/frequency_rvq_libero/best \
  --output-dir data/frequency_rvq_libero_codes
```

The current OpenPI transform integration encodes actions online. The offline files are intended for analysis and a
future dataset schema that stores action-code columns directly.

## 4. Fine-tune pi0_fast_base

Full fine-tuning:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_frequency_rvq_libero \
  --exp-name=my_frequency_rvq_run
```

LoRA fine-tuning:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_frequency_rvq_libero_low_mem_finetune \
  --exp-name=my_frequency_rvq_lora_run
```

The policy predicts exactly four tokens. Each step is restricted to its own legal PaliGemma token interval, and the
training loss weights LOW/RES1/RES2/RES3 by `2.0/1.0/0.7/0.5`. Output tokens are decoded by the frozen PyTorch RVQ
decoder before OpenPI applies action unnormalization.

## Vendored quantizer

`residual_vq.py` and `vector_quantize_pytorch.py` are vendored from the MIT-licensed VQ-BeT implementation. Their
license is retained in `src/openpi/models/utils/VECTOR_QUANTIZE_LICENSE`.
