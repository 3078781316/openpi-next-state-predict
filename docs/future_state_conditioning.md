# Explicit future-state conditioning for pi0.5

This repository variant can fine-tune pi0.5 through an explicit numerical future-state bottleneck:

```text
image + prompt + q_t
        -> q_hat_{t+1:t+H}
        -> predicted-state KV cache
        -> action flow matching for a_{t:t+H-1}
```

The state at index `k` is aligned with action index `k`:

```text
future_states[k] = q_{t+k+1}
actions[k]       = a_{t+k}
```

The action expert receives the predicted trajectory in two ways:

1. All action queries attend to a time-independent KV block built by re-encoding
   `[q_t, q_hat_{t+1}, ..., q_hat_{t+H}]`.
2. Action token `k` receives a residual embedding of predicted state `k`.

Only the numerical prediction crosses the boundary between the state predictor and action expert. By default it is
also gradient-detached before action conditioning, so the action loss cannot turn the intermediate state into an
uninterpretable latent code.

## Configurations

- `pi05_libero_future_state`
- `pi05_libero_future_state_lora`
- `pi05_full_droid_future_state_finetune`
- `pi05_droid_future_state_finetune`
- `pi05_droid_future_state_lora_finetune`

The LoRA variants adapt both the PaliGemma language transformer and action expert, freeze
their base weights and the SigLIP vision tower, and fully train the future-state/action heads.

Change the placeholder DROID paths/repository IDs before training. Compute normalization statistics as usual when
training on a new dataset; future states reuse the current-state statistics.

Important model options:

```python
Pi0Config(
    pi05=True,
    predict_future_states=True,
    state_dim=8,  # real state dimension before padding
    future_state_loss_weight=1.0,
    detach_future_state_condition=True,
)
```

`state_dim` must match the robot observation state. Padded dimensions are forced to zero so that they cannot become
an unsupervised communication channel.

## Loss

Training minimizes the existing action flow-matching loss plus a masked Huber trajectory loss:

```text
L = L_action_flow + future_state_loss_weight * L_future_state
```

Episode-tail padding and padded state dimensions are excluded from the future-state loss. Future states come from
future observations, not by integrating actions, because action semantics differ across DROID, ALOHA, and LIBERO.
