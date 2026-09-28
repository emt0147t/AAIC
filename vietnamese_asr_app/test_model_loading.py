import gc

import torch

from asr.model import ASRModelManager


MODEL_ID = "vinai/PhoWhisper-tiny"


print("=" * 70)
print("B5 MODEL LOADING TEST")
print("=" * 70)

print("Requested model:", MODEL_ID)
print("CUDA available:", torch.cuda.is_available())


# ---------------------------------------------------------------------
# 1. Load model + processor
# ---------------------------------------------------------------------
model, processor, hw = ASRModelManager.get_model_and_processor(
    model_id=MODEL_ID,
    preferred_device="auto",
)

print("\n[1] Hardware diagnostics")
print("Device:", hw.device)
print("Diagnostics:", hw)


# ---------------------------------------------------------------------
# 2. Basic object checks
# ---------------------------------------------------------------------
print("\n[2] Object checks")
print("Model type:", type(model))
print("Processor type:", type(processor))

assert model is not None
assert processor is not None


# ---------------------------------------------------------------------
# 3. Model parameter checks
# ---------------------------------------------------------------------
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("\n[3] Parameters")
print("Total parameters:", total_params)
print("Trainable parameters:", trainable_params)


# ---------------------------------------------------------------------
# 4. Device + dtype checks
# ---------------------------------------------------------------------
first_param = next(model.parameters())

print("\n[4] Model state")
print("Parameter device:", first_param.device)
print("Parameter dtype:", first_param.dtype)
print("Training mode:", model.training)
print("Eval mode:", not model.training)

assert not model.training


# ---------------------------------------------------------------------
# 5. Generation config safety
# ---------------------------------------------------------------------
generation_config = getattr(model, "generation_config", None)

print("\n[5] Generation config")

if generation_config is not None:
    print(
        "forced_decoder_ids:",
        generation_config.forced_decoder_ids
    )

    assert generation_config.forced_decoder_ids is None
else:
    print("generation_config: None")


# ---------------------------------------------------------------------
# 6. Singleton/cache behavior
# ---------------------------------------------------------------------
model2, processor2, hw2 = (
    ASRModelManager.get_model_and_processor(
        model_id=MODEL_ID,
        preferred_device="auto",
    )
)

print("\n[6] Cache behavior")
print("Same model object:", model is model2)
print("Same processor object:", processor is processor2)
print("Same device:", hw.device == hw2.device)

assert model is model2
assert processor is processor2


# ---------------------------------------------------------------------
# 7. Clear cache
# ---------------------------------------------------------------------
ASRModelManager.clear_cache()
gc.collect()

print("\n[7] Cache clearing")
print("Cache cleared: PASS")


print("\n" + "=" * 70)
print("B5 MODEL LOADING TEST: PASS")
print("=" * 70)