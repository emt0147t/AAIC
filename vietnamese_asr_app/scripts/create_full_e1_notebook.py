import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Full E1 Training: PhoWhisper-tiny LoRA on 26,671 Vietnamese Utterances\n",
                "\n",
                "**Project:** Vietnamese Speech-to-Text Research (Phase 3 Full-Scale Experiment E1)  \n",
                "**Target Environment:** Google Colab Linux GPU (NVIDIA Tesla T4 / L4 / A100)  \n",
                "**Base Model:** `vinai/PhoWhisper-tiny` (Pinned commit `cc51d32be916efebde04ff549854fa1741cb5c02`)  \n",
                "**PEFT Version:** `0.21.0`  \n",
                "**LoRA Configuration:** $r=8, \\alpha=16$, dropout=0.05, `target_modules=['q_proj', 'v_proj']`, `bias='none'` (147,456 trainable parameters, 0.389%)  \n",
                "**Training Corpus:** 26,671 training-ready utterances (VIVOS train: 11,660 + ViMD train: 15,011)  \n",
                "**Budget:** Micro Batch = 4, Grad Accum = 8, Effective Batch = 32, FP16 mixed precision  \n",
                "**Schedule:** 834 steps/epoch × 3 epochs = **2,502 optimizer steps** (80,064 sample exposures)  \n",
                "**Validation:** `manifests/full_val_manifest.csv` (1,900 ViMD valid utterances, 10.26 hours)  \n",
                "**Frozen Gold Test:** `manifests/test_manifest.csv` (Evaluated strictly post-training on final checkpoint)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 0. Install Pinned Dependencies"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install -q peft==0.21.0 accelerate==1.15.0 transformers==4.49.0 datasets soundfile jiwer pandas pyyaml scipy"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Hardware Verification & Frozen Test Hash Check"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, sys, hashlib, torch, peft, transformers\n",
                "\n",
                "print(f\"OS:              {sys.platform}\")\n",
                "print(f\"Python:          {sys.version.split()[0]}\")\n",
                "print(f\"PyTorch:         {torch.__version__}\")\n",
                "print(f\"PEFT:            {peft.__version__}\")\n",
                "print(f\"Transformers:    {transformers.__version__}\")\n",
                "print(f\"CUDA Available:  {torch.cuda.is_available()}\")\n",
                "assert torch.cuda.is_available(), 'CRITICAL ERROR: CUDA is required for FP16 Full E1 training!'\n",
                "\n",
                "gpu_name = torch.cuda.get_device_name(0)\n",
                "total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)\n",
                "print(f\"GPU Model:       {gpu_name} ({total_vram_mb:.1f} MB)\")\n",
                "\n",
                "# Check frozen test manifest hash\n",
                "test_manifest = 'manifests/test_manifest.csv'\n",
                "if os.path.exists(test_manifest):\n",
                "    h = hashlib.sha256(open(test_manifest, 'rb').read()).hexdigest()\n",
                "    print(f\"Test Hash:       {h}\")\n",
                "    assert h == 'efe54856d12613f109cebc9482a6f224ca4e2370d0ff3f8a2fe4ea0a013b60ca', 'Test manifest altered!'\n",
                "    print(\"Test manifest integrity strictly VERIFIED.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Load Model & Setup LoRA (cc51d32be916efebde04ff549854fa1741cb5c02)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from peft import LoraConfig, get_peft_model\n",
                "from transformers import WhisperForConditionalGeneration, WhisperProcessor\n",
                "\n",
                "MODEL_NAME = 'vinai/PhoWhisper-tiny'\n",
                "MODEL_REVISION = 'cc51d32be916efebde04ff549854fa1741cb5c02'\n",
                "\n",
                "processor = WhisperProcessor.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)\n",
                "base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)\n",
                "\n",
                "device = torch.device('cuda')\n",
                "lora_config = LoraConfig(\n",
                "    r=8,\n",
                "    lora_alpha=16,\n",
                "    lora_dropout=0.05,\n",
                "    target_modules=['q_proj', 'v_proj'],\n",
                "    bias='none'\n",
                ")\n",
                "model = get_peft_model(base_model, lora_config)\n",
                "model.to(device)\n",
                "\n",
                "trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)\n",
                "total = sum(p.numel() for p in model.parameters())\n",
                "print(f\"Trainable parameters: {trainable:,} ({100*trainable/total:.4f}%)\")\n",
                "assert trainable == 147456, f'Expected 147,456 trainable parameters, got {trainable}'"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Verify Deterministic Exposure Policy Across Epochs"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import numpy as np\n",
                "\n",
                "AUDITED_REPEATED_INDICES = {\n",
                "    1: [186, 2394, 3596, 4922, 5311, 9527, 9922, 11904, 12434, 13590, 14261, 15356, 15589, 16068, 19817, 19918, 20082],\n",
                "    2: [791, 2720, 3793, 4460, 4869, 6765, 7657, 8088, 10060, 10616, 12573, 15738, 17952, 22257, 23086, 23991, 26608],\n",
                "    3: [329, 854, 868, 2852, 2949, 7126, 9800, 12278, 12298, 17346, 17498, 20781, 21165, 21287, 22885, 24648, 26633]\n",
                "}\n",
                "\n",
                "for ep in [1, 2, 3]:\n",
                "    epoch_seed = 42 + (ep - 1)\n",
                "    pad_rng = np.random.default_rng(epoch_seed + 1000)\n",
                "    pad_idx = sorted(pad_rng.choice(26671, size=17, replace=False).tolist())\n",
                "    assert pad_idx == AUDITED_REPEATED_INDICES[ep], f'Exposure policy mismatch in epoch {ep}!'\n",
                "    print(f'Epoch {ep} exposure policy verified: matches audited schedule.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Run Full E1 Training (2,502 Steps / 3 Epochs)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Launch full training with FP16 GradScaler, saving to checkpoints/FULL_E1/\n",
                "print(\"Starting Full E1 Training Execution...\")\n",
                "!python scripts/train_full_e1.py"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

output_path_root = r"D:\Vietnamese_ASR_Week6\ASR_FULL_E1_TRAINING.ipynb"
with open(output_path_root, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)
print(f"Created {output_path_root}")

output_path_app = r"D:\Vietnamese_ASR_Week6\vietnamese_asr_app\ASR_FULL_E1_TRAINING.ipynb"
with open(output_path_app, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)
print(f"Created {output_path_app}")
