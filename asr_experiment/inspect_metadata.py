"""Lightweight HuggingFace API-based metadata inspection (no large downloads)."""
from huggingface_hub import HfApi, dataset_info
import json

api = HfApi()

datasets_to_check = [
    "AILAB-VNUHCM/vivos",
    "linhtran92/fpt_fosd",
    "thenghia2206/ViMD",
    "doof-ferb/VietMed",
    "thanhnew2001/VietSuperSpeech",
]

for ds_name in datasets_to_check:
    try:
        info = api.dataset_info(ds_name)
        print(f"=== {ds_name} ===")
        print(f"  ID: {info.id}")
        print(f"  License: {info.card_data.license if info.card_data and hasattr(info.card_data, 'license') else 'NOT SET'}")
        print(f"  Tags: {info.tags[:10] if info.tags else 'None'}")
        print(f"  Downloads: {info.downloads}")
        print(f"  Likes: {info.likes}")
        print(f"  Gated: {info.gated}")
        
        # List first few files to understand structure
        files = api.list_repo_files(ds_name, repo_type="dataset")
        file_list = list(files)[:30]
        print(f"  Files ({len(list(files))} total, showing first 30):")
        for f in file_list:
            print(f"    {f}")
        print()
    except Exception as e:
        print(f"=== {ds_name} === ERROR: {e}")
        print()
