"""
Download and convert medical SFT datasets to MedicalGPT sharegpt format.
Datasets:
  1. FreedomIntelligence/HuatuoGPT-sft-data-v1  (220K Chinese medical conversations)
  2. shibing624/medical (SFT split, Chinese medical Q&A)
"""

import json
import os

from datasets import load_dataset
from loguru import logger

OUTPUT_DIR = "./data/finetune"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(data)} samples to {path}")


def strip_qa_prefix(text, prefix):
    text = (text or "").strip()
    if text.startswith(prefix):
        return text[len(prefix):].strip()
    return text


# ── 1. HuatuoGPT-sft-data-v1 ──────────────────────────────────────────────
logger.info("Downloading FreedomIntelligence/HuatuoGPT-sft-data-v1 ...")
try:
    ds = load_dataset("FreedomIntelligence/HuatuoGPT-sft-data-v1", split="train")
    logger.info(f"HuatuoGPT dataset size: {len(ds)}")
    logger.info(f"Columns: {ds.column_names}")
    logger.info(f"Sample: {ds[0]}")

    results = []
    for item in ds:
        # Format: {"conversations": [{"from": "human", "value": ...}, {"from": "gpt", "value": ...}]}
        if "conversations" in item:
            results.append({"conversations": item["conversations"]})
        elif "data" in item and isinstance(item["data"], list) and len(item["data"]) >= 2:
            question = strip_qa_prefix(item["data"][0], "问：")
            answer = strip_qa_prefix(item["data"][1], "答：")
            if question and answer:
                results.append({"conversations": [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": answer},
                ]})
        elif "input" in item and "output" in item:
            results.append({"conversations": [
                {"from": "human", "value": item["input"]},
                {"from": "gpt", "value": item["output"]},
            ]})
        elif "question" in item and "answer" in item:
            results.append({"conversations": [
                {"from": "human", "value": item["question"]},
                {"from": "gpt", "value": item["answer"]},
            ]})

    save_jsonl(results, f"{OUTPUT_DIR}/huatuogpt_sft_220k.jsonl")
except Exception as e:
    logger.error(f"HuatuoGPT download failed: {e}")


# ── 2. shibing624/medical (finetune split) ─────────────────────────────────
logger.info("Downloading shibing624/medical (finetune split) ...")
try:
    medical_files = [
        "https://huggingface.co/datasets/shibing624/medical/resolve/main/finetune/train_zh_0.json",
        "https://huggingface.co/datasets/shibing624/medical/resolve/main/finetune/train_en_1.json",
    ]
    ds2 = load_dataset("json", data_files=medical_files, split="train")
    logger.info(f"shibing624/medical finetune size: {len(ds2)}")
    logger.info(f"Columns: {ds2.column_names}")
    logger.info(f"Sample: {ds2[0]}")

    results2 = []
    for item in ds2:
        if "instruction" in item:
            q = item["instruction"]
            if item.get("input"):
                q = q + "\n\n" + item["input"]
            results2.append({"conversations": [
                {"from": "human", "value": q},
                {"from": "gpt", "value": item["output"]},
            ]})
        elif "conversations" in item:
            results2.append({"conversations": item["conversations"]})

    save_jsonl(results2, f"{OUTPUT_DIR}/shibing624_medical_sft.jsonl")
except Exception as e:
    logger.error(f"shibing624/medical download failed: {e}")

logger.info("Done! Check ./data/finetune/ for downloaded files.")
