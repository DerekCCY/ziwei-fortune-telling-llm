import json

def load_system_intro():
    """
    從當前目錄下的 sft_prompt.txt 讀取 system intro，
    SFT 和 evaluation 共用同一份前置說明。
    """
    path = "data/sft_prompt.txt"  # <== 寫死檔名
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        # 後面接 <INPUT> 比較好讀，多補兩個換行
        SYSTEM_INTRO = text + "\n\n"
        # print(f"[System Intro] Loaded from {path}, length={len(SYSTEM_INTRO)} chars.")
    except FileNotFoundError:
        SYSTEM_INTRO = ""
        print(f"[System Intro] File not found at {path}, proceed without system intro.")
    except Exception as e:
        SYSTEM_INTRO = ""
        print(f"[System Intro] Failed to load from {path}: {e}")
    return SYSTEM_INTRO

def build_prompt(birth_info):
    """
    回傳到 </think> 結束的 prefix：
    system + user + <|im_start|>assistant + <think>...</think>
    """
    birth_date = birth_info.get("生日")
    gender = birth_info.get("性別")
    time_index = birth_info.get("時辰_index")

    SYSTEM_INTRO = load_system_intro()

    system_block = (
        "<|im_start|>system\n"
        f"{SYSTEM_INTRO}\n"
        "<|im_end|>\n"
    )

    user_block = (
        "<|im_start|>user\n"
        f"出生: {birth_date}\n"
        f"性別: {gender}\n"
        f"時辰_index: {time_index}\n"
        "<|im_end|>\n"
    )

    # 這裡固定到 </think>，真正要學的回答會接在後面
    assistant_prefix = "<|im_start|>assistant\n<think>\n\n</think>\n"

    return system_block + user_block + assistant_prefix


def build_training_text(sample):
    """
    sample 來自 jsonl 一行解析後的 dict：
    {
      "出生資料": {...},
      "命盤": {...},
      "解讀": "..."
    }
    回傳一個 dict，裡面有 "text"，給 SFTTrainer 用。
    """
    birth_info = sample["出生資料"]
    answer = build_answer(sample)

    prefix = build_prompt(birth_info)  # 到 </think> 結束
    full_text = prefix + answer        # 真正要學的回答接在後面

    # 如果你想把 assistant 也關掉，可以加上 <|im_end|>
    # full_text = prefix + answer + "<|im_end|>\n"

    return {"text": full_text}

def build_answer(sample) -> str:

    obj = {
        "命盤": sample["命盤"],
        "解讀": sample["解讀"],
    }

    # 讓中文正常顯示
    # 想要好看一點可以加 indent=2
    return json.dumps(obj, ensure_ascii=False)


from dataclasses import dataclass
from typing import List, Dict, Any
import torch

@dataclass
class CompletionOnlyCollator:
    tokenizer: Any
    response_template: str
    max_length: int = 2048

    def __post_init__(self):
        # 將 template 轉成 token id 序列，後面拿來找位置
        self.template_ids = self.tokenizer(
            self.response_template,
            add_special_tokens=False,
        )["input_ids"]

    def _find_template_end(self, ids: List[int]) -> int:
        """
        在 ids 裡面找出 template_ids 的結束位置（end index，exclusive）。
        如果沒找到，就回傳 0（表示不 mask 前綴）。
        """
        t = self.template_ids
        t_len = len(t)
        if t_len == 0:
            return 0

        for i in range(len(ids) - t_len + 1):
            if ids[i : i + t_len] == t:
                return i + t_len  # 回傳「template 結束」的 index
        return 0

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        features 可能有兩種型態：
        1) 原始：{"text": "..."}（我們自己 tokenize）
        2) 已處理：{"input_ids": [...], "attention_mask": [...]}（SFTTrainer 處理過）
        """
        if "text" in features[0]:
            # 情況一：dataset 仍然是文字，這裡做 tokenize + padding
            texts = [f["text"] for f in features]
            batch = self.tokenizer(
                texts,
                padding="max_length",
                # padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        else:
            # 情況二：SFTTrainer 已經把 text 轉成 input_ids 等欄位
            # 這裡只負責 padding + 後續 label masking
            batch = self.tokenizer.pad(
                features,
                padding=True,
                return_tensors="pt",
            )

        input_ids = batch["input_ids"]
        labels = input_ids.clone()

        # 逐條 sample 做 label masking
        for i in range(labels.size(0)):
            ids = labels[i].tolist()
            end = self._find_template_end(ids)
            if end > 0:
                labels[i, :end] = -100  # template 之前的 token 不算 loss
        
        pad_token_id = self.tokenizer.pad_token_id
        if "attention_mask" in batch:
            labels[batch["attention_mask"] == 0] = -100
        elif pad_token_id is not None:
            labels[input_ids == pad_token_id] = -100

        batch["labels"] = labels
        return batch