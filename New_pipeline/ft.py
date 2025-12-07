from helper import build_training_text
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from datasets import load_dataset
# from trl.trainer.utils import DataCollatorForCompletionOnlyLM
from helper import CompletionOnlyCollator

# 1. 設定模型名稱 (使用 Qwen2.5-3B)
model_name = "Qwen/Qwen3-4b"

# 2. 準備一個極簡的測試數據集 (格式：Prompt -> Response)
# 實際使用時，請替換成讀取你的 JSON/CSV 檔案

raw_dataset = load_dataset(
    "json",
    data_files="data/train.jsonl",
    split="train"
)

# 映射成 {"text": "..."}
dataset = raw_dataset.map(
    build_training_text,
    remove_columns=raw_dataset.column_names,
)

print("dataset columns:", dataset.column_names)

# for i in range(2):  # 看前兩筆就好
#     print("\n================ SAMPLE", i, "================")
#     print(dataset[i]["text"])
#     print("================ END SAMPLE ================\n")

# 3. 載入 Tokenizer 與 模型
tokenizer = AutoTokenizer.from_pretrained(model_name)



# tokenizer.pad_token = tokenizer.eos_token # Qwen 需要這行設定

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16, # 如果顯卡舊，改用 torch.float16
    device_map="auto"
)

# TMP
# 優先用模型已經設定的 pad_token_id
pad_id = None

if model.generation_config.pad_token_id is not None:
    print(f"PAD using model.generation_config")
    pad_id = model.generation_config.pad_token_id
elif tokenizer.pad_token_id is not None:
    print(f"PAD using tokenizer")
    pad_id = tokenizer.pad_token_id
else:
    # 實在找不到，就退一步用 eos 當 pad
    print("PAD using tokenizer.eos_token_id")
    pad_id = tokenizer.eos_token_id

# 設定 tokenizer 的 pad_token_id
tokenizer.pad_token_id = pad_id
# 也可以順便設一下 pad_token 方便 debug
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.convert_ids_to_tokens(pad_id)

# 同步回 model.config / generation_config
model.config.pad_token_id = pad_id
model.generation_config.pad_token_id = pad_id
# TMP

# 4.設定 LoRA 參數 (最關鍵的部分)
peft_config = LoraConfig(
    r=64,                       # LoRA 的秩，越大參數量越多
    lora_alpha=32,              # 縮放係數，通常是 r 的兩倍
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"] # 指定要微調的層
)

# 5. 設定訓練參數
training_args = SFTConfig(
    output_dir="./qwen_lora_output",   # 模型儲存路徑
    # max_steps=3,                     # ⬅ 建議拿掉，改用 num_train_epochs
    num_train_epochs=5,                # 訓練 5 個 epochs
    per_device_train_batch_size=2,
    gradient_accumulation_steps=5,     # 梯度累積 5 次
    learning_rate=2e-4,
    logging_steps=1,
    dataset_text_field="text",
    packing=False,
)

# 6. 開始訓練

response_template = "<|im_start|>assistant\n<think>\n\n</think>\n"

data_collator = CompletionOnlyCollator(
    tokenizer=tokenizer,
    response_template=response_template,
    max_length=4096,  # 看你模型的 context 長度
)

# ! debug
debug = True
if debug is True:
    # 只拿一筆樣本出來測試
    sample = dataset[0]           # {'text': '......'}
    batch = data_collator([sample])  # 模擬一個 batch，batch_size=1

    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]

    print("input_ids shape:", input_ids.shape)
    print("labels shape:", labels.shape)

    # 找出第一個不是 -100 的位置（理論上就是 template 後第一個 token）
    first_non_mask = (labels != -100).nonzero(as_tuple=True)[0][0].item()
    print("first non-masked index:", first_non_mask)

    print("\n=== Tokens around boundary ===")
    start = max(0, first_non_mask - 20)
    end = first_non_mask + 20

    for idx in range(start, end):
        tid = input_ids[idx].item()
        lid = labels[idx].item()
        tok = tokenizer.decode([tid])
        flag = "L" if lid == -100 else "T"  # L=masked(label -100), T=trained
        print(f"{idx:4d} | {flag} | {repr(tok)} | label={lid}")

    print("\n=== Decoded full text ===")
    print(tokenizer.decode(input_ids, skip_special_tokens=False))

    print("\n=== 最後 40 個 token（含 padding）===")
    end = len(input_ids)
    start = max(0, end - 40)
    # ===== Debug：確認 padding 有沒有被 mask 掉 =====
    sample = dataset[0]                    # 拿第一筆樣本來測
    batch = data_collator([sample])        # 模擬一個 batch（size=1）

    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]
    attn = batch["attention_mask"][0]

    print("input_ids shape:", input_ids.shape)
    print("labels shape:", labels.shape)

    # 找出 attention_mask = 0 的 index（也就是 padding 的位置）
    pad_positions = (attn == 0).nonzero(as_tuple=True)[0].tolist()
    print("pad positions:", pad_positions)

    if pad_positions:
        print("\n=== 檢查每一個 padding 位置的 label ===")
        for idx in pad_positions[:20]:  # 最多印前 20 個
            tid = input_ids[idx].item()
            lid = labels[idx].item()
            tok = tokenizer.decode([tid])
            print(f"idx={idx:4d} | token={repr(tok)} | label={lid}")
    else:
        print("這筆 sample 沒有 padding（全部都是真實 token）。")


    # 快速 sanity check：所有 padding 位置的 label 是否都是 -100
    if pad_positions:
        ok = (labels[attn == 0] == -100).all().item()
        print("\n所有 padding 的 label 是否都是 -100？", ok)
    for idx in range(start, end):
        tid = input_ids[idx].item()
        lid = labels[idx].item()
        am = attn[idx].item()
        tok = tokenizer.decode([tid])
        flag = "PAD" if am == 0 else "TOK"
        print(f"{idx:4d} | {flag} | token={repr(tok)} | label={lid}")
    
    print("\n=== Debug pad alignment ===")
    print("tokenizer.pad_token_id:", tokenizer.pad_token_id)
    print("model.config.pad_token_id:", model.config.pad_token_id)
    print("model.generation_config.pad_token_id:", model.generation_config.pad_token_id)

    print("tokenizer.eos_token_id:", tokenizer.eos_token_id)
    print("model.config.eos_token_id:", model.config.eos_token_id)
    print("model.generation_config.eos_token_id:", model.generation_config.eos_token_id)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=training_args,
    data_collator=data_collator,  # ⭐ 關鍵在這行
)

print("開始訓練...")
trainer.train()

# 7. 儲存模型 (只儲存 LoRA adapter，檔案很小)
trainer.save_model("./qwen_lora_output")
print("模型已儲存至 ./qwen_lora_output")