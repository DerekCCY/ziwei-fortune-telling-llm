import os
import json
import torch
import wandb
import argparse
import inspect
import pandas as pd
import math
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer

# Try importing SFTConfig
try:
    from trl import SFTConfig
    HAS_SFT_CONFIG = True
except ImportError:
    HAS_SFT_CONFIG = False

SYSTEM_INTRO = ""

def load_system_intro():
    """
    從當前目錄下的 sft_prompt.txt 讀取 system intro，
    SFT 和 evaluation 共用同一份前置說明。
    """
    global SYSTEM_INTRO
    path = "src/prompt/sft_prompt.txt"  # <== 寫死檔名

    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        # 後面接 <INPUT> 比較好讀，多補兩個換行
        SYSTEM_INTRO = text + "\n\n"
        print(f"[System Intro] Loaded from {path}, length={len(SYSTEM_INTRO)} chars.")
    except FileNotFoundError:
        SYSTEM_INTRO = ""
        print(f"[System Intro] File not found at {path}, proceed without system intro.")
    except Exception as e:
        SYSTEM_INTRO = ""
        print(f"[System Intro] Failed to load from {path}: {e}")

# ------------------------------------------------------------------------
# 1. Argument Parsing
# ------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen on Ziwei Doushu tasks")
    
    # Data Arguments
    parser.add_argument("--data_path", type=str, default="/home/ec2-user/ziwei-fortune-telling-llm/data/interpretations/merged/input.jsonl", help="Path to the training data JSONL file")

    # Model Arguments
    parser.add_argument("--model_size", type=str, default="4b", choices=["0.5b", "4b"], help="Model size to use (8b or 4b)")
    
    # Training Hyperparameters
    parser.add_argument("--batch_size", type=int, default=1, help="Per device train batch size")
    parser.add_argument("--grad_acc_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length")

    # Evaluation frequency (every N epochs)
    parser.add_argument(
        "--eval_every_epochs",
        type=int,
        default=1,
        help="Run validation every N epochs (if validation set exists).",
    )

    # Validation / Test split
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Fraction of data used for validation (e.g., 0.1 = 10%)",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Fraction of data used for testing (e.g., 0.1 = 10%)",
    )

    # Scheduler Hyperparameters
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
        help="Learning rate scheduler type"
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio (fraction of total steps)")
    
    # LoRA Hyperparameters
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")

    # Resume training
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a checkpoint directory to resume training from. "
            "If set to 'latest', will automatically use the last checkpoint found in output_dir."
        ),
    )
    
    # WandB and Output
    parser.add_argument("--wandb_project", type=str, default="Ziwei-GenAI-2025_AWS", help="WandB project name")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory for checkpoints")
    
    return parser.parse_args()

# ------------------------------------------------------------------------
# Prompt construction: INPUT (birth info) + OUTPUT (chart + interpretation)
# ------------------------------------------------------------------------
def build_prompt(sample):
    """
    給單一 sample 建立完整 prompt：
    - (前面) SYSTEM_INTRO：從 sft_prompt.txt 讀進來
    - <INPUT>：出生、性別、時辰_index
    - <Question>：統一的任務說明
    - <Natal Chart>：標準 json 命盤
    - <Interpretation>：事業與財運解讀
    """
    global SYSTEM_INTRO

    # ----- 1. Input：出生資料 -----
    birth_info = sample.get("出生資料", {}) or {}
    birth_date = birth_info.get("生日", "Unknown")
    gender = birth_info.get("性別", "Unknown")
    time_index = birth_info.get("時辰_index", "Unknown")

    input_block = (
        "<INPUT>\n"
        f"出生: {birth_date}\n"
        f"性別: {gender}\n"
        f"時辰_index: {time_index}\n"
        "</INPUT>\n"
        # "<Question>\n"
        # "請根據上述出生資料，先自行排出完整的紫微斗數 json 命盤，"
        # "接著依照命盤撰寫未來十年重點在事業與財運的命理解讀，"
        # "輸出格式必須嚴格為 <Natal Chart> + <Interpretation> 兩個區塊。\n"
        # "</Question>\n"
    )

    # ----- 2. Output：命盤 JSON -----
    # 你的標註資料裡命盤 json 一般會放在 "命盤" 或 "chart" 之類欄位
    chart_json = (
        sample.get("命盤")
        or sample.get("chart")
        or sample.get("chart_json")
        or {}
    )
    chart_str = json.dumps(chart_json, ensure_ascii=False, default=str)

    # ----- 3. Output：文字解讀 -----
    # 從 sample 中抓 interpret / 解說 等欄位
    interpretation = (
        sample.get("解讀")
    )

    output_block = (
        "<Natal Chart>\n"
        f"{chart_str}\n"
        # "</Natal Chart>\n"
        # "<Interpretation>\n"
        f"{interpretation}\n"
        # "</Interpretation>\n"
    )

    # ✅ 最終：system_intro + input + output
    return (SYSTEM_INTRO or "") + input_block + output_block

class BestModelCallback(TrainerCallback):
    """
    在 training 過程中，每 N 個 epoch 跑一次 validation，
    用 eval_loss 挑出目前最佳的 model，存到 <output_dir>/best_model。
    """
    def __init__(
        self,
        trainer,
        eval_dataset,
        eval_every_epochs: int,
        output_dir: str,
        metric_name: str = "eval_loss",
        greater_is_better: bool = False,
    ):
        self.trainer = trainer
        self.eval_dataset = eval_dataset
        self.eval_every_epochs = max(1, int(eval_every_epochs))
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.best_metric = None
        self.best_epoch = None
        self.best_model_dir = os.path.join(output_dir, "best_model")
        os.makedirs(self.best_model_dir, exist_ok=True)

    def on_epoch_end(self, args, state, control, **kwargs):
        # 沒有 val_dataset 就不用做事
        if self.eval_dataset is None:
            return

        if state.epoch is None:
            return

        # state.epoch 可能是 float，例如 1.0, 2.0
        epoch = int(state.epoch)
        if epoch <= 0:
            return

        # 不是 eval frequency，就跳過
        if epoch % self.eval_every_epochs != 0:
            return

        print(f"[Eval] Running validation at epoch {epoch}...")
        metrics = self.trainer.evaluate(eval_dataset=self.eval_dataset)
        print(f"[Eval] Metrics at epoch {epoch}: {metrics}")

        metric_value = metrics.get(self.metric_name, None)
        if metric_value is None:
            print(f"[Eval] Metric {self.metric_name} not found in metrics, skip best-model check.")
            return

        is_better = False
        if self.best_metric is None:
            is_better = True
        else:
            if self.greater_is_better:
                is_better = metric_value > self.best_metric
            else:
                is_better = metric_value < self.best_metric

        if is_better:
            self.best_metric = metric_value
            self.best_epoch = epoch
            print(f"[BestModel] New best {self.metric_name}={metric_value:.4f} at epoch {epoch}.")
            print(f"[BestModel] Saving to {self.best_model_dir} ...")
            self.trainer.save_model(self.best_model_dir)
            if self.trainer.tokenizer is not None:
                self.trainer.tokenizer.save_pretrained(self.best_model_dir)
# ------------------------------------------------------------------------
# 3. Main Training Logic
# ------------------------------------------------------------------------
def main():
    args = parse_args()

    load_system_intro()

    # Configuration: mapping from our logical sizes to HF model ids
    MODEL_MAP = {
        "0.5b": "Qwen/Qwen2-0.5B",
        "4b": "Qwen/Qwen3-4B",
    }

    MODEL_NAME = MODEL_MAP[args.model_size]
    NEW_MODEL_NAME = f"Ziwei-Doushu-{args.model_size.upper()}-SFT"

    # WandB metadata directory (under output_dir）
    os.makedirs(args.output_dir, exist_ok=True)
    WANDB_META_PATH = os.path.join(args.output_dir, "wandb_meta.json")

    timestamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M")
    WANDB_RUN_NAME = (
        f"run-{args.model_size}-lr{args.learning_rate}-bs{args.batch_size}-{timestamp}"
    )

    # --------------------------------------------------------------------
    # Load Data
    # --------------------------------------------------------------------
    print(f"Loading data from: {args.data_path}")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found at {args.data_path}")

    raw_dataset = load_dataset("json", data_files=args.data_path, split="train")
    print(f"Loaded {len(raw_dataset)} samples.")
    print(f"Sample keys found: {list(raw_dataset[0].keys())}")

    # --------------------------------------------------------------------
    # Split into train / val / test
    # --------------------------------------------------------------------
    test_ratio = max(0.0, min(args.test_ratio, 0.5)) # ensure not over 50%
    val_ratio = max(0.0, min(args.val_ratio, 0.5)) # ensure not over 50%

    if test_ratio > 0:
        dataset_dict = raw_dataset.train_test_split(test_size=test_ratio, seed=42)
        train_val_raw = dataset_dict["train"]
        test_raw = dataset_dict["test"]
    else:
        train_val_raw = raw_dataset
        test_raw = None

    if val_ratio > 0:
        # val_ratio 是相對 whole dataset，要換算成相對 train_val 的比例
        if test_ratio < 1.0:
            val_in_train_val = val_ratio / (1.0 - test_ratio)
        else:
            val_in_train_val = 0.0
        if val_in_train_val > 0:
            tv_split = train_val_raw.train_test_split(test_size=val_in_train_val, seed=42)
            train_raw = tv_split["train"]
            val_raw = tv_split["test"]
        else:
            train_raw = train_val_raw
            val_raw = None
    else:
        train_raw = train_val_raw
        val_raw = None

    print(f"Train size: {len(train_raw)}")
    print(f"Val size:   {len(val_raw) if val_raw is not None else 0}")
    print(f"Test size:  {len(test_raw) if test_raw is not None else 0}")

    # --------------------------------------------------------------------
    # Model Loading (4-bit base + LoRA on top)
    # --------------------------------------------------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    print(f"Loading Base Model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = args.max_seq_length

    # LoRA configuration (PEFT)
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj", # attention modules
            "o_proj","gate_proj","up_proj","down_proj",],) # Feed-Forward modules

    # --------------------------------------------------------------------
    # Preprocess dataset: tokenize + mask input labels
    # --------------------------------------------------------------------
    def preprocess_function(example):
        text = build_prompt(example)
        
        # Tokenization
        tokenized = tokenizer(
            text,
            max_length=args.max_seq_length,
            truncation=True,
            padding="max_length",
        )
        
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        # 建立 labels，預設全為 -100 (忽略計算 loss)
        labels = [-100] * len(input_ids)
        
        # -------------------------------------------------------
        # 修正 1: 準確找到 User Input 與 Model Output 的邊界
        # 你的 Prompt 結構是: ... </Question>\n<Natal Chart> ...
        # 我們希望模型從 "<Natal Chart>" 開始預測
        # -------------------------------------------------------
        
        # 為了避免 tokenizer 對空白符號處理的差異，建議先 tokenize 整個 text，
        # 再 tokenize "Prompt 部分"，計算長度來做 mask
        
        # 你的 prompt 結束點應該是在 <Question> 區塊結束之後
        # 讓我們定義一個明確的分隔符，根據你的 build_prompt，
        # output 是從 <Natal Chart> 開始
        split_token = "<Natal Chart>"
        
        try:
            # 找到分隔符在純文字中的位置
            split_idx = text.index(split_token)
            # 取得 Prompt 部分的文字 (包含 System, Input, Question)
            prompt_text = text[:split_idx]
            
            # 將 Prompt 部分轉為 token id
            prompt_ids = tokenizer(
                prompt_text, 
                add_special_tokens=True, # 確保開頭處理一致
                truncation=True, 
                max_length=args.max_seq_length
            )["input_ids"]
            
            prompt_len = len(prompt_ids)
            
        except ValueError:
            # 如果找不到分隔符，這筆資料可能有問題，全部 mask 掉或設為 0
            prompt_len = 0
            print(f"Warning: split token '{split_token}' not found in text.")

        # -------------------------------------------------------
        # 修正 2: 填入 Label 並處理 Padding
        # -------------------------------------------------------
        for i in range(len(input_ids)):
            # 條件 A: 如果是 Padding (attention_mask == 0)，保持 -100
            if attention_mask[i] == 0:
                labels[i] = -100
            # 條件 B: 如果在 Prompt 範圍內，保持 -100
            elif i < prompt_len:
                labels[i] = -100
            # 條件 C: 剩下的就是真正的 Output，填入 input_id 讓模型學習
            else:
                labels[i] = input_ids[i]

        # 確保 truncation 沒有切掉所有的 labels
        # 如果 prompt_len >= max_seq_length，這筆資料就廢了
        
        tokenized["labels"] = labels
        # print(len(labels))
        # if all(l == -100 for l in labels):
        #     # print("Warning: All labels are -100 after processing. Check prompt construction and tokenization.")
        #     raise ValueError("All labels are -100 after processing. Check prompt construction and tokenization.")
        return tokenized

    print("Tokenizing and preparing dataset (this may take a while)...")
    train_dataset = train_raw.map(
        preprocess_function,
        remove_columns=train_raw.column_names,
    )
    # 🔍 這裡加 debug，看 label 到底長怎樣
    print("=== DEBUG: check labels of first train sample ===")
    sample = train_dataset[0]
    labels = sample["labels"]
    print("Unique label values:", set(labels))
    print("Number of tokens with label != -100:",
          sum(1 for x in labels if x != -100))
    print(sample["attention_mask"])

    val_dataset = None
    if val_raw is not None:
        val_dataset = val_raw.map(
            preprocess_function,
            remove_columns=val_raw.column_names,
        )

    test_dataset = None
    if test_raw is not None:
        test_dataset = test_raw.map(
            preprocess_function,
            remove_columns=test_raw.column_names,
        )

    # --------------------------------------------------------------------
    # TrainingArguments & WandB
    # --------------------------------------------------------------------
    common_args = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1, 
        gradient_accumulation_steps=args.grad_acc_steps,
        optim="paged_adamw_32bit",
        save_strategy="no",
        # save_steps=25,
        logging_steps=5,
        learning_rate=args.learning_rate,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=args.warmup_ratio,
        group_by_length=True,
        lr_scheduler_type=args.lr_scheduler_type,
        report_to="wandb",
        run_name=WANDB_RUN_NAME,
    )

    # --------------------------------------------------------------------
    # WandB：支援新 run / 接續舊 run
    # --------------------------------------------------------------------
    wandb_id = None
    # 如果有要求 resume，就試著從 output_dir 裡讀取舊的 run_id
    if args.resume_from_checkpoint and os.path.exists(WANDB_META_PATH):
        try:
            with open(WANDB_META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
            wandb_id = meta.get("run_id")
            # 也順便沿用舊的 run_name，比較乾淨
            if meta.get("run_name"):
                WANDB_RUN_NAME = meta["run_name"]
            print(f"[WandB] Resuming existing run: id={wandb_id}, name={WANDB_RUN_NAME}")
        except Exception as e:
            print(f"[WandB] Failed to load wandb_meta.json: {e}")
            print("[WandB] A new run will be created.")

    if wandb_id:
        # ✅ 接續同一個 WandB run
        run = wandb.init(
            project=args.wandb_project,
            name=WANDB_RUN_NAME,
            config=common_args,
            id=wandb_id,
            resume="must",
        )
    else:
        # ✅ 建立新的 WandB run，並把 run_id 存起來
        run = wandb.init(
            project=args.wandb_project,
            name=WANDB_RUN_NAME,
            config=common_args,
        )
        try:
            with open(WANDB_META_PATH, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "run_id": run.id,
                        "run_name": run.name,
                        "project": args.wandb_project,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"[WandB] Saved run metadata to {WANDB_META_PATH}")
        except Exception as e:
            print(f"[WandB] Failed to save wandb_meta.json: {e}")

    training_args = TrainingArguments(**common_args)

    # --------------------------------------------------------------------
    # Initialize SFTTrainer
    # --------------------------------------------------------------------
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU count:", torch.cuda.device_count())
        print("Current GPU:", torch.cuda.current_device(), torch.cuda.get_device_name())

    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,   # 可以是 None，Trainer 會自己處理
        peft_config=peft_config,
    )
    print("Trainer / accelerator device:", trainer.accelerator.device)

    # --------------------------------------------------------------------
    # Attach BestModelCallback：每 N 個 epoch 做一次 validation 並存最佳模型
    # --------------------------------------------------------------------
    if val_dataset is not None and args.eval_every_epochs and args.eval_every_epochs > 0:
        print(f"[Callback] Enable BestModelCallback: eval every {args.eval_every_epochs} epochs.")
        best_model_cb = BestModelCallback(
            trainer=trainer,
            eval_dataset=val_dataset,
            eval_every_epochs=args.eval_every_epochs,
            output_dir=args.output_dir,
            metric_name="eval_loss",
            greater_is_better=False,  # loss 越小越好
        )
        trainer.add_callback(best_model_cb)
    else:
        print("[Callback] BestModelCallback disabled (no val set or eval_every_epochs <= 0).")

    # --------------------------------------------------------------------
    # Train & Save
    # --------------------------------------------------------------------
    # 決定是否接續訓練
    resume_checkpoint = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            # 自動從 output_dir 找最後一個 checkpoint
            last_ckpt = get_last_checkpoint(args.output_dir)
            if last_ckpt is None:
                print(f"[Resume] No checkpoint found in {args.output_dir}, starting from scratch.")
            else:
                resume_checkpoint = last_ckpt
                print(f"[Resume] Resuming from latest checkpoint: {resume_checkpoint}")
        else:
            # 使用使用者指定的 checkpoint 路徑
            if os.path.isdir(args.resume_from_checkpoint):
                resume_checkpoint = args.resume_from_checkpoint
                print(f"[Resume] Resuming from checkpoint: {resume_checkpoint}")
            else:
                print(f"[Resume] Provided checkpoint path does not exist: {args.resume_from_checkpoint}")
                print("[Resume] Training will start from scratch.")

    # --------------------------------------------------------------------
    # Train & Save
    # --------------------------------------------------------------------
    # 🔧 Workaround: 如果要 resume，就移除 AMP scaler 檔案，避免 fp16 scaler mismatch
    if resume_checkpoint is not None:
        amp_scaler_path = os.path.join(resume_checkpoint, "amp_scaler.pt")
        if os.path.exists(amp_scaler_path):
            print(f"[Resume] Removing AMP scaler state at {amp_scaler_path} to avoid scaler load issues.")
            try:
                os.remove(amp_scaler_path)
            except Exception as e:
                print(f"[Resume] Failed to remove {amp_scaler_path}: {e}")

    print(f"Starting training for {args.num_epochs} epochs...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # --------------------------------------------------------------------
    # Validation summary（已經在訓練過程中自動 log 到 WandB）
    # --------------------------------------------------------------------
    if val_dataset is not None:
        print("Running final evaluation on validation set...")
        val_results = trainer.evaluate(eval_dataset=val_dataset)
        print("Final validation results:", val_results)
        try:
            if "eval_loss" in val_results:
                val_ppl = math.exp(val_results["eval_loss"])
                print(f"Validation perplexity: {val_ppl:.4f}")
                wandb.log({
                    "final/val_loss": val_results["eval_loss"],
                    "final/val_perplexity": val_ppl,
                })
            else:
                wandb.log({f"final/{k}": v for k, v in val_results.items()})
        except Exception as e:
            print(f"[WandB] Failed to log final validation metrics: {e}")

    # --------------------------------------------------------------------
    # Testing：在獨立 test set 上做一次 evaluate，metric_key_prefix='test'
    # --------------------------------------------------------------------
    if test_dataset is not None:
        print("Running evaluation on test set...")
        test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
        print("Test results:", test_results)
        try:
            if "test_loss" in test_results:
                test_ppl = math.exp(test_results["test_loss"])
                print(f"Test perplexity: {test_ppl:.4f}")
                wandb.log({
                    "final/test_loss": test_results["test_loss"],
                    "final/test_perplexity": test_ppl,
                })
            else:
                wandb.log({f"final/{k}": v for k, v in test_results.items()})
        except Exception as e:
            print(f"[WandB] Failed to log final test metrics: {e}")

    # print(f"Saving model to {NEW_MODEL_NAME}...")
    # trainer.model.save_pretrained(NEW_MODEL_NAME)
    # tokenizer.save_pretrained(NEW_MODEL_NAME)
    # print("Done!")
    print("Training finished. Best model (by eval_loss) is saved under <output_dir>/best_model.")


if __name__ == "__main__":
    main()