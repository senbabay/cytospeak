#!/usr/bin/env python3
import os
import math
import signal
import argparse
import yaml
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from typing import Optional

# # -----------------------
# # Dataset wrapper for Hugging Face datasets
# # -----------------------
# class ChessDataset(Dataset):
#     """Wraps an HF dataset so it behaves like a standard PyTorch Dataset."""
#     def __init__(self, hf_dataset):
#         self.dataset = hf_dataset
#     def __len__(self):
#         return len(self.dataset)
#     def __getitem__(self, idx):
#         return self.dataset[idx]

# -----------------------
# Custom collator for multimodal Qwen input
# -----------------------
class AssistantOnlyCollator:
    """
    Formats examples into Qwen chat template:
    - User message contains text + image placeholder.
    - Assistant message contains only the caption (target).
    - Masks all tokens except the assistant's answer in labels.
    """
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        texts, images, targets = [], [], []
        for ex in examples:
            question = "What do you see here?"
            answer = ex["caption"]
            image = ex["image"]

            # Construct multi-turn conversation with image
            messages = [
                {"role": "user", "content": [{"type": "text", "text": question}, {"type": "image"}]},
                {"role": "assistant", "content": [{"type": "text", "text": answer}]}
            ]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=False)

            texts.append(text.strip())
            images.append([image])
            targets.append(answer)

        # Tokenize and pad batch
        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)

        # Mask all tokens except the assistant's answer in labels
        labels = batch["input_ids"].clone()
        for i, (input_ids, target) in enumerate(zip(batch["input_ids"], targets)):
            # Tokenize just the assistant response
            target_ids = self.processor.tokenizer(target, return_tensors="pt")["input_ids"][0]
            # Find where the assistant tokens start in the input sequence
            # This method ensures we match the tokenized assistant response, not the raw text
            start = self.find_subsequence(input_ids, target_ids)
            if start is not None:
                end = start + len(target_ids)
                # Ensure EOS token remains unmasked
                eos_id = getattr(self.processor.tokenizer, "eos_token_id", None) # guard against missing EOS
                if end < len(input_ids) and eos_id is not None and input_ids[end] == eos_id:
                    end += 1
                # Mask user prompt and everything after answer
                labels[i, :start] = -100
                labels[i, end:] = -100
            else:
                # If the answer sequence isn't found, mask everything to avoid corrupt loss
                labels[i, :] = -100
        batch["labels"] = labels
        return batch

    def find_subsequence(self, seq, subseq):
        """Find exact subsequence match in token IDs (for locating assistant's answer)."""
        for i in range(len(seq) - len(subseq) + 1):
            if torch.equal(seq[i:i + len(subseq)], subseq):
                return i
        return None

# -----------------------
# LightningModule with LoRA-wrapped Qwen
# -----------------------
class QwenLoraModule(pl.LightningModule):
    """
    LightningModule wrapping:
    - Qwen2.5-VL model with LoRA applied
    - AdamW optimizer with HF scheduler
    - Optional gradient checkpointing for VRAM savings
    """
    def __init__(
        self,
        model_id,
        lr,
        weight_decay,
        lr_scheduler_type,
        warmup_steps,
        num_training_steps,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        gradient_checkpointing=True,
        gc_use_reentrant=False,  # False avoids Qwen checkpointing bug
        attn_implementation="eager",
        tf32=True,
        lora_cfg=None,
        device_map=None,
        processor: Optional[AutoProcessor] = None,
    ):
        super().__init__()
        # DON'T serialize huge/dynamic objects
        self.save_hyperparameters(ignore=["num_training_steps","lora_cfg", "processor"])
        self.num_training_steps = num_training_steps

        # Enable TensorFloat32 for faster matmul on Ampere+ GPUs
        if tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

        # Use provided processor or load if missing
        if processor is None:
            self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        else:
            self.processor = processor

        # Load Qwen base model in bf16 for memory savings
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
            device_map=device_map,   # None => Lightning moves it; "auto" => HF places
        )

        # --- Robust gradient checkpointing across Transformers versions ---
        if gradient_checkpointing:
            enabled = False
            try:
                # Newer API accepts kwargs dict
                base_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": gc_use_reentrant}
                )
                enabled = True
            except TypeError:
                pass
            if not enabled:
                try:
                    # Older API: no kwargs
                    base_model.gradient_checkpointing_enable()
                    enabled = True
                except TypeError:
                    # Very old fallbacks
                    if hasattr(base_model, "enable_gradient_checkpointing"):
                        base_model.enable_gradient_checkpointing()
                        enabled = True
                    elif hasattr(base_model, "set_gradient_checkpointing"):
                        base_model.set_gradient_checkpointing(True)
                        enabled = True
            if hasattr(base_model, "enable_input_require_grads"):
                base_model.enable_input_require_grads()
        
        # --- LoRA configuration from YAML ---
        if not isinstance(lora_cfg, dict) or len(lora_cfg) == 0:
            raise ValueError(
                "LoRA configuration is missing. Please provide a 'lora:' section in qwen2vl_config.yaml."
            )
        lora_config = LoraConfig(**lora_cfg)

        # Get LoRA model
        self.model = get_peft_model(base_model, lora_config)

        # Training-friendly defaults
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False
            if getattr(self.model.config, "pad_token_id", None) is None:
                self.model.config.pad_token_id = self.processor.tokenizer.eos_token_id

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        out = self(**batch)
        loss = out.loss

        # ---- Accuracy (token-level) ----
        with torch.no_grad():
            preds = out.logits.argmax(dim=-1)   # [batch, seq]
            labels = batch["labels"]
            mask = labels != -100               # ignore masked tokens
            correct = (preds == labels) & mask
            acc = correct.sum().float() / mask.sum().float()

        # Log both loss and accuracy
        # Log training loss per step (no epoch avg to match HF behavior)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        out = self(**batch)
        loss = out.loss

        # ---- Accuracy (token-level) ----
        preds = out.logits.argmax(dim=-1)
        labels = batch["labels"]
        mask = labels != -100
        correct = (preds == labels) & mask
        acc = correct.sum().float() / mask.sum().float()

        # Log both loss and accuracy (averaged across epoch)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """
        Set up:
        - AdamW optimizer with HF's beta/eps/weight decay settings
        - LR scheduler from transformers.optimization.get_scheduler
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
            eps=self.hparams.adam_epsilon,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = get_scheduler(
            name=self.hparams.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.num_training_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # step-based scheduler like HF
                "frequency": 1,
                "name": "lr"
            }
        }

# -----------------------
# Helper functions
# -----------------------
def as_int_or_str(v, default):
    if v is None:
        return default
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        s = v.strip()
        if s.isdigit():
            return int(s)
        return s  # e.g., "auto"
    return v

def to_int(v, default=None):
    if v is None: return default
    if isinstance(v, int): return v
    if isinstance(v, str) and v.strip().isdigit(): return int(v)
    return int(float(v))  # handles "3.0"

def to_float(v, default=None):
    if v is None: return default
    if isinstance(v, (int, float)): return float(v)
    return float(str(v).strip())


# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="qwen2vl_config.yaml")

    # mirror TrainingArguments you shared
    parser.add_argument("--epochs", type=int, default=None)                     # num_train_epochs
    parser.add_argument("--lr", type=float, default=None)                       # learning_rate
    parser.add_argument("--lr_scheduler_type", type=str, default=None,          # lr_scheduler_type
                        choices=["linear","cosine","cosine_with_restarts","polynomial","constant","constant_with_warmup"])
    parser.add_argument("--batch_size", type=int, default=None)                 # per_device_train_batch_size
    parser.add_argument("--eval_batch_size", type=int, default=None)            # per_device_eval_batch_size
    parser.add_argument("--grad_accum", type=int, default=None)                 # gradient_accumulation_steps
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--logging_fraction", type=float, default=None)         # to derive logging_steps
    parser.add_argument("--eval_fraction", type=float, default=None)            # to derive eval_steps
    parser.add_argument("--warmup_steps", type=int, default=None)               # optional
    parser.add_argument("--warmup_ratio", type=float, default=None)             # optional, if steps not given

    # runtime/system
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--precision", type=str, default=None, choices=["bf16-mixed","16-mixed","32-true"])
    parser.add_argument("--devices", type=str, default=None)
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--no_tf32", dest="tf32", action="store_false")
    parser.set_defaults(tf32=True)

    # Single or Multi GPU training
    parser.add_argument("--device_map", type=str, default=None,
                    help='Hugging Face device_map (e.g., "auto"). Usually leave unset/None when using Lightning.')


    # logging / names / output
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--logging_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    # GC controls (Trainer: gradient_checkpointing, kwargs)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.set_defaults(gradient_checkpointing=True)
    parser.add_argument("--gc_use_reentrant", action="store_true")
    parser.add_argument("--gc_no_reentrant", dest="gc_use_reentrant", action="store_false")
    parser.set_defaults(gc_use_reentrant=False)  # Qwen-friendly default

    # HF Hub
    parser.add_argument("--hub_model_id", type=str, default=None)

    # ---- LoRA CLI overrides 
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=float, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--lora_bias", type=str, default=None, choices=["none","all","lora_only"])
    parser.add_argument("--lora_task_type", type=str, default=None, help="e.g., CAUSAL_LM")

    parser.add_argument("--lora_use_rslora", dest="lora_use_rslora", action="store_true")
    parser.add_argument("--lora_no_use_rslora", dest="lora_use_rslora", action="store_false")
    parser.set_defaults(lora_use_rslora=None)  # None = not provided on CLI

    parser.add_argument("--lora_target_modules", type=str, default=None,
                        help="Comma-separated list e.g. 'q_proj,k_proj,v_proj,o_proj,...'")
    parser.add_argument("--lora_modules_to_save", type=str, default=None,
                        help="Comma-separated list e.g. 'lm_head,embed_tokens'")

    parser.add_argument("--lora_extras_yaml", type=str, default=None,
                        help="Inline YAML/JSON dict of extra LoraConfig fields (e.g., rank_pattern)")

    args = parser.parse_args()

    # Load YAML and merge
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    def pick(key, default=None):
        return getattr(args, key) if getattr(args, key) is not None else config.get(key, default)

    # Map your provided defaults
    epochs          = to_int(pick("epochs", 3))
    lr              = to_float(pick("lr", 1e-4)) 
    lr_scheduler    = pick("lr_scheduler_type", "linear")
    batch_size      = to_int(pick("batch_size", 1))
    eval_batch_size = to_int(pick("eval_batch_size", 1))
    grad_accum      = to_int(pick("grad_accum", 4))
    weight_decay    = pick("weight_decay", 0.01)
    logging_frac    = to_float(pick("logging_fraction", 0.10))
    eval_frac       = to_float(pick("eval_fraction", 0.10))
    warmup_steps_cfg= pick("warmup_steps", None)
    warmup_ratio    = pick("warmup_ratio", None)  # if you decide to use ratio
    if warmup_steps_cfg is not None: warmup_steps_cfg = to_int(warmup_steps_cfg)
    if warmup_ratio is not None:     warmup_ratio     = to_float(warmup_ratio)

    model_id        = pick("model_id")
    dataset_name    = pick("dataset")
    num_workers     = to_int(pick("num_workers", 4))
    precision       = pick("precision", "bf16-mixed")
    # devices/strategy with normalization
    devices_raw     = pick("devices", 1)           # may be int or "auto"
    strategy        = pick("strategy", "auto")
    devices         = as_int_or_str(devices_raw, 1)

    device_map      = pick("device_map", None)

    seed            = pick("seed", 42)
    tf32            = args.tf32 if hasattr(args, "tf32") else config.get("tf32", True)

    run_name        = pick("run_name", f"shenbaba-chess-{lr}_lr-{epochs}_epochs-{lr_scheduler}_schedule-completions-only-annealing")
    logging_dir     = pick("logging_dir", f"./logs/{run_name}")
    output_dir      = pick("output_dir", "fine-tuned-model")

    gradient_checkpointing = args.gradient_checkpointing if hasattr(args, "gradient_checkpointing") else config.get("gradient_checkpointing", True)
    gc_use_reentrant       = args.gc_use_reentrant if hasattr(args, "gc_use_reentrant") else config.get("gc_use_reentrant", False)

    hub_model_id    = pick("hub_model_id", "shenbaba/Qwen2.5-VLM-3B-chess")

    # If Lightning is doing multi-GPU or non-auto strategy, force HF device_map=None
    multi_gpu = (isinstance(devices, int) and devices > 1)
    non_auto_devices = (isinstance(devices, str) and devices not in (None, "auto"))
    if multi_gpu or non_auto_devices or (strategy and strategy != "auto"):
        device_map = None


    # --- LoRA config (YAML + CLI overrides via pick-like behavior) ---
    lora_from_yaml = config.get("lora", {}) or {}
    _default_lora = {
        "r": 32,
        "lora_alpha": 16,
        "use_rslora": True,
        "target_modules": ["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj","mlp.0","mlp.2"],
        "modules_to_save": ["lm_head","embed_tokens"],
        "lora_dropout": 0.1,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }

    def _split_csv(s):
        return [x.strip() for x in s.split(",")] if s else None

    def pick_lora(field, default=None):
        # CLI first
        cli_val = getattr(args, f"lora_{field}", None)
        if cli_val is not None:
            return cli_val
        # YAML next
        if field in lora_from_yaml and lora_from_yaml[field] is not None:
            return lora_from_yaml[field]
        # Fallback
        return _default_lora.get(field, default)

    lora_cfg = {
        "r": to_int(pick_lora("r")),
        "lora_alpha": to_float(pick_lora("lora_alpha")),
        "lora_dropout": to_float(pick_lora("lora_dropout")),
        "use_rslora": pick_lora("use_rslora"),
        "bias": pick_lora("bias"),
        "task_type": pick_lora("task_type"),
        "target_modules": _split_csv(args.lora_target_modules)
                          if args.lora_target_modules is not None
                          else pick_lora("target_modules"),
        "modules_to_save": _split_csv(args.lora_modules_to_save)
                          if args.lora_modules_to_save is not None
                          else pick_lora("modules_to_save"),
    }

    if args.lora_extras_yaml:
        try:
            extra = yaml.safe_load(args.lora_extras_yaml)
            if isinstance(extra, dict):
                lora_cfg.update(extra)
        except Exception:
            pass

    pl.seed_everything(seed, workers=True)

    # Data
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    dataset = load_dataset(dataset_name)

    # Optional: quick shrink for prototyping (same as before)
    resize_image = lambda ex: {"image": ex["image"].resize((ex["image"].width // 4, ex["image"].height // 4))}
    train_dataset = dataset["train"].map(resize_image)
    val_dataset   = dataset["test"].map(resize_image)

    # Replace your current resize lambda with a fixed target size
    TARGET = 448  # try 384, 448
    def resize_image(ex):
        im = ex["image"].convert("RGB")
        return {"image": im.resize((TARGET, TARGET))}

    # train_ds = ChessDataset(train_dataset)
    # val_ds   = ChessDataset(val_dataset)
    collator = AssistantOnlyCollator(processor)

    # Steps accounting to mirror your Trainer math
    dataset_len = len(train_dataset)
    steps_per_epoch = math.ceil(dataset_len / (batch_size * max(1, grad_accum)))
    total_steps = steps_per_epoch * epochs

    # Fractions → concrete steps
    logging_steps = max(1, int(total_steps * float(logging_frac)))

    batches_per_epoch = math.ceil(dataset_len / batch_size) #check this again
    eval_steps = max(1, int(batches_per_epoch * eval_frac)) # check this again
    # eval_steps    = max(1, int(total_steps * float(eval_frac)))

    # Warmup resolution
    if warmup_steps_cfg is not None:
        warmup_steps = int(warmup_steps_cfg)
    elif warmup_ratio is not None:
        warmup_steps = int(total_steps * float(warmup_ratio))
    else:
        warmup_steps = 0  # matches your commented-out default

    # -----------------------
    # Dataloaders
    # -----------------------
    # Pin + persistent workers improve performance on repeated small batches
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=max(1, num_workers // 2),
        pin_memory=True,
        persistent_workers=(max(1, num_workers // 2) > 0),
        drop_last=False,
    )

   # -----------------------
    # Logging & checkpoints
    # -----------------------
    # Logger & callbacks (TensorBoard + step-based eval/checkpointing)
    logger = TensorBoardLogger(
        save_dir=logging_dir,
        name=run_name,
        default_hp_metric=False  # prevents Lightning adding HP metric noise
    )
    ckpt_cb = ModelCheckpoint(
        dirpath=output_dir,
        monitor="val_loss",
        mode="min",
        save_top_k=1,               # keep best model only (like save_total_limit=1)
        every_n_train_steps=eval_steps,  # save on same cadence as eval
        filename="step{step}-valloss{val_loss:.4f}",
        auto_insert_metric_name=False,
        save_last=True,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")

    # Module
    module = QwenLoraModule(
        model_id=model_id,
        lr=lr,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler,
        warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        gradient_checkpointing=gradient_checkpointing,
        gc_use_reentrant=gc_use_reentrant,
        attn_implementation="eager",
        tf32=tf32,
        lora_cfg=lora_cfg,
        device_map=device_map,
        processor=processor,  
    )

    # Trainer (Lightning: step-based val via val_check_interval)
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=devices,
        precision=precision,
        gradient_clip_val=1.0,
        log_every_n_steps=logging_steps,
        logger=logger,
        callbacks=[ckpt_cb, lr_cb],
        accumulate_grad_batches=grad_accum,
        plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)],
        strategy=strategy,
        val_check_interval=eval_steps,  # "eval_strategy=steps"
    )

    trainer.fit(module, train_loader, val_loader)
