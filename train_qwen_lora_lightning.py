#!/usr/bin/env python
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

# -----------------------
# Dataset wrapper for Hugging Face datasets
# -----------------------
class ChessDataset(Dataset):
    """Wraps an HF dataset so it behaves like a standard PyTorch Dataset."""
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        return self.dataset[idx]

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
            target_ids = self.processor.tokenizer(target, return_tensors="pt")["input_ids"][0]
            start = self.find_subsequence(input_ids, target_ids)
            if start is not None:
                end = start + len(target_ids)
                if end < len(input_ids) and input_ids[end] == self.processor.tokenizer.eos_token_id:
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
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["num_training_steps"])
        self.num_training_steps = num_training_steps

        # Enable TensorFloat32 for faster matmul on Ampere+ GPUs
        if tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        # Processor handles both text & image preprocessing
        self.processor = AutoProcessor.from_pretrained(model_id)

        # Load Qwen base model in bf16 for memory savings
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
            device_map="auto"
        )

        # Enable gradient checkpointing if requested (VRAM reduction)
        if gradient_checkpointing:
            base_model.gradient_checkpointing_enable(use_reentrant=gc_use_reentrant)
            base_model.enable_input_require_grads()

        # Configure LoRA adapters for transformer layers & embedding/lm_head
        lora_config = LoraConfig(
            r=32,
            lora_alpha=16,
            use_rslora=True,
            target_modules=["q_proj","k_proj","v_proj","o_proj",
                            "up_proj","down_proj","gate_proj","mlp.0","mlp.2"],
            modules_to_save=["lm_head","embed_tokens"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(base_model, lora_config)

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        out = self(**batch)
        # Log training loss per step (no epoch avg to match HF behavior)
        self.log("train_loss", out.loss, prog_bar=True, on_step=True, on_epoch=False)
        return out.loss

    def validation_step(self, batch, batch_idx):
        out = self(**batch)
        # Log validation loss averaged over an epoch
        self.log("val_loss", out.loss, prog_bar=True, on_step=False, on_epoch=True)
        return out.loss

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
    parser.add_argument("--devices", type=int, default=None)
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--no_tf32", dest="tf32", action="store_false")
    parser.set_defaults(tf32=True)

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

    args = parser.parse_args()

    # Load YAML and merge
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    def pick(key, default=None):
        return getattr(args, key) if getattr(args, key) is not None else config.get(key, default)

    # Map your provided defaults
    epochs          = pick("epochs", 3)
    lr              = pick("lr", 1e-4)
    lr_scheduler    = pick("lr_scheduler_type", "linear")
    batch_size      = pick("batch_size", 1)
    eval_batch_size = pick("eval_batch_size", 1)
    grad_accum      = pick("grad_accum", 4)
    weight_decay    = pick("weight_decay", 0.01)
    logging_frac    = pick("logging_fraction", 0.10)
    eval_frac       = pick("eval_fraction", 0.10)
    warmup_steps_cfg= pick("warmup_steps", None)
    warmup_ratio    = pick("warmup_ratio", None)  # if you decide to use ratio

    model_id        = pick("model_id")
    dataset_name    = pick("dataset")
    num_workers     = pick("num_workers", 4)
    precision       = pick("precision", "bf16-mixed")
    devices         = pick("devices", 1)
    strategy        = pick("strategy", "ddp")
    seed            = pick("seed", 42)
    tf32            = args.tf32 if "tf32" in args else config.get("tf32", True)

    run_name        = pick("run_name", f"trelis-chess-{lr}_lr-{epochs}_epochs-{lr_scheduler}_schedule-completions-only-annealing")
    logging_dir     = pick("logging_dir", f"./logs/{run_name}")
    output_dir      = pick("output_dir", "fine-tuned-model")

    gradient_checkpointing = args.gradient_checkpointing if "gradient_checkpointing" in args else config.get("gradient_checkpointing", True)
    gc_use_reentrant       = args.gc_use_reentrant if "gc_use_reentrant" in args else config.get("gc_use_reentrant", False)

    hub_model_id    = pick("hub_model_id", "Trelis/Qwen2.5-VLM-3B-chess")

    pl.seed_everything(seed, workers=True)

    # Data
    processor = AutoProcessor.from_pretrained(model_id)
    dataset = load_dataset(dataset_name)

    # Optional: quick shrink for prototyping (same as before)
    resize_image = lambda ex: {"image": ex["image"].resize((ex["image"].width // 4, ex["image"].height // 4))}
    train_dataset = dataset["train"].map(resize_image)
    val_dataset   = dataset["test"].map(resize_image)

    train_ds = ChessDataset(train_dataset)
    val_ds   = ChessDataset(val_dataset)
    collator = AssistantOnlyCollator(processor)

    # Steps accounting to mirror your Trainer math
    dataset_len = len(train_ds)
    steps_per_epoch = math.ceil(dataset_len / (batch_size * max(1, grad_accum)))
    total_steps = steps_per_epoch * epochs

    # Fractions → concrete steps
    logging_steps = max(1, int(total_steps * float(logging_frac)))
    eval_steps    = max(1, int(total_steps * float(eval_frac)))

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
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=max(1, num_workers // 2),
        pin_memory=True,
        persistent_workers=True,
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
