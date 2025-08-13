#!/usr/bin/env python
import os
import math
import signal
import torch
import argparse
import yaml
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from torchvision.transforms.functional import resize
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment


class ChessDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class AssistantOnlyCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        texts, images, targets = [], [], []
        for ex in examples:
            question = "What do you see here?"
            answer = ex["caption"]
            image = ex["image"]
            messages = [
                {"role": "user", "content": [{"type": "text", "text": question}, {"type": "image"}]},
                {"role": "assistant", "content": [{"type": "text", "text": answer}]}
            ]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text.strip())
            images.append([image])
            targets.append(answer)

        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        for i, (input_ids, target) in enumerate(zip(batch["input_ids"], targets)):
            target_ids = self.processor.tokenizer(target, return_tensors="pt")["input_ids"][0]
            start = self.find_subsequence(input_ids, target_ids)
            if start is not None:
                end = start + len(target_ids)
                if end < len(input_ids) and input_ids[end] == self.processor.tokenizer.eos_token_id:
                    end += 1
                labels[i, :start] = -100
                labels[i, end:] = -100
        batch["labels"] = labels
        return batch

    def find_subsequence(self, seq, subseq):
        for i in range(len(seq) - len(subseq) + 1):
            if torch.equal(seq[i:i + len(subseq)], subseq):
                return i
        return None


class QwenLoraModule(pl.LightningModule):
    def __init__(self, model_id, lr):
        super().__init__()
        self.save_hyperparameters()
        self.processor = AutoProcessor.from_pretrained(model_id)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map="auto"
        )
        lora_config = LoraConfig(
            r=32,
            lora_alpha=16,
            use_rslora=True,
            target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj","mlp.0", "mlp.2"],
            modules_to_save=["lm_head","embed_tokens"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(model, lora_config)

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        out = self(**batch)
        self.log("train_loss", out.loss)
        return out.loss

    def validation_step(self, batch, batch_idx):
        out = self(**batch)
        self.log("val_loss", out.loss)
        return out.loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="qwen2vl_config.yaml")
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_id = args.model_id or config["model_id"]
    dataset_name = args.dataset or config["dataset"]
    batch_size = args.batch_size or config["batch_size"]
    epochs = args.epochs or config["epochs"]
    lr = args.lr or config["lr"]
    grad_accum = args.grad_accum or config["grad_accum"]

    processor = AutoProcessor.from_pretrained(model_id)
    dataset = load_dataset(dataset_name)
    resize_image = lambda ex: {"image": ex["image"].resize((ex["image"].width // 4, ex["image"].height // 4))}
    train_dataset = dataset["train"].map(resize_image)
    val_dataset = dataset["test"].map(resize_image)

    train_ds = ChessDataset(train_dataset)
    val_ds = ChessDataset(val_dataset)
    collator = AssistantOnlyCollator(processor)

    steps = math.ceil(len(train_ds) / (batch_size * grad_accum)) * epochs
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

    logger = TensorBoardLogger("logs", name="qwen2vl")
    model = QwenLoraModule(model_id, lr)

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        logger=logger,
        strategy="ddp",
        plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)]
    )
    trainer.fit(model, train_loader, val_loader)
