
# %%
import torch
import pytorch_lightning as pl
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from viscy.data.triplet import TripletDataModule
from viscy.transforms import ScaleIntensityRangePercentilesd, NormalizeSampled, Decollated
import pandas as pd
import yaml
from peft import PeftModel, LoraConfig, get_peft_model
from typing import Dict, Tuple, Any, Iterable, List

# %% ---- Load same config as training ----
config_file = "qwen_predict_notebooks.yaml"
with open(config_file, "r") as f:
    config = yaml.safe_load(f)

model_id = config.get("model_id")
ckpt_path = "/hpc/projects/intracellular_dashboard/organelle_box/cytospeak/Model/last-v2.ckpt"   # <-- change to your best checkpoint path

# %% --- QwenLoraModule ---
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
    ):
        super().__init__()
        # Save all hparams including lora_cfg (except num_training_steps which is large/dynamic)
        self.save_hyperparameters(ignore=["num_training_steps"])
        self.num_training_steps = num_training_steps

        # Enable TensorFloat32 for faster matmul on Ampere+ GPUs
        if tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        # Processor handles both text & image preprocessing
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

        # Load Qwen base model in bf16 for memory savings
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
            device_map=device_map,
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

class CollatingLoader:
    """Wrap an existing DataLoader and apply `collator` to each batch."""
    def __init__(self, base_loader, collator):
        self.base_loader = base_loader
        self.collator = collator

    def __iter__(self):
        for raw_batch in self.base_loader:
            yield self.collator(raw_batch)

    def __len__(self):
        return len(self.base_loader)

    # Handy passthroughs so PL can read dataset/sampler/batch_size, etc.
    @property
    def dataset(self):
        return self.base_loader.dataset

    @property
    def batch_size(self):
        return getattr(self.base_loader, "batch_size", None)

    @property
    def sampler(self):
        return getattr(self.base_loader, "sampler", None)

    @property
    def batch_sampler(self):
        return getattr(self.base_loader, "batch_sampler", None)

    def __getattr__(self, name):
        # delegate anything else to the base loader (e.g., drop_last, pin_memory, etc.)
        return getattr(self.base_loader, name)

class QwenIndexAnchorCollator:
    """
    Expects a batch:
      batch = {
        "index": {
          "fov_name": list[str],   # len B
          "track_id": Tensor[B],   # int
          "t":        Tensor[B],   # int
          "parent_id":Tensor[B],   # int
          # other fields are ignored
        },
        "anchor": Tensor[B, 3, 1, H, W],  # images
      }

    Produces a dict ready for Qwen forward(**dict), with assistant-only labels.
    """
    def __init__(
        self,
        processor,
        anno_lookup: Dict[Tuple[Any,...], Dict[str,Any]],
        question: str = (
                            "You are given a fluorescence microscopy image.\n\n"
                            "Task: classify three attributes.\n"
                            "Output format: exactly three words separated by single spaces, in this order: "
                            "ORGANELLE PHASE INFECTION\n"
                            "Allowed vocabularies:\n"
                            "- ORGANELLE ∈ {ER, mitochondria, golgi, lysosome, nucleus, stress_granule}\n"
                            "- PHASE ∈ {interphase, mitotic}\n"
                            "- INFECTION ∈ {infected, uninfected}\n"
                            "Rules: no punctuation, no explanations, no quotes, no newlines. "
                            "If uncertain, guess the most likely label from the allowed set."
                        ),
        pad_to_multiple_of: int | None = 8,
        fail_on_missing: bool = True,  # True: error; False: skip unmatched rows
    ):
        self.processor = processor
        self.anno_lookup = anno_lookup
        self.question = question
        self.pad_to_multiple_of = pad_to_multiple_of
        self.fail_on_missing = fail_on_missing

        # cache tokenizer ids we use often
        self.pad_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        self.eos_id = getattr(self.processor.tokenizer, "eos_token_id", None)

    @staticmethod
    def _find_subsequence(seq: torch.Tensor, subseq: torch.Tensor) -> int | None:
        n, m = len(seq), len(subseq)
        if m == 0 or m > n:
            return None
        # naive scan is fine at typical batch sizes
        for s in range(n - m + 1):
            if torch.equal(seq[s:s+m], subseq):
                return s
        return None

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        idx = batch["index"]
        imgs = batch["anchor"]  # [B, 3, 1, H, W]

        # 1) images -> [B, 3, H, W]
        if imgs.ndim == 5 and imgs.shape[2] == 1:
            imgs = imgs.squeeze(2)
        assert imgs.ndim == 4 and imgs.shape[1] == 3, f"Expected [B,3,H,W], got {tuple(imgs.shape)}"

        B = imgs.shape[0]

        # 2) extract quadruplets
        fovs = [str(s) for s in idx["fov_name"]]
        tids = idx["track_id"].tolist() if torch.is_tensor(idx["track_id"]) else list(idx["track_id"])
        ts   = idx["t"].tolist()        if torch.is_tensor(idx["t"])        else list(idx["t"])
        pids = idx["parent_id"].tolist()if torch.is_tensor(idx["parent_id"]) else list(idx["parent_id"])

        texts: List[str] = []
        images_for_qwen: List[List[torch.Tensor]] = []
        targets: List[str] = []

        # 3) per-sample pairing + message building
        kept = 0
        for i in range(B):
            key = (str(fovs[i]), int(tids[i]), int(ts[i]), int(pids[i]))
            row = self.anno_lookup.get(key)
            if row is None:
                if self.fail_on_missing:
                    raise KeyError(f"Missing annotation for quadruplet {key}")
                else:
                    continue

            caption = str(row["__caption__"]).strip()

            # ensure per-image tensor is [3,H,W] (not HxWx3)
            im = imgs[i]
            if im.ndim == 3 and im.shape[0] != 3 and im.shape[-1] == 3:
                im = im.permute(2,0,1)

            # Qwen chat
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": self.question},
                    {"type": "image"},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": caption}
                ]},
            ]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=False)

            texts.append(text.strip())
            images_for_qwen.append([im])   # list-of-images per sample
            targets.append(caption)
            kept += 1

        if kept == 0:
            raise RuntimeError("After filtering/missing annotations, the batch is empty.")

        # 4) processor (tokenize + patchify)
        out = self.processor(
            text=texts,
            images=images_for_qwen,
            return_tensors="pt",
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of
        )

        # 5) assistant-only labels
        labels = out["input_ids"].clone()

        # tokenize targets once (batched)
        tgt_tok = self.processor.tokenizer(
            targets, add_special_tokens=False, padding=True, return_tensors="pt"
        )
        for i, input_ids in enumerate(out["input_ids"]):
            tgt_ids = tgt_tok["input_ids"][i]
            if self.pad_id is not None:
                tgt_ids = tgt_ids[tgt_ids != self.pad_id]
            start = self._find_subsequence(input_ids, tgt_ids)
            if start is not None:
                end = start + len(tgt_ids)
                if self.eos_id is not None and end < len(input_ids) and input_ids[end].item() == self.eos_id:
                    end += 1
                labels[i, :start] = -100
                labels[i, end:]   = -100
            else:
                labels[i, :] = -100  # safe fallback if not found

        out["labels"] = labels
        return out

def build_annotation_lookup(
    csv_or_df: Any,
    key_cols=("fov_name","track_id","t","parent_id"),
    value_cols=("organelle","predicted_cellstate","predicted_infection"),
    caption_col: str | None = None,
):
    """Return dict[(fov_name,track_id,t,parent_id)] -> {value_cols..., '__caption__': str}"""
    df = pd.read_csv(csv_or_df) if isinstance(csv_or_df, (str, bytes)) else csv_or_df.copy()

    # normalize dtypes for exact matching
    df["fov_name"] = df["fov_name"].astype(str)
    for c in ("track_id","t","parent_id"):
        df[c] = pd.to_numeric(df[c], downcast="integer")

    def make_caption(row):
        if caption_col and caption_col in row and pd.notna(row[caption_col]):
            return str(row[caption_col]).strip()
        org   = str(row.get("organelle","unknown")).strip()
        phase = str(row.get("predicted_cellstate","unknown")).strip()
        inf   = str(row.get("predicted_infection","unknown")).strip()
        return f"{org}; {phase}; {inf}"

    lookup: Dict[Tuple[Any,...], Dict[str,Any]] = {}
    for _, row in df.iterrows():
        key = tuple(row[c] for c in key_cols)
        payload = {c: row[c] for c in value_cols if c in df.columns}
        payload["__caption__"] = make_caption(row)
        lookup[key] = payload
    return lookup

# %% --- Rebuild processor ---
processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
processor.image_processor.do_rescale = False
processor.image_processor.do_normalize = False
processor.image_processor.do_convert_rgb = False

# Fix the padding warning by setting padding_side to 'left'
processor.tokenizer.padding_side = 'left'

# --- Data Module (same as training) ---
data_path = "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV/4-phenotyping/train-test/2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV.zarr"        # <-- change to your paths
tracks_path = "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV/1-preprocess/label-free/3-track/2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV_cropped.zarr"
source_channel = ["Phase3D","GFP EX488 EM525-45","mCherry EX561 EM600-37"]

dm = TripletDataModule(
    data_path=data_path,
    tracks_path=tracks_path,
    source_channel=source_channel,
    batch_size=10,
    num_workers=4,
    z_range=(0,1),
    initial_yx_patch_size=(160, 160),
    final_yx_patch_size=(160,160),
    normalizations=[
        NormalizeSampled(keys=["Phase3D"], level="fov_statistics", subtrahend="mean", divisor="std"),
        Decollated(keys=source_channel),
        ScaleIntensityRangePercentilesd(keys=["GFP EX488 EM525-45"], lower=50, upper=99, b_min=0.0, b_max=1.0),
        ScaleIntensityRangePercentilesd(keys=["mCherry EX561 EM600-37"], lower=50, upper=99, b_min=0.0, b_max=1.0),
    ],
    return_negative=False
)
dm.prepare_data()
dm.setup("predict")
predict_loader = dm.predict_dataloader()

# %% --- Load trained LoRA-wrapped model ---
module = QwenLoraModule.load_from_checkpoint(
    ckpt_path,
    model_id=model_id,
    lr=config.get("lr",1e-4),
    weight_decay=config.get("weight_decay",0.01),
    lr_scheduler_type=config.get("lr_scheduler_type","linear"),
    warmup_steps=0,
    num_training_steps=1,          # not used in inference
    gradient_checkpointing=False,
    gc_use_reentrant=False,
    tf32=True,
    lora_cfg=config.get("lora",{})
)
module.eval().cuda()

# --- Inference collator: simplified for pure inference ---
class QwenPureInferenceCollator:
    def __init__(self, processor, pad_to_multiple_of=8):
        self.processor = processor
        self.pad_to_multiple_of = pad_to_multiple_of
        self.question = (
            "You are given a fluorescence microscopy image.\n\n"
            "Task: classify three attributes.\n"
            "Output format: exactly three words separated by single spaces, in this order: "
            "ORGANELLE PHASE INFECTION\n"
            "Allowed vocabularies:\n"
            "- ORGANELLE ∈ {ER, mitochondria, golgi, lysosome, nucleus, stress_granule}\n"
            "- PHASE ∈ {interphase, mitotic}\n"
            "- INFECTION ∈ {infected, uninfected}\n"
            "Rules: no punctuation, no explanations, no quotes, no newlines. "
            "If uncertain, guess the most likely label from the allowed set."
        )
    
    def __call__(self, batch):
        idx = batch["index"]
        imgs = batch["anchor"]  # [B, 3, 1, H, W]
        
        # 1) images -> [B, 3, H, W]
        if imgs.ndim == 5 and imgs.shape[2] == 1:
            imgs = imgs.squeeze(2)
        assert imgs.ndim == 4 and imgs.shape[1] == 3, f"Expected [B,3,H,W], got {tuple(imgs.shape)}"
        
        B = imgs.shape[0]
        
        # 2) build messages for each sample (no annotation lookup needed)
        texts = []
        images_for_qwen = []
        
        for i in range(B):
            # Qwen chat template
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": self.question},
                    {"type": "image"},
                ]}
            ]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            
            texts.append(text.strip())
            images_for_qwen.append([imgs[i]])
        
        # 3) process with tokenizer
        out = self.processor(
            text=texts,
            images=images_for_qwen,
            return_tensors="pt",
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of
        )
        
        # 4) Preserve metadata for later use
        out["metadata"] = {
            "fov_name": idx["fov_name"],
            "track_id": idx["track_id"],
            "t": idx["t"],
            "parent_id": idx["parent_id"]
        }
        
        return out

# Use the simplified collator
collator = QwenPureInferenceCollator(processor, pad_to_multiple_of=8)

predict_loader_qwen = CollatingLoader(predict_loader, collator)

# ---- Run predictions (limit to 10 images) ----
predictions = []
sample_info = []
count = 0
max_samples = 10

# Get the first few batches to collect 10 samples
for batch in predict_loader_qwen:
    if count >= max_samples:
        break
    
    # Extract metadata before moving to GPU
    metadata = batch.pop("metadata")  # Remove metadata from batch
    
    # Move batch to GPU
    for k,v in batch.items():
        batch[k] = v.cuda()
    
    # Generate predictions
    with torch.no_grad():
        gen_tokens = module.model.generate(
            **batch,
            max_new_tokens=10,        # enough to output 3 labels
            do_sample=False
        )
    
    # Decode predictions
    texts = processor.batch_decode(gen_tokens, skip_special_tokens=True)
    
    # Process each sample in the batch
    batch_size = len(texts)
    for i in range(batch_size):
        if count >= max_samples:
            break
            
        # Get metadata from the preserved metadata
        fov_name = str(metadata["fov_name"][i]) if i < len(metadata["fov_name"]) else "unknown"
        track_id = int(metadata["track_id"][i]) if i < len(metadata["track_id"]) else -1
        time_point = int(metadata["t"][i]) if i < len(metadata["t"]) else -1
        parent_id = int(metadata["parent_id"][i]) if i < len(metadata["parent_id"]) else -1
        
        sample_info.append({
            "sample_id": count,
            "fov_name": fov_name,
            "track_id": track_id,
            "time_point": time_point,
            "parent_id": parent_id,
            "prediction": texts[i]
        })
        
        predictions.append(texts[i])
        count += 1
    
    if count >= max_samples:
        break

# --- Save to CSV with metadata ---
df_predictions = pd.DataFrame(sample_info)
df_predictions.to_csv("predictions_10_samples.csv", index=False)

# --- Print results ---
print(f"\nGenerated predictions for {len(predictions)} samples:")
for info in sample_info:
    print(f"[Sample {info['sample_id']}] FOV: {info['fov_name']}, Track: {info['track_id']}, Time: {info['time_point']}")
    print(f"  Prediction: {info['prediction']}")
    print()

print(f"Saved predictions to predictions_10_samples.csv")
print(f"Total samples processed: {count}")

# %%
