import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

# --- Paths ---
base_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"      # same as training base
lora_ckpt_or_dir = "fine-tuned-model/last"         # your saved adapter dir (ModelCheckpoint dir)
# If you saved adapters with model.save_pretrained(...), use that directory instead.

# --- Load processor & model (+ LoRA) ---
processor = AutoProcessor.from_pretrained(base_model_id)

base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",
)

# Attach LoRA adapters
model = PeftModel.from_pretrained(base, lora_ckpt_or_dir)
model.eval()

# (Optional) merge adapters for faster inference:
# model = model.merge_and_unload()   # returns a plain Qwen2_5_VLForConditionalGeneration

# --- Prepare your image & prompt ---
img = Image.open("sample.jpg").convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What do you see here?"},
            {"type": "image"}  # placeholder for the image
        ]
    }
]

# Build model inputs (same template as training)
inputs = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
# Note: for VL models, pass images separately via processor()
batch = processor(text=[inputs], images=[[img]], return_tensors="pt", padding=True)

# Move tensors to model device and dtype
device = model.device
for k in batch:
    if isinstance(batch[k], torch.Tensor):
        batch[k] = batch[k].to(device)
# Generation params (tweak as you like)
gen_kwargs = dict(
    max_new_tokens=64,
    temperature=0.2,
    top_p=0.9,
    do_sample=True,
)

with torch.no_grad():
    output_ids = model.generate(**batch, **gen_kwargs)

# Decode only the newly generated portion if you prefer; simplest is full decode:
text = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("\n=== MODEL OUTPUT ===")
print(text)

# -----------------------
# 2) Compute loss on a single (image, caption) pair
# -----------------------
# This mirrors the training collator (masking everything except the assistant's answer)
# so we can get a numerical loss for quick sanity checks.

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
lora_ckpt_or_dir = "fine-tuned-model/last"

processor = AutoProcessor.from_pretrained(base_model_id)
base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_model_id, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
)
model = PeftModel.from_pretrained(base, lora_ckpt_or_dir).eval()

# Your eval pair
img = Image.open("sample.jpg").convert("RGB")
answer = "A white king and black rook on a chessboard."   # reference caption/answer

messages = [
    {"role": "user", "content": [{"type": "text", "text": "What do you see here?"}, {"type": "image"}]},
    {"role": "assistant", "content": [{"type": "text", "text": answer}]}
]

# Prepare tokens (no generation prompt; we compute loss)
text = processor.apply_chat_template(messages, add_generation_prompt=False)
batch = processor(text=[text], images=[[img]], return_tensors="pt", padding=True)

# Build labels that mask everything except the assistant answer
labels = batch["input_ids"].clone()
target_ids = processor.tokenizer(answer, return_tensors="pt")["input_ids"][0]

def find_subseq(seq, subseq):
    for i in range(len(seq) - len(subseq) + 1):
        if torch.equal(seq[i:i+len(subseq)], subseq):
            return i
    return None

start = find_subseq(batch["input_ids"][0], target_ids)
if start is not None:
    end = start + len(target_ids)
    eos = processor.tokenizer.eos_token_id
    if end < batch["input_ids"].shape[1] and batch["input_ids"][0, end].item() == eos:
        end += 1
    labels[:, :start] = -100
    labels[:, end:]  = -100
else:
    labels[:] = -100

batch["labels"] = labels

# Move to device
device = model.device
for k in batch:
    if isinstance(batch[k], torch.Tensor):
        batch[k] = batch[k].to(device)

with torch.no_grad():
    out = model(**batch)

print(f"Loss on this sample: {out.loss.item():.4f}")

Here’s a simple, reliable way to run **inference on a single image** with your fine-tuned LoRA model. I’ll show two variants:

1. **Generate a caption/answer** (typical eval-by-eye)
2. **Compute loss** on a (image, caption) pair (numerical check)

Both assume you trained with the same chat template and LoRA setup you shared.

---

# 1) Generate text for one image

```python
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

# --- Paths ---
base_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"      # same as training base
lora_ckpt_or_dir = "fine-tuned-model/last"         # your saved adapter dir (ModelCheckpoint dir)
# If you saved adapters with model.save_pretrained(...), use that directory instead.

# --- Load processor & model (+ LoRA) ---
processor = AutoProcessor.from_pretrained(base_model_id)

base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",
)

# Attach LoRA adapters
model = PeftModel.from_pretrained(base, lora_ckpt_or_dir)
model.eval()

# (Optional) merge adapters for faster inference:
# model = model.merge_and_unload()   # returns a plain Qwen2_5_VLForConditionalGeneration

# --- Prepare your image & prompt ---
img = Image.open("sample.jpg").convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What do you see here?"},
            {"type": "image"}  # placeholder for the image
        ]
    }
]

# Build model inputs (same template as training)
inputs = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
# Note: for VL models, pass images separately via processor()
batch = processor(text=[inputs], images=[[img]], return_tensors="pt", padding=True)

# Move tensors to model device and dtype
device = model.device
for k in batch:
    if isinstance(batch[k], torch.Tensor):
        batch[k] = batch[k].to(device)
# Generation params (tweak as you like)
gen_kwargs = dict(
    max_new_tokens=64,
    temperature=0.2,
    top_p=0.9,
    do_sample=True,
)

with torch.no_grad():
    output_ids = model.generate(**batch, **gen_kwargs)

# Decode only the newly generated portion if you prefer; simplest is full decode:
text = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("\n=== MODEL OUTPUT ===")
print(text)
```

**Notes**

* If you saved only Lightning checkpoints (`.ckpt`) and not PEFT adapters: load your checkpoint into the same code that created the model and then call `model.save_pretrained("fine-tuned-model/adapters")`. From then on, the snippet above works with that directory.
* If you used a `run_name`/`output_dir`, the latest checkpoint path often looks like `fine-tuned-model/last/` (or the best checkpoint directory saved by `ModelCheckpoint`). Use that as `lora_ckpt_or_dir`.

---

# 2) Compute loss on a single (image, caption) pair

This mirrors your training collator (masking everything except the assistant’s answer) so you can get a **numerical loss** for quick sanity checks.

```python
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
lora_ckpt_or_dir = "fine-tuned-model/last"

processor = AutoProcessor.from_pretrained(base_model_id)
base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_model_id, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
)
model = PeftModel.from_pretrained(base, lora_ckpt_or_dir).eval()

# Your eval pair
img = Image.open("sample.jpg").convert("RGB")
answer = "A white king and black rook on a chessboard."   # reference caption/answer

messages = [
    {"role": "user", "content": [{"type": "text", "text": "What do you see here?"}, {"type": "image"}]},
    {"role": "assistant", "content": [{"type": "text", "text": answer}]}
]

# Prepare tokens (no generation prompt; we compute loss)
text = processor.apply_chat_template(messages, add_generation_prompt=False)
batch = processor(text=[text], images=[[img]], return_tensors="pt", padding=True)

# Build labels that mask everything except the assistant answer
labels = batch["input_ids"].clone()
target_ids = processor.tokenizer(answer, return_tensors="pt")["input_ids"][0]

def find_subseq(seq, subseq):
    for i in range(len(seq) - len(subseq) + 1):
        if torch.equal(seq[i:i+len(subseq)], subseq):
            return i
    return None

start = find_subseq(batch["input_ids"][0], target_ids)
if start is not None:
    end = start + len(target_ids)
    eos = processor.tokenizer.eos_token_id
    if end < batch["input_ids"].shape[1] and batch["input_ids"][0, end].item() == eos:
        end += 1
    labels[:, :start] = -100
    labels[:, end:]  = -100
else:
    labels[:] = -100

batch["labels"] = labels

# Move to device
device = model.device
for k in batch:
    if isinstance(batch[k], torch.Tensor):
        batch[k] = batch[k].to(device)

with torch.no_grad():
    out = model(**batch)

print(f"Loss on this sample: {out.loss.item():.4f}")
```

# -----------------------
# Tips and common gotchas:
# -----------------------
# Adapters vs. full model**: With LoRA, you typically load the base model first, then load the adapters using `PeftModel.from_pretrained(...)`. If you want a single consolidated model for export or serving, you should call `merge_and_unload()` after loading the adapters, and then save the model with `save_pretrained(...)`.
# dtype/precision**: If you trained in bf16, make sure to load with `torch_dtype=torch.bfloat16` and keep autocast off. Generation works well in bf16 on Ampere or newer GPUs.
# Images list-of-list**: For Qwen VL, always pass `images=[[img]]` even for a single image. This matches the chat turn structure, where one image is inside a single user message.
# Template consistency**: Always use the same chat template you used during training (with `apply_chat_template`) to avoid any tokenization drift during evaluation.
# Determinism**: For consistent, repeatable results, set `do_sample=False` (for greedy or beam search) or fix the random seed with `torch.manual_seed(...)` before calling `generate`.

# If you let me know where your best checkpoint is saved (for example, `fine-tuned-model/epoch=2-step=1200.ckpt` or a Hugging Face Hub repo), I can provide a ready-to-use code snippet for that exact path.
