from unsloth import FastLanguageModel, FastModel
import torch
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from unsloth import to_sharegpt
from unsloth import standardize_sharegpt
from unsloth import is_bfloat16_supported
from unsloth import apply_chat_template
from transformers import TrainingArguments
from transformers import TextStreamer
from torchao.quantization import quantize_
from torchao.quantization.qat import QATConfig

max_seq_length = 4096

model, processor = FastLanguageModel.from_pretrained(
    model_name = "/root/autodl-tmp/qwen35-4b",
    load_in_4bit = False,
    load_in_8bit = False,
    full_finetuning = False,
    use_gradient_checkpointing = True,
    local_files_only=True,
)
tokenizer = processor.tokenizer

model = FastLanguageModel.get_peft_model(
   model,
   finetune_vision_layers     = True, # False if not finetuning vision layers
   finetune_language_layers   = True, # False if not finetuning language layers
   finetune_attention_modules = True, # False if not finetuning attention layers
   finetune_mlp_modules       = True, # False if not finetuning MLP layers
   r = 32,
   target_modules = ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj",],
   lora_alpha = 32,
   lora_dropout = 0,
   bias = "none",
   use_gradient_checkpointing = True,
   random_state = 3407,
   use_rslora = False,
   loftq_config = None,
)

dataset = load_dataset(path="imagefolder", data_dir="./dataset-img-train")

instruction = "解读这个医学报告，一句话指出其中异常指标"

def convert_to_conversation(sample):
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : instruction},
            {"type" : "image", "image" : sample["image"]} ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample["additional_feature"]} ]
        },
    ]
    return { "messages" : conversation }
pass

converted_dataset = [convert_to_conversation(sample) for sample in dataset["train"]]

FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 8,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 16,
        warmup_steps = 5,
        #max_steps = 6,
        num_train_epochs = 3, # For longer training runs!
        learning_rate = 5e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 2,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)

trainer_stats = trainer.train()

print("saving...")
model.save_pretrained_merged("/root/autodl-tmp/qwen35-4b-med-vl", tokenizer)
#tokenizer.save_pretrained("/root/autodl-tmp/qwen35-08b-finetuned-ffmpeglog")
print("saved")
