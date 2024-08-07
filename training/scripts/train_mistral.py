import argparse
import os
import logging
import torch
from datasets import load_from_disk
import transformers
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from peft import (
    LoraConfig,
    get_peft_model,
)
import os
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--micro_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--base_model", type=str)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--cutoff_len", type=str, default=512)
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--group_by_length", type=str, default=False)
    parser.add_argument("--use_special_token", type=bool, default=False)



    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)


    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--eval_steps", type=int, default=40)
    parser.add_argument("--save_steps", type=int, default=200)


    # Data, model, and output directories
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_VAL"])


    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # load datasets
    train_dataset = load_from_disk(args.training_dir)
    test_dataset = load_from_disk(args.test_dir)

    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(test_dataset)}")

    bitsandbytes= BitsAndBytesConfig(
        load_in_4bit=True,
        # bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        attn_implementation="flash_attention_2", # set this to True if your GPU supports it
        quantization_config=bitsandbytes,
        device_map=args.device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if args.use_special_token:
        tokenizer.add_tokens(["\'"], special_tokens=True)

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        # target_modules=lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    tokenizer.padding_side = "left"  # Allow batched inference
    # set pad_token_id equal to the eos_token_id if not set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info(model.print_trainable_parameters())

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            # eval_accumuluation_steps=10,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.epochs, #num_epochs,
            learning_rate=float(args.learning_rate),
            bf16=True, # GPU needs to support this
            logging_steps=10,
            logging_dir=f"{args.output_data_dir}/logs",
            optim="adamw_torch",
            evaluation_strategy="steps",
            # report_to="mlflow",
            save_strategy="steps",
            eval_steps=20,
            save_steps=200,
            output_dir=args.model_dir,
            save_total_limit=3,
            load_best_model_at_end=True,
            group_by_length=args.group_by_length,
            # report_to="wandb" if use_wandb else None,
            # run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    trainer.train()

    model.save_pretrained(args.model_dir)