import torch
import os
import argparse
from pathlib import Path
import warnings
import pandas as pd
from transformers import BitsAndBytesConfig
import LLaVA
from accelerate import Accelerator
import wandb
from LLaVA.llava.utils import disable_torch_init
from prepare_model_tokenizer import prepare_model_for_peft, prepare_tokenizer
from datasets import load_dataset
from dataloader import FlexibleLLaVADataset, create_dataloader
from train_and_validate import train, validate
from orpo import ORPOLoss
from utils import generate_examples

warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")

def get_args_parser():

    parser = argparse.ArgumentParser(description='LLaVA Alignment', add_help=False) 

    # Dataset/data loader
    parser.add_argument('--dataset_path', default='openbmb/RLAIF-V-Dataset', type=str, help='path to dataset') 
    parser.add_argument('--total_samples', default=30000, type=int) # Use None for full dataset
    parser.add_argument('--val_ratio', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--max_length', default=512, type=int, help='maximum length for tokenizer' )

    # Model 
    parser.add_argument('--model_path', default='liuhaotian/llava-v1.6-mistral-7b', type=str, help='path to model') 
    parser.add_argument('--model_name', default='llava-v1.6-mistral-7b', type=str, help='path to model') 
    parser.add_argument('--checkpoint_path', default='', type=str, help='path to checkpoint') 

    # Training/finetuning
    parser.add_argument('--lora_rank', default=16, type=int)
    parser.add_argument('--beta', default=0.2, type=float)
    parser.add_argument('--lambda_', default=0.15, type=float)
    parser.add_argument('--lora_dropout', default=0.05, type=float)
    parser.add_argument("--target_modules", nargs='+', type=str, default=["q_proj", "k_proj", "v_proj", "mm_proj ", "gate_proj", "up_proj", "lm_head"], help="List of target modules for peft") # ["lm_head", "q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
    parser.add_argument('--lr', default=2.5e-6, type=float)
    parser.add_argument('--epochs', default=1, type=int)  
    
    # Miscelleneous
    parser.add_argument('--output_dir', default='./result', help='path where to save, empty for no saving')  #output_dir assigned

    return parser

def main(args):

    wandb.login(key="e05e5183a6925ffb1c2129b0bc4ec1a661e183cb") #input your wandb key here

    if torch.cuda.is_available():
        print("PyTorch is connected to GPU.")
        print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        device = torch.device('cuda')
    else:
        print("PyTorch is not connected to GPU. Fine-tuning will be slow.")
        return
    # print(f'Running LLaVA Alignment on {device}')

    try:
        from LLaVA.llava.model.builder import load_pretrained_model
        print("LLaVA is correctly accessible!")
    except ImportError:
        print("LLaVA is NOT in your Python path.")

    # # Disable initial torch operations for faster loading
    disable_torch_init()    

    print('Loading model...')
    # Load model with explicit config
    quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path, #args.model_path
        model_base=None,
        model_name=args.model_name,
        quantization_config=quant_config,  # Pass config here
        device_map="auto",
        device="cuda",
        use_flash_attn=False )
    
    # print('Model:')
    # print(model)
    # print('....................................')
    # print('Tokenizer:')
    # print(tokenizer)
    # print('....................................')
    # print('Image Processor:')
    # print(image_processor)
    # print('....................................')

    print(f"Processor type: {type(image_processor)}")
    if hasattr(image_processor, "tokenizer"):
        print("Processor has tokenizer:", image_processor.tokenizer)
    else:
        print("Processor does NOT have tokenizer")
        
    tokenizer, model = prepare_tokenizer(args, tokenizer, model)
    model = prepare_model_for_peft(args, model)
    model.lm_head.requires_grad = True

    dataset = load_dataset(args.dataset_path, split="train")

    # Take subset if needed, if None take the whole samples in the dataset
    if args.total_samples:
        dataset = dataset.select(range(args.total_samples))

    print(f"Final dataset size: {len(dataset)}")

    split_dataset = dataset.train_test_split(test_size=args.val_ratio, seed=42)

    # Create datasets
    train_dataset = FlexibleLLaVADataset(image_processor, tokenizer, split_dataset, split="train", max_length = args.max_length)
    val_dataset = FlexibleLLaVADataset(image_processor, tokenizer, split_dataset, split="test", max_length = args.max_length)

    # Create dataloaders
    train_loader = create_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=args.batch_size)

    # Verify dataset statistics
    print("\nDataset Statistics:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Batch counts - Train: {len(train_loader)}, Val: {len(val_loader)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10000, T_mult=1, eta_min=0.0, last_epoch=-1, verbose='deprecated')
    
    accelerator = Accelerator(
                mixed_precision="fp16", #mixed_precision="bf16",
                gradient_accumulation_steps=8,
                device_placement=True,
                log_with="wandb" )

    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler)
    
    # print('model config:', model.config)

    # Initialize tracking
    accelerator.init_trackers(project_name="llava-orpo-training-RLAIF-V_LoRA_rank_16")

    # Initialize ORPO loss
    loss_fn = ORPOLoss(
        model=model,
        alpha=args.lambda_,
        pad_token_id=tokenizer.pad_token_id,
        disable_prompt_loss=True)

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    for epoch in range(args.epochs):

        train_loss = train(epoch, train_loader, model, optimizer, lr_scheduler, accelerator, loss_fn)
        val_loss, val_acc = validate(epoch, val_loader, model, accelerator, loss_fn)

        # Log metrics and examples
        accelerator.log({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc })  
       
        # Generate and log examples
        examples = generate_examples(args, model, image_processor, tokenizer, val_dataset, num_examples=4, device="cuda")
        
        accelerator.log({f"epoch_{epoch+1}_examples": wandb.Table(dataframe=pd.DataFrame(examples))})

        # Save checkpoint
        checkpoint_dir = os.path.join(args.output_dir, f"epoch_{epoch+1}_checkpoint")
        accelerator.save_state(checkpoint_dir)


    # Final cleanup
    accelerator.end_training()
    # Unwrap the model from accelerator
    unwrapped_model = accelerator.unwrap_model(model)
    # Merge LoRA adapters before saving for inference
    merged_model = unwrapped_model.merge_and_unload()
    # Save the final merged model
    merged_model.save_pretrained(args.output_dir)
    # Save processor
    image_processor.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('LLaVA Alignment', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

