import matplotlib.pyplot as plt
import numpy as np
import random
from peft import PeftModel
import wandb
import pandas as pd
from generator import VisionGroundedGenerator

def generate_examples(args, model, image_processor, tokenizer, dataset, num_examples, device):
    """
    Generate and log example inferences from the model.

    This function generates a specified number of inference examples using the given model, tokenizer,
    and dataset. It performs autoregressive text generation based on the input image and text samples, 
    and logs the generated responses along with corresponding ground-truth references.

    Args:
        args (argparse.Namespace): Command-line arguments or configuration object containing parameters 
                                   such as max_length for text generation.
        model (torch.nn.Module): Pretrained model used for generating text responses.
        image_processor (object): Processor for handling image preprocessing (unused in this function).
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for encoding and decoding text inputs.
        dataset (torch.utils.data.Dataset): Dataset containing input images, text, and reference responses.
        num_examples (int): Number of inference examples to generate.
        device (torch.device): Device to perform inference on (e.g., 'cuda' or 'cpu').

    Returns:
        list: A list of dictionaries where each dictionary contains:
            - "image" (wandb.Image): The input image wrapped as a W&B image object.
            - "prompt" (str): The decoded input prompt.
            - "response" (str): The generated response from the model.
            - "chosen_ref" (str): The reference response labeled as the correct response.
            - "rejected_ref" (str): The reference response labeled as the incorrect response.

    """
    
    model.eval()
    examples = []

    for idx in range(num_examples):
        
        idx = random.randint(0, len(dataset) - 1)
        sample = dataset[idx]

        images = sample["pixel_values"].unsqueeze(0)
        input_ids = sample["input_ids"].unsqueeze(0)
        attention_mask = sample["attention_mask"].unsqueeze(0)

        generator = VisionGroundedGenerator(model, tokenizer, max_length=args.max_length)
        generated_text = generator.generate(
            images, 
            input_ids,
            attention_mask
        )

        print("Decoded response:", generated_text)
        
        # Log results
        examples.append({
            "image": wandb.Image(sample["pixel_values"]),
            "prompt": tokenizer.decode(sample["input_ids"], skip_special_tokens=True),
            "response": str(generated_text),
            "chosen_ref": tokenizer.decode(sample["chosen_labels"], skip_special_tokens=True),
            "rejected_ref": tokenizer.decode(sample["rejected_labels"], skip_special_tokens=True)
        })

    return examples