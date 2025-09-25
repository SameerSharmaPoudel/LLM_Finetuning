import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from PIL import Image

class FlexibleLLaVADataset(Dataset):

    """
    A flexible dataset class for vision-language models, designed to handle image-text pairs 
    with both chosen and rejected responses.

    Args:
        image_processor (Callable): A function or processor to preprocess images.
        tokenizer (PreTrainedTokenizer): The tokenizer to convert text into input IDs.
        dataset (dict): A dictionary containing split-specific datasets.
        split (str): The dataset split to load ('train', 'validation', or 'test').
        max_length (int, optional): Maximum length for tokenized input sequences. Defaults to 512.
        image_size (int, optional): Size of images after processing. Defaults to 336.

    Raises:
        ValueError: If the tokenizer's chat template is not initialized.

    Returns:
        dict: A dictionary containing preprocessed inputs:
            - "pixel_values" (torch.Tensor): Processed image tensor.
            - "input_ids" (torch.Tensor): Token IDs for the input prompt.
            - "attention_mask" (torch.Tensor): Attention mask for the input prompt.
            - "chosen_labels" (torch.Tensor): Token IDs for the chosen response.
            - "rejected_labels" (torch.Tensor): Token IDs for the rejected response.
            - "chosen_attention_mask" (torch.Tensor): Attention mask for the chosen response.
            - "rejected_attention_mask" (torch.Tensor): Attention mask for the rejected response.
    """

    def __init__(
        self,
        image_processor,
        tokenizer,
        dataset,  # Pass dataset directly
        split: str,
        max_length: int = 512,
        image_size: int = 336,
    ):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_size = image_size
        self.dataset = dataset[split]
        self.image_token_id = tokenizer.convert_tokens_to_ids("<image>")

        # Verify chat template setup
        if self.tokenizer.chat_template is None:
            raise ValueError("Chat template not initialized!")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # Process image
        try:
            image = sample["image"]
            if not isinstance(image, Image.Image):
                image = Image.open(image).convert("RGB")
        except Exception:
            return self.__getitem__((idx + 1) % len(self))  # Retry with next sample
        
        pixel_values = self.image_processor(image, return_tensors="pt").pixel_values

        # Format input prompt template
        input_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": f"<image>\n{sample['question']}"}],
            add_generation_prompt=True, tokenize=False )
        
        # For chosen response
        chosen_text = self.tokenizer.apply_chat_template([
            {"role": "user", "content": f"<image>\n{sample['question']}"},
            {"role": "assistant", "content": sample["chosen"]}], tokenize=False)
        
        # For rejected response  
        rejected_text = self.tokenizer.apply_chat_template([
            {"role": "user", "content": f"<image>\n{sample['question']}"},
            {"role": "assistant", "content": sample["rejected"]}], tokenize=False)
        
        # Tokenize 
        model_inputs  = self.tokenizer(
            input_prompt,  
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt" )
        

        # Verify <image> token exists
        assert self.image_token_id in model_inputs.input_ids, "Missing <image> token!"

        # Tokenize responses separately
        chosen = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt" )
        

        rejected = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt" )
        

        return {
            "pixel_values": pixel_values.squeeze(),
            "input_ids": model_inputs.input_ids.squeeze(),
            "attention_mask": model_inputs.attention_mask.squeeze(),
            "chosen_labels": chosen.input_ids.squeeze(0),
            "rejected_labels": rejected.input_ids.squeeze(0),
            "chosen_attention_mask": chosen.attention_mask.squeeze(0),
            "rejected_attention_mask": rejected.attention_mask.squeeze(0)
        }


def collate_fn(batch):
    
    """
    Custom collate function to batch samples from the FlexibleLLaVADataset.

    Args:
        batch (list of dict): A list of sample dictionaries from the dataset.

    Returns:
        dict: A dictionary containing batched tensors:
            - "pixel_values" (torch.Tensor): Stacked image tensors.
            - "input_ids" (torch.Tensor): Padded input IDs.
            - "attention_mask" (torch.Tensor): Padded attention masks.
            - "chosen_labels" (torch.Tensor): Padded chosen response token IDs.
            - "rejected_labels" (torch.Tensor): Padded rejected response token IDs.
            - "chosen_attention_mask" (torch.Tensor): Padded chosen response attention masks.
            - "rejected_attention_mask" (torch.Tensor): Padded rejected response attention masks.
    """

    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "input_ids": pad_sequence([x["input_ids"] for x in batch], batch_first=True),
        "attention_mask": pad_sequence([x["attention_mask"] for x in batch], batch_first=True),
        "chosen_labels": pad_sequence([x["chosen_labels"] for x in batch], batch_first=True),
        "rejected_labels": pad_sequence([x["rejected_labels"] for x in batch], batch_first=True),
        "chosen_attention_mask": pad_sequence([x["chosen_attention_mask"] for x in batch], batch_first=True),
        "rejected_attention_mask": pad_sequence([x["rejected_attention_mask"] for x in batch], batch_first=True)
    }


def create_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = False):
    
    """
    Create a DataLoader for efficient data loading and batching.

    Args:
        dataset (Dataset): The dataset object to load from.
        batch_size (int): Number of samples per batch.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to False.

    Returns:
        DataLoader: A DataLoader configured with the given dataset and batch size.
    """

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, pin_memory=True,
                      num_workers=2,  
                      persistent_workers=True,
                      prefetch_factor=2
                     )