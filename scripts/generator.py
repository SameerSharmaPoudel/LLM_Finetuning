import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

class VisionGroundedGenerator:
    
    """
    VisionGroundedGenerator: A Vision-Language Model Response Generator with CLIP-based Verification

    This class generates text responses conditioned on image and text inputs using an autoregressive 
    language model. It incorporates CLIP for vision-grounded verification, reducing hallucinations.

    Key Features:
    - Autoregressive text generation with temperature scaling, top-k, and top-p filtering.
    - Safe token sampling with numerical stability and error handling.
    - CLIP-based candidate selection to improve image-text alignment.
    - Handles CUDA memory cleanup and outlier cases to prevent errors.

    Classes:
        - VisionGroundedGenerator: Generates image-conditioned text responses with grounding verification.

    Methods:
        - __init__: Initializes the generator with model, tokenizer, and CLIP verification.
        - _safe_generate: Handles autoregressive generation with error handling.
        - _top_k_top_p_filtering: Applies top-k and top-p sampling to logits.
        - generate: Generates multiple candidates and selects the best using CLIP score.
        - _clip_score: Computes CLIP-based alignment score for image-text pairs.

    """

    def __init__(self, model, tokenizer, max_length=50, clip_model_name="openai/clip-vit-base-patch32", device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.vocab_size = len(tokenizer)
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        print('max length:', self.max_length)
        
        # CLIP for grounding verification
        self.clip = CLIPModel.from_pretrained(clip_model_name).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        # Generation hyperparameters
        self.base_temp = 0.7
        self.min_temp = 0.3
        self.top_k = 30
        self.top_p = 0.85

    def _safe_generate(self, images, input_ids, attention_mask=None):
        """Core generation with comprehensive error handling"""
        try:
            # Clean CUDA state and ensure device alignment
            torch.cuda.empty_cache()
            images = images.to(self.device)
            input_ids = input_ids.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            for step in range(self.max_length):
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                    # Forward pass with numerical checks
                        outputs = self.model(
                            images=images,
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                        
                    logits = outputs.logits[:, -1, :]
                    
                    # Validate logits
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        raise ValueError("Invalid logits detected")
                    
                    # Temperature sampling with stability
                    temp = max(self.min_temp, self.base_temp * (1 - step/self.max_length))
                    scaled_logits = logits / temp
                    
                    # Top-k/p filtering
                    filtered_logits = self._top_k_top_p_filtering(scaled_logits)
                    probs = F.softmax(filtered_logits, dim=-1)
                    
                    # Numerical stability
                    probs = torch.nan_to_num(probs, nan=1e-5, posinf=1e5, neginf=1e-5)
                    probs = probs / probs.sum()
                    
                    # Safe sampling with bounds checking
                    next_token = torch.multinomial(
                        probs,
                        num_samples=1,
                        replacement=True  # Safer sampling
                    )
                    
                    # Validate token
                    if next_token >= self.vocab_size:
                        next_token = torch.tensor([[self.pad_token_id]], device=self.device)
                    
                    # Update sequences
                    input_ids = torch.cat([input_ids, next_token], dim=-1)
                    if attention_mask is not None:
                        attention_mask = torch.cat(
                            [attention_mask, torch.ones_like(next_token)],
                            dim=-1
                        )
                    
                    # Early stopping
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break

        except Exception as e:
            print(f"Generation error: {str(e)}")
            # Return partial output with safety padding
            pad_tokens = torch.tensor(
                [[self.pad_token_id] * (self.max_length - input_ids.size(1))],
                device=self.device
            )
            input_ids = torch.cat([input_ids, pad_tokens], dim=-1)
        
        return input_ids

    def _top_k_top_p_filtering(self, logits):
        """Modified filtering with numerical stability"""
        logits = logits.clone()
        
        # Top-k filtering
        if self.top_k > 0:
            top_k = min(self.top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float('Inf')
        
        # Top-p filtering
        if 0 < self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            
            # Apply masking
            for i in range(logits.size(0)):
                logits[i, sorted_indices[i, sorted_indices_to_remove[i]]] = -float('Inf')
        
        return logits

    def generate(self, images, input_ids, attention_mask, num_candidates=3):
        """Main generation API with candidate selection"""

        # Generate candidates
        candidates = []
        for _ in range(num_candidates):
            try:
                output_ids = self._safe_generate(
                    images,
                    input_ids.to(self.device),
                    attention_mask.to(self.device) if attention_mask is not None else None
                )
                text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                candidates.append((text, self._clip_score(images, text)))
            except Exception as e:
                print(f"Candidate generation failed: {e}")
                candidates.append(("[ERROR]", -float('inf')))

        print('Candidates:', candidates)
        
        # Return best candidate by CLIP score
        return max(candidates, key=lambda x: x[1])[0]

    def _clip_score(self, image, text):
        """Compute image-text alignment score"""
        try:
            inputs = self.clip_processor(
                text=[text],
                images=image,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.clip(**inputs)
            return float(outputs.logits_per_image[0, 0])
        except Exception:
            return -float('inf')  # Return worst score if CLIP fails

