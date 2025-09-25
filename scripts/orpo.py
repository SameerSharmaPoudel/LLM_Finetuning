import torch
from transformers import Trainer

import torch
import wandb

class ORPOLoss:
    def __init__(self, model, alpha=0.1, pad_token_id=0, disable_prompt_loss=True):
        # Initializes the ORPO loss function with model and loss configuration.
        """
        Args:
            model (torch.nn.Module): The pre-trained language model to be fine-tuned.
            alpha (float, optional): Weight factor for the log odds ratio. Default is 0.1.
            pad_token_id (int, optional): Token ID used for padding. Default is 0.
            disable_prompt_loss (bool, optional): Disables loss computation on prompt tokens.
        """
        self.model = model
        self.alpha = alpha
        self.pad_token_id = pad_token_id
        self.disable_prompt_loss = disable_prompt_loss

    def compute_logps(self, prompt_mask, response_inputs, response_attention_mask, logits):
        #Computes the per-token log probabilities of the response given model logits.
        """"
        Args:
            prompt_mask (torch.Tensor): Mask indicating prompt tokens (batch_size, seq_len).
            response_inputs (torch.Tensor): Input IDs for the response (batch_size, seq_len).
            response_attention_mask (torch.Tensor): Attention mask for the response (batch_size, seq_len).
            logits (torch.Tensor): Model logits (batch_size, seq_len, vocab_size).

        Returns:
            torch.Tensor: Log probabilities of response tokens (batch_size).
        """

        prompt_lengths = prompt_mask.sum(dim=1, keepdim=True)
        seq_len = response_inputs.size(1)
        mask = (torch.arange(seq_len-1, device=response_inputs.device) >= prompt_lengths).float()
        
        # Compute log probs for response tokens
        shift_logits = logits[:, :-1, :].contiguous().log_softmax(-1)
        shift_labels = response_inputs[:, 1:].contiguous()

        per_token_logps = torch.gather(shift_logits, 2, shift_labels.unsqueeze(2)).squeeze(2)
        
        return (per_token_logps * mask).sum(dim=1) / mask.sum(dim=1)
    
    def __call__(self, batch, accelerator):
        # Calculates the ORPO loss for a given batch of data.
        """"
        Args:
            batch (dict): Dictionary containing input data and labels:
                - 'pixel_values': Image data for vision-language models.
                - 'chosen_labels': Token IDs for the chosen response.
                - 'rejected_labels': Token IDs for the rejected response.
                - 'chosen_attention_mask': Attention mask for the chosen response.
                - 'rejected_attention_mask': Attention mask for the rejected response.
                - 'attention_mask': Combined attention mask.
            accelerator: Accelerator object for training.

        Returns:
            tuple: A tuple containing:
                - final_loss (torch.Tensor): The computed ORPO loss.
                - loss_dict (dict): Dictionary of loss metrics, including:
                    - 'orpo_loss': ORPO loss value.
                    - 'pos_prob_mean': Mean log probability of the chosen response.
                    - 'neg_prob_mean': Mean log probability of the rejected response.
                    - 'log_odds': Mean log odds ratio.
                    - 'pos_prob': Per-sample positive log probabilities.
                    - 'neg_prob': Per-sample negative log probabilities.
                    - 'nll_loss': Negative log likelihood loss value.
        """

        # Mask labels
        pos_labels = batch['chosen_labels'].clone()
        neg_labels = batch['rejected_labels'].clone()

        if self.disable_prompt_loss:
            mask = batch['attention_mask'] * batch['chosen_attention_mask']
            pos_labels = pos_labels *  mask.logical_not()
            pos_labels[pos_labels == 0] = self.pad_token_id

        # Replace padding with -100 for CE loss
        pos_labels[pos_labels == self.pad_token_id] = -100
        neg_labels[neg_labels == self.pad_token_id] = -100

        outputs_pos = self.model(
            images=batch['pixel_values'],
            input_ids=batch['chosen_labels'],
            attention_mask=batch['chosen_attention_mask'],
            labels=pos_labels, output_hidden_states=True)
        
        outputs_neg = self.model(
            images=batch['pixel_values'],
            input_ids=batch['rejected_labels'],
            attention_mask=batch['rejected_attention_mask'],
            labels=neg_labels, output_hidden_states=True)
        
        outputs_pos_logits = outputs_pos.logits  # Shape: (batch_size, effective_seq_len, vocab_size)
        max_len_pos = batch['chosen_attention_mask'].size(1)
        # Pad logits to match max sequence length
        outputs_pos_padded_logits = torch.zeros((outputs_pos_logits.size(0), max_len_pos, outputs_pos_logits.size(2)), device=outputs_pos_logits.device)
        outputs_pos_padded_logits[:, :outputs_pos_logits.size(1), :] = outputs_pos.logits

        outputs_neg_logits = outputs_neg.logits  # Shape: (batch_size, effective_seq_len, vocab_size)
        max_len_neg = batch['rejected_attention_mask'].size(1)
        # Pad logits to match max sequence length
        outputs_neg_padded_logits = torch.zeros((outputs_neg_logits.size(0), max_len_neg, outputs_neg_logits.size(2)), device=outputs_neg_logits.device)
        outputs_neg_padded_logits[:, :outputs_neg_logits.size(1), :] = outputs_neg.logits

        # Calculate log probabilities 
        pos_prob = self.compute_logps(
            batch['attention_mask'],
            batch['chosen_labels'],
            batch['chosen_attention_mask'],
            outputs_pos_padded_logits)
        
        neg_prob = self.compute_logps(
            batch['attention_mask'],
            batch['rejected_labels'],
            batch['rejected_attention_mask'],
            outputs_neg_padded_logits)

        # ORPO loss components
        # the following seems to be numerically unstable, so a simpler version is used
        # log_odds = (pos_prob - neg_prob) -  (
        #     torch.log1p(-torch.exp(pos_prob)) - 
        #     torch.log1p(-torch.exp(neg_prob)))

        log_odds = pos_prob - neg_prob
        sig_ratio = torch.nn.functional.sigmoid(log_odds)
        ratio = torch.log(sig_ratio)

        # Combine losses
        final_loss = torch.mean(outputs_pos.loss - self.alpha * ratio)

        # Optional logging
        loss_dict = {
                    'orpo_loss': final_loss.item(),
                    'pos_prob_mean': pos_prob.mean().item(),
                    'neg_prob_mean': neg_prob.mean().item(),
                    'log_odds': log_odds.mean().item(),
                    'pos_prob': pos_prob,          # Shape: [batch_size]
                    'neg_prob': neg_prob,          # Shape: [batch_size]
                    'nll_loss': torch.mean(outputs_pos.loss).item()
                    }
        
        return final_loss, loss_dict