import torch
from tqdm import tqdm 

def train(epoch, train_loader, model, optimizer, lr_scheduler, accelerator, loss_fn):
    """ 
    Trains the model for one epoch using the given data loader and optimizer.

    Args:
        epoch (int): Current epoch number.
        train_loader (torch.utils.data.DataLoader): Data loader for training samples.
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        lr_scheduler (torch.optim.lr_scheduler): Learning rate scheduler for adjusting the learning rate.
        accelerator: An instance of the Hugging Face `Accelerator` for mixed precision and distributed training.
        loss_fn (callable): Loss function that computes the training loss.

    Returns:
        float: The average training loss for the epoch.
    """
    
    # Training Phase
    model.train()
    epoch_train_loss = 0

    train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
    for batch_idx, batch in enumerate(train_progress):

        with accelerator.accumulate(model):
            # Forward passes
            with accelerator.autocast(): 
                loss, loss_dict = loss_fn(batch, accelerator)
                
            # Backpropagation
            accelerator.backward(loss)     

            # Memory management
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                torch.cuda.empty_cache()

            lr_scheduler.step()
            
            # Update metrics
            epoch_train_loss += loss.item()
            train_progress.set_postfix({"loss": loss.item(), "batch": batch_idx})

            loss_dict['epoch'] = epoch + 1
            loss_dict['batch_idx'] = batch_idx + 1
            loss_dict['global_step'] =  epoch * len(train_loader) + batch_idx

            del loss_dict["pos_prob"]
            del loss_dict["neg_prob"]

            accelerator.log(loss_dict)

    train_loss = epoch_train_loss / len(train_loader)

    return train_loss


def validate(epoch, val_loader, model, accelerator, loss_fn):
    """"
    Validates the model on the validation dataset and computes validation loss and accuracy.

    Args:
        epoch (int): Current epoch number.
        val_loader (torch.utils.data.DataLoader): Data loader for validation samples.
        model (torch.nn.Module): The model to be evaluated.
        accelerator: An instance of the Hugging Face `Accelerator` for mixed precision and distributed training.
        loss_fn (callable): Loss function that computes the validation loss.

    Returns:
        tuple:
            - val_loss (float): The average validation loss for the epoch.
            - val_acc (float): The accuracy of model predictions on the validation set.
    """

    # Validation Phase
    model.eval()
    epoch_val_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation")
        for batch_idx, batch in enumerate(val_progress):
            # Forward passes
            with accelerator.autocast():
                # Calculate metrics
                loss, loss_dict = loss_fn(batch, accelerator)
                epoch_val_loss += loss.item()

            # Calculate preference accuracy
            # Extract per-example probabilities
            pos_prob = loss_dict['pos_prob']  # Shape: [batch_size]
            neg_prob = loss_dict['neg_prob']  # Shape: [batch_size]

            # Compute accuracy: chosen prob > rejected prob
            correct += (pos_prob > neg_prob).sum().item()
            total += pos_prob.size(0)  # Batch size

            val_progress.set_postfix({"val_loss": loss.item(), "batch": batch_idx})

    # Calculate epoch metrics
    val_loss = epoch_val_loss / len(val_loader)
    val_acc = correct / total

    return val_loss, val_acc
