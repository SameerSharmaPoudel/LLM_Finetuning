from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import torch 

def prepare_model_for_peft(args, model):

    """
    Prepare a model for Parameter-Efficient Fine-Tuning (PEFT) using QLoRA.

    This function configures a model for PEFT by applying Quantized Low-Rank Adaptation (QLoRA).
    It sets up LoRA configuration, prepares the model for low-bit training, 
    and prints the trainable parameters.

    Args:
        args (Namespace): An object containing the following attributes:
            - lora_rank (int): Rank of the LoRA adaptation matrices.
            - target_modules (list of str): Names of the modules to which LoRA is applied.
            - lora_dropout (float): Dropout rate for LoRA layers.
        model (PreTrainedModel): The pre-trained language model to be prepared for PEFT.

    Returns:
        PreTrainedModel: The model configured for parameter-efficient fine-tuning using QLoRA.

    """

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank*2,
        target_modules=args.target_modules, 
        lora_dropout=args.lora_dropout,
        task_type="CAUSAL_LM"
    )
    # model.lm_head.requires_grad = True
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print('Model for parameter efficient finetuning using QLoRA:', model)

    return model

def prepare_tokenizer(args, tokenizer, model):

    """
    Prepare and configure the tokenizer and model for fine-tuning with image and instruction tokens.

    This function sets up the tokenizer with custom tokens and templates specifically designed for 
    handling chat-based input and image tokens. It also resizes the model's embedding layer to 
    accommodate the new tokens and ensures that the new tokens have trainable embeddings.

    Args:
        args (Namespace): An object containing configuration parameters.
        tokenizer (PreTrainedTokenizer): The tokenizer to be configured.
        model (PreTrainedModel): The pre-trained language model to be updated with new embeddings.

    Returns:
        tuple: A tuple containing the configured tokenizer and model.

    Tokenizer Configuration:
        - Adds special tokens: ["<image>", "[INST]", "[/INST]"]
        - Sets `padding_side` to 'right'.
        - Sets the padding token to EOS (`</s>`).
        - Applies a chat template with the Mistral instruct format.

    Model Embedding Adjustment:
        - Resizes the model's token embeddings to accommodate new tokens.
        - Enables trainable embeddings for the newly added tokens.
        - Updates model configuration with the new padding and image token IDs.

    Side Effects:
        - Prints the model's embedding size before and after resizing.
        - Prints the tokenizer's vocabulary size after modification.
    """
    # Load Tokenizer
    print(" Loading Tokenizer")
    # print('Tokenizer special tokens:', tokenizer.special_tokens_map) 

    if tokenizer.chat_template is None:

        tokenizer.add_bos_token = False  # Matches sep=""
        tokenizer.add_eos_token = False  # EOS handled via sep2="</s>"
        
        tokenizer.chat_template = """\
        {% for message in messages %}
            {% if message['role'] == 'user' %}
                {{ '[INST]' }}
                {% if '<image>' not in message['content'] %}
                    <image>\n
                {% endif %}
                {{ message['content'] }}
                {{ '[/INST] ' }}
            {% else %}
                {{ message['content'] }}
                {{ '</s>' }}
            {% endif %}
        {% endfor %}\
        """
        print(" Chat Template Applied (Mistral Instruct: [INST] + </s>)")
    else:
        pass

    print("Model embedding size before resizing:", model.get_input_embeddings().weight.shape[0])

    tokenizer.padding_side = 'right'  
    new_tokens = ["<image>", "[INST]", "[/INST]"]
    tokenizer.add_tokens(new_tokens)
    tokenizer.pad_token = tokenizer.eos_token # replacing the current pad token from <unk> to </s>
    model.resize_token_embeddings(len(tokenizer))
    model.config.image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.padding_side = tokenizer.padding_side

    # Get embedding layer
    embeddings = model.get_input_embeddings()
    
    # Create new embeddings with trainable positions
    with torch.no_grad():
        # Clone existing embeddings
        new_embeddings = torch.nn.Embedding.from_pretrained(
            embeddings.weight.clone(),
            freeze=False
        )
        
        # Enable gradients only for new tokens
        for token in new_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            new_embeddings.weight[token_id].requires_grad = True
            
    # Replace model embeddings
    model.set_input_embeddings(new_embeddings)

    print("Model embedding size after resizing:", model.get_input_embeddings().weight.shape[0])
    print("Tokenizer vocab size:", len(tokenizer))

    return tokenizer, model
