# Aligning LLaVA-1.6 7B with Human Preferences using ORPO

This repository contains an implementation for aligning the **LLaVA-1.6 7B Vision–Language Model (VLM)** with human preferences using **ORPO (Odds Ratio Preference Optimization)** on the **RLAIF-V** dataset.

The project explores preference-based alignment without using a separate reward or reference model, while operating under limited GPU memory using **QLoRA**.

---

## 🔍 Project Overview

* **Model:** LLaVA-1.6 7B (CLIP vision encoder + Mistral LLM)
* **Dataset:** RLAIF-V (human preference dataset for vision-language tasks)
* **Alignment Method:** ORPO (Monolithic Preference Optimization without Reference Model)
* **Fine-Tuning Method:** QLoRA (4-bit quantization + LoRA adapters)
* **Hardware:** 2× NVIDIA T4 GPUs (16 GB each)
* **Frameworks:** PyTorch, Hugging Face Transformers, Accelerate

---

## 🧠 ORPO at a Glance

ORPO aligns models by directly optimizing the preference ordering between a **chosen** and a **rejected** response.

```
log_odds = log P(chosen) − log P(rejected)
```

The training objective combines standard language modeling loss with a preference term:

```
L = L_ce − α · log(sigmoid(log_odds))
```

This removes the need for a separate reward model while still encouraging preferred outputs.

---

## ⚙️ Training Setup

* **Quantization:** NF4 with double quantization
* **LoRA ranks tested:** 8 and 16
* **Trainable parameters:** ~15.6M
* **Optimizer:** AdamW
* **Scheduler:** Cosine annealing
* **Training tricks used:**

  * Mixed precision training
  * Gradient accumulation
  * Gradient checkpointing

---

## 📈 Training Observations

* **ORPO loss** shows a gradual decrease, indicating effective preference learning
* **Positive and negative log probabilities** both decrease over time
* **Rejected responses** are suppressed more strongly than chosen ones
* **Log odds** show a slight increasing trend, confirming improved preference separation
* Loss curves exhibit fluctuations due to a small effective batch size

---

## 🖼️ Generation Behavior

* The model generates fluent and detailed responses
* However, it frequently produces **factually incorrect landmark descriptions**
* This indicates a **failure in visual grounding**, despite successful preference optimization
* Preference alignment alone does not guarantee factual or image-consistent outputs

---

## ⚠️ Limitations

* Hallucinated responses are not explicitly included as negative samples
* Vision encoder remains frozen during fine-tuning
* ORPO optimizes relative preference, not factual correctness
* Low effective batch size leads to noisy loss curves

---

## 🔮 Future Improvements

* Add hallucinated responses explicitly as rejected samples
* Increase LoRA rank or partially unfreeze vision-language projection layers
* Fine-tune vision encoder (or its last layers)
* Introduce auxiliary losses for image-text consistency
* Improve attention to image tokens during generation

---

## 📂 Repository Structure

```
.
├── main.py                  # Training entry point
├── orpo.py                  # ORPO loss implementation
├── dataloader.py            # Dataset and batching logic
├── train_and_validate.py    # Training and validation loops
├── generator.py             # Inference utilities
├── utils.py                 # Helper functions
└── requirements.txt
```

---

## 🚀 Run Training

Install dependencies and start training:

```bash
python main.py
```

Make sure the official LLaVA repository is cloned and properly set up, as the required model version is not directly available from Hugging Face.

---

## 📚 References

* LLaVA: Large Language and Vision Assistant
* ORPO: Odds Ratio Preference Optimization
* RLAIF-V Dataset
* QLoRA: Efficient Fine-Tuning of Quantized LLMs

---
