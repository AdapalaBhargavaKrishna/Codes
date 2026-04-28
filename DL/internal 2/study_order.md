# 📚 DL Internal 2 — Study Order (Hard → Easy)
### 2-Hour Exam Prep Strategy

> [!IMPORTANT]
> Study the **hard ones first** while your brain is fresh. The easy ones at the bottom share patterns with the harder ones, so you'll pick them up quickly later.

---

## Difficulty Ranking

| # | File | Difficulty | ⏱ Time | Key Challenge |
|---|------|-----------|---------|---------------|
| 1 | **BERT.py** | 🔴 Hardest | 20 min | Transformer architecture, 110M params, tokenizer internals, PyTorch vs TF, pre-training concepts (MLM, NSP) |
| 2 | **French.py** (Encoder-Decoder Translation) | 🔴 Hard | 20 min | Seq2Seq with **two separate models** (encoder + decoder), teacher forcing, inference loop, context vector, separate tokenizers |
| 3 | **GAN.py** | 🔴 Hard | 15 min | **Two competing networks**, adversarial loss (min-max game), Nash equilibrium, pseudo-labeling, unique training dynamics |
| 4 | **LSTM.py** (Next Word Prediction) | 🟠 Medium-Hard | 15 min | LSTM gates (forget/input/output), n-gram sequence creation, embedding layer, 4-gate parameter calculation |
| 5 | **ResNet vs VGG.py** (Ensemble) | 🟠 Medium | 10 min | Parallel dual-backbone, feature concatenation (512+2048=2560), frozen layers, ensemble concept |
| 6 | **denoising encoder.py** | 🟠 Medium | 10 min | Conv autoencoder (encoder+decoder), noise addition, UpSampling2D, noisy→clean mapping |
| 7 | **Alexnet 8.py** | 🟡 Medium-Easy | 8 min | 5 conv layers + 3 dense, large kernels (11×11), 46M params, Dropout — straightforward sequential CNN |
| 8 | **ResNet 50.py** (Transfer Learning) | 🟡 Easy | 5 min | Frozen ResNet50 + custom head, Flatten→Dense→Sigmoid, very similar to VGG16 |
| 9 | **Vgg 16.py** (Transfer Learning) | 🟡 Easy | 5 min | Frozen VGG16 + custom head, nearly identical pattern to ResNet50 |
| 10 | **simple encoder.py** | 🟢 Easy | 5 min | Simplest autoencoder: just 2 Dense layers (784→32→784), reconstruction task |
| 11 | **lenet5.py** | 🟢 Easiest | 4 min | Classic LeNet-5, small model (44K params), tanh + AveragePooling, MNIST digit classification |
| 12 | **custom kernels.py** | 🟢 Easiest | 3 min | No training at all, just applies conv filter to an image, edge detection demo |

**Total: ~120 min = 2 hours ✅**

---

## Why This Order Works

### 🔴 Tier 1 — Hardest (55 min) — Study First
| File | What Makes It Hard |
|------|--------------------|
| **BERT** | Completely different paradigm (Transformers, not CNN). Pre-trained 110M param model, tokenizer with `[CLS]`/`[SEP]`, `torch.no_grad()`, softmax vs sigmoid discussion, PyTorch syntax |
| **French (Enc-Dec)** | Two LSTM models connected via context vector `[h,c]`. Separate training vs inference models. Teacher forcing concept. `startseq`/`endseq` tokens. RMSprop optimizer (different from others) |
| **GAN** | Unique adversarial training — Generator vs Discriminator. Two separate loss functions. β1=0.5 Adam. Unsupervised/pseudo-labeled. Mode collapse, Nash equilibrium concepts |

### 🟠 Tier 2 — Medium (35 min)
| File | What Makes It Medium |
|------|---------------------|
| **LSTM** | Need to understand 4 LSTM gates, embedding layer, n-gram creation, padding. Self-supervised labeling |
| **ResNet vs VGG Ensemble** | Two frozen models in parallel → Concatenate → Dense. Unique ensemble architecture |
| **Denoising Encoder** | Conv2D autoencoder with noise injection. Need to understand encoder-decoder with UpSampling |

### 🟡🟢 Tier 3 — Easy (30 min) — Study Last
| File | Why It's Easy |
|------|---------------|
| **AlexNet** | Standard sequential CNN, just remember layer sizes |
| **ResNet 50** | Frozen base + Dense head — 3 lines of custom code |
| **VGG 16** | Almost identical to ResNet50 code, just different base model |
| **Simple Encoder** | 2 Dense layers, that's it |
| **LeNet-5** | Smallest CNN, classic architecture, very few params |
| **Custom Kernels** | No training, no loss, no optimizer — just filter visualization |

---

## ⚡ Quick-Reference: Common Patterns Across All 12

| Concept | Files That Use It |
|---------|------------------|
| **Binary Cross-Entropy** | AlexNet, VGG16, ResNet50, ResNet vs VGG, GAN, Simple Encoder, Denoising Encoder |
| **Categorical Cross-Entropy** | LeNet5, LSTM, French Translation, BERT |
| **Adam Optimizer** | All except French (RMSprop) and BERT (AdamW) |
| **Sigmoid output** | AlexNet, VGG16, ResNet50, ResNet vs VGG, GAN (discriminator) |
| **Softmax output** | LeNet5, LSTM, French, BERT |
| **Transfer Learning (frozen)** | VGG16, ResNet50, ResNet vs VGG |
| **Unlabeled/Self-supervised** | Simple Encoder, Denoising Encoder, GAN, LSTM |
| **Labeled (Supervised)** | LeNet5, AlexNet, VGG16, ResNet50, ResNet vs VGG, French |
| **MNIST dataset** | LeNet5, Simple Encoder, Denoising Encoder, GAN |
| **Dogs vs Cats dataset** | AlexNet, VGG16, ResNet50, ResNet vs VGG |

---

> [!TIP]
> **VGG16, ResNet50, and ResNet vs VGG** share 90% of their code structure. Study one well and the other two follow. Same goes for **Simple Encoder → Denoising Encoder** (the denoising one just adds noise + uses Conv2D instead of Dense).

Good luck with your exam! 🎯
