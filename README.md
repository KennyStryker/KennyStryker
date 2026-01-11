# 🚀 Profile

## 🛠️ Research Focus & Technical Expertise

| Focus Area | Machine Learning & Foundations | Generative AI & LLMs |
| :--- | :--- | :--- |
| **Architectures** | SVMs, XGBoost, ResNets, Dense-connections, Residual Bottlenecks, ConvNets. | Hypernetworks, Diffusion, Transformer Blocks, MoE (Mixture of Experts). |
| **Optimization** | Init Schemes, Triton Kernels, Adam. | RLHF, DPO, GRPO. |
| **Fine-Tuning** | Hyperparameter Opt (Bayesian), Transfer Learning, Domain Adaptation. | **PEFT:** LoRA, QLoRA, DoRA. |
| **Scaling & Infra** | Distributed Training (FSDP, DeepSpeed ZeRO 1-3). | **Inference:** FlashAttention-3, vLLM, Speculative Decoding. |
| **Model Compression** | Pruning, Knowledge Distillation, Weight Clustering. | **Quantization:** 4-bit/8-bit (bitsandbytes, GGUF). |
| **Theory & Math** | Manifold Density, Bias-Variance Tradeoff, Convergence Analysis. | Scaling Laws, Tokenization. |
| **Data & Eval** | Feature Engineering, PCA, AUC-ROC, Calibration Curves. | RAG, Synthetic Data, Reward Modeling. |
---

## 🔬 Featured Projects

### 🎨 [MMDIT-PyTorch](https://github.com/KennyStryker/mmdit-pytorch) | Multi-Modal Diffusion Transformer
*A high-performance PyTorch implementation of the Multi-Modal Diffusion Transformer (MM-DiT) architecture, as featured in Stable Diffusion 3.*
* **Core Innovation:** Implemented the "Symmetric Attention" mechanism where text and image modalities maintain their own learned embeddings while inter-modality information flows through a shared attention block.
* **Engineering:** Optimized for memory efficiency using scaled dot-product attention and modular block design to support varying sequence lengths across modalities.
* **Theory:** Integrated flow-matching objectives and time-conditioned modulation layers to achieve superior alignment between cross-modal representations.

### ⛓️ [mHC-PyTorch](https://github.com/KennyStryker/manifold-constrained-hyper-connections) | Manifold-Constrained Hyper-Connections
*An implementation of the mHC framework, designed to stabilize and scale widened residual streams in deep foundational models.*

* **Core Innovation:** Developed a projection-based connectivity layer that constrains residual mixing matrices to the **Birkhoff Polytope** (doubly stochastic manifold).
* **Engineering:** Integrated the **Sinkhorn-Knopp algorithm** for differentiable manifold projection and implemented kernel fusion techniques to minimize the memory access overhead caused by widened residual streams.
* **Theory:** Leveraged the closure property of doubly stochastic matrices under multiplication to guarantee stability.

---

## 🛠️ Technical Stack

- **Languages:** `Python`,  `SQL`
- **Frameworks:** `PyTorch`, `TensorFlow`, `Transformers`, `Diffusers`, `Scikit-Learn`, `OpenAI`, `LangChain`, `LangGraph`, `LangFuse`, `LlamaIndex`
- **Data Processing & Visualization:** `NumPy`, `Pandas`, `Matplotlib, Seaborn`
- **Databases:** `MySQL`, `MongoDB`, `PostgreSQL`
- **Vector Stores:** `Azure AI Search`, `Qdrant`, `Weaviate`, `Vertex AI Vector Store`, `Pinecone`
- **Cloud & DevOps Tools:** `Docker`, `Kubernetes`, `Microsoft Azure`, `Amazon Web Services (AWS)`, `Google Cloud Platform (GCP)`, `Lambda Labs Cloud`, `Runpod`, `Minio`, `S3`
- **Machine Learning & Deep Learning:** `Linear Regression`, `SVM`, `KNN`, `CNN`, `Transformer`, `GAN`, `Autoencoders`, `Reinforcement Learning`, `Large Language Models (LLMs)`, `Diffusion`, `MLP`
- **Applied Machine Learning:** `Retrieval-Augmented Generation (RAG)`, `Encoder-Decoder Modeling and Fine-Tuning`, `Contrastive Learning`, `Flash Attention`, `LoRA`, `QLoRA`, `DreamBooth`, `Hugging Face Ecosystem`
- **MLOps:** `FastAPI`, `Flask`, `ONNX`, `Weights&Biases`, `TensorBoard`, `MLFlow`, `KServe`, `TFServing`, `TorchServe`, `Llama.cpp`, `vLLM`, `Ollama`, `Nvidia Triton Inference Service`, `DeepSpeed`, `TensorRT`, `BitsAndBytes`

---

## 📬 Contact Information

* **Email:** [nggkenny@gmail.com](mailto:nggkenny@gmail.com)
