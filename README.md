               Toy LLM Fine-tuning & RAG Implementation
A Retrieval-Augmented Generation (RAG) system powered by a custom Toy LLM (Large Language Model) trained on the works of Jane Austen and finetuned with the works ofArthur Conan Doyle.

📖 Project Overview
This project implements a lightweight, locally-run RAG pipeline. It uses a fine tuned custom Transformer-based Toy LLM to generate responses that are contextually enriched by a FAISS vector database containing the complete Sherlock Holmes stories.

Key Features
Toy LLM Architecture: A custom transformer model built to demonstrate the mechanics of self-attention and generative text.

Domain-Specific Fine-tuning: Adapts the pre-trained transformer weights to capture the specific prose, vocabulary, and structural nuances of the Sherlock corpus, ensuring the model's internal "logic" matches the target era.

Hybrid Context: Blends the narrative style of Jane Austen with the deductive reasoning of Sherlock Holmes.

Local RAG: Uses LangChain and FAISS for efficient, local document retrieval to provide "knowledge" to the Toy LLM.

GPU Optimized: Designed to run in real-time on consumer hardware (e.g., NVIDIA RTX 3050).

🛠 Tech Stack
Language: Python 3.10+

Deep Learning: TensorFlow 2.10.1 (Native Windows GPU Support)

Orchestration: LangChain

Vector Store: FAISS

API Framework: FastAPI & Uvicorn

Embeddings: all-MiniLM-L6-v2 (via Sentence-Transformers)

🚀 Getting Started
1. Prerequisites
Ensure you have the following installed:

Python 3.10 (Recommended for TensorFlow 2.10 compatibility)

NVIDIA Drivers & CUDA 11.2 / cuDNN 8.1 (For GPU acceleration)

## 📂 PROJECT MAP

slm_rag/
├── config/                 # Global configurations (configs.py)
├── data/
│   ├── raw/                # Raw text corpora (doyle-sherlock.txt)
│   └── processed/          # FAISS indices & .h5 model weights
├── scripts/
│   └── hub_module.py       # 🧠 THE ORCHESTRATOR (Connects RAG, SLM, & API)
├── src/                    # THE ENGINE ROOM
│   ├── austen_slm/         # 🤖 TOY LLM CORE
│   │   ├── model.py        # Custom Transformer Architecture
│   │   ├── train.py        # Base training loop
│   │   └── data_loader.py  # Text preprocessing pipeline
│   ├── fine_tuning/        # 🎓 SPECIALIZATION
│   │   └── fine_tune.py    # Transfer learning logic
│   ├── hugging_face/       # ☁️ CLOUD SYNC
│   │   ├── hf_downloader.py# Fetches pretrained weights
│   │   └── upload_to_hub.py# Pushes trained models
│   └── rag/                # 📚 MEMORY SYSTEMS
│       ├── build_faiss.py  # Vector database construction
│       ├── langchain_wrapper.py # RAG Logic & Retrieval
│       └── expand_knowledge.py  # Inject new docs into memory
├── templates/              # Jinja2 Frontend (index.html)
├── main.py                 # FastAPI Application Entry
└── app.py                  # Legacy CLI Entry Point

The Git & Environment SetupBefore running any scripts, ensure you have your local repository and the correct environment active.

Clone the repository

git clone <your-repository-url>
cd slm_rag

Set up Experiment Tracking: Log in to your Weights & Biases account:

Bash

wandb login

# Install dependencies
pip install -r requirements.txt
 The Execution Sequence 
 Run these commands in the exact order listed below to build your "Neural Detective" from the ground up
 .StepCommandDescription
 1. Data Ingestionpython src/rag/expand_knowledge.py Cleans & chunks raw text (Sherlock/Austen) for both training and the RAG index.
 2. Base Training   python src/austen_slm/train.py Trains the Transformer core from scratch to learn basic language patterns.
 3. Specialization  python src/fine_tuning/fine_tune.py Adapts the model specifically to the Sherlock vs. Austen personality.
 4. Persistence     python src/hugging_face/upload_to_hub.py Saves your progress by pushing the weights to your Hugging Face account.
 5. Memory Build    python src/rag/build_faiss.py Vectorizes the knowledge base so the model can retrieve facts later.
 6. RAG Deployment      python main.py Launches the FastAPI server and the interactive dashboard.


