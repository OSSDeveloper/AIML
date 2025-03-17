# 🚀 AIML: Curated List of AI/ML Libraries, Tools, and Applications


## 🌟 Overview

Welcome to **AIML**! This repository curates the best libraries, frameworks, and tools for **AI and Machine Learning**, categorized for training, fine-tuning, application development, inference, and more. Each entry includes a description, evaluation score (1-10), and alternatives to help you choose the right tool for your needs.

<div align="center">
  <a href="https://github.com/OSSDeveloper/AIML/stargazers"><img src="https://img.shields.io/github/stars/OSSDeveloper/AIML?style=social" alt="Stars Badge"/></a>
  <a href="https://github.com/OSSDeveloper/AIML/network/members"><img src="https://img.shields.io/github/forks/OSSDeveloper/AIML?style=social" alt="Forks Badge"/></a>
  <a href="https://github.com/OSSDeveloper/AIML/issues"><img src="https://img.shields.io/github/issues/OSSDeveloper/AIML?color=red" alt="Issues Badge"/></a>
  <a href="https://github.com/OSSDeveloper/AIML/blob/main/LICENSE"><img src="https://img.shields.io/github/license/OSSDeveloper/AIML?color=blue" alt="License Badge"/></a>
</div>

<div align="center">
  <hr>
  <p>
    <sub>💪 Maintained by the community with ❤️</sub>
    <br>
    <sub>Star ⭐ this repository if you find it helpful!</sub>
  </p>
  
  <a href="https://github.com/OSSDeveloper/AIML/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=OSSDeveloper/AIML" />
  </a>
</div>

<div align="center">
  <h2>🔍 Quick Navigation</h2>
  
  <p align="center">
    <a href="#-llm-training-and-fine-tuning">🛠️ Training & Fine-Tuning</a> •
    <a href="#-llm-application-development-frameworks">📱 Development Frameworks</a> •
    <a href="#-multi-api-access">🌐 Multi API Access</a> •
    <a href="#-routers">🛤️ Routers</a>
  </p>
  
  <p align="center">
    <a href="#-memory">🧠 Memory</a> •
    <a href="#-interface">🖥️ Interface</a> •
    <a href="#-low-code">🧩 Low Code</a> •
    <a href="#-cache">⚡ Cache</a>
  </p>
  
  <p align="center">
    <a href="#-llm-rag">📚 RAG</a> •
    <a href="#-llm-inference">⚙️ Inference</a> •
    <a href="#-llm-serving">🌍 Serving</a> •
    <a href="#-llm-data-extraction">📜 Data Extraction</a>
  </p>
  
  <p align="center">
    <a href="#-llm-data-generation">📦 Data Generation</a> •
    <a href="#-llm-agents">🤖 Agents</a> •
    <a href="#-llm-evaluation">📊 Evaluation</a> •
    <a href="#-llm-safety-and-security">🔒 Safety</a>
  </p>

  <p align="center">
    <a href="#-llm-structured-outputs">📋 Structured Outputs</a> •
    <a href="#-others">🎁 Others</a> •
    <a href="#-how-to-use">✨ How to Use</a> •
    <a href="#-contributing">🤝 Contributing</a>
  </p>
</div>

---

## 🛠️ LLM Training and Fine-Tuning

| Library | GitHub Stats | Description | Score | Alternatives |
|:--------|:------------|:------------|:-----:|:-------------|
| **Unsloth** | ![Stars](https://img.shields.io/github/stars/unslothai/unsloth?style=flat) ![Issues](https://img.shields.io/github/issues/unslothai/unsloth?style=flat) ![PRs](https://img.shields.io/github/issues-pr/unslothai/unsloth?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/unslothai/unsloth?style=flat) | Fine-tune LLMs faster with less memory | 8 | LoRA, QLoRA |
| **PEFT** | ![Stars](https://img.shields.io/github/stars/huggingface/peft?style=flat) ![Issues](https://img.shields.io/github/issues/huggingface/peft?style=flat) ![PRs](https://img.shields.io/github/issues-pr/huggingface/peft?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/huggingface/peft?style=flat) | State-of-the-art Parameter-Efficient Fine-Tuning | 9 | Adapters, Prompt Tuning |
| **TRL** | ![Stars](https://img.shields.io/github/stars/huggingface/trl?style=flat) ![Issues](https://img.shields.io/github/issues/huggingface/trl?style=flat) ![PRs](https://img.shields.io/github/issues-pr/huggingface/trl?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/huggingface/trl?style=flat) | Train transformer models with reinforcement learning | 8 | RLHF-Toolkit, DeepRL |
| **Transformers** | ![Stars](https://img.shields.io/github/stars/huggingface/transformers?style=flat) ![Issues](https://img.shields.io/github/issues/huggingface/transformers?style=flat) ![PRs](https://img.shields.io/github/issues-pr/huggingface/transformers?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/huggingface/transformers?style=flat) | Thousands of pretrained models for various tasks | 10 | Fairseq, OpenNMT |
| **Axolotl** | ![Stars](https://img.shields.io/github/stars/OpenAccess-AI-Collective/axolotl?style=flat) ![Issues](https://img.shields.io/github/issues/OpenAccess-AI-Collective/axolotl?style=flat) ![PRs](https://img.shields.io/github/issues-pr/OpenAccess-AI-Collective/axolotl?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/OpenAccess-AI-Collective/axolotl?style=flat) | Streamlines post-training for various AI models | 7 | Llama-Factory, XTuring |
| **LitGPT** | ![Stars](https://img.shields.io/github/stars/Lightning-AI/lit-gpt?style=flat) ![Issues](https://img.shields.io/github/issues/Lightning-AI/lit-gpt?style=flat) ![PRs](https://img.shields.io/github/issues-pr/Lightning-AI/lit-gpt?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/Lightning-AI/lit-gpt?style=flat) | Train and fine-tune LLMs lightning fast | 8 | PyTorch Lightning, DeepSpeed |
| **Llama-Factory** | ![Stars](https://img.shields.io/github/stars/hiyouga/LLaMA-Factory?style=flat) ![Issues](https://img.shields.io/github/issues/hiyouga/LLaMA-Factory?style=flat) ![PRs](https://img.shields.io/github/issues-pr/hiyouga/LLaMA-Factory?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/hiyouga/LLaMA-Factory?style=flat) | Easy and efficient LLM fine-tuning | 8 | Axolotl, Unsloth |
| **torchtune** | [WIP - GitHub Stats Not Available] | PyTorch-native library for fine-tuning LLMs | 8 | PyTorch Lightning, Transformers |
| **PyTorch Lightning** | ![Stars](https://img.shields.io/github/stars/Lightning-AI/lightning?style=flat) ![Issues](https://img.shields.io/github/issues/Lightning-AI/lightning?style=flat) ![PRs](https://img.shields.io/github/issues-pr/Lightning-AI/lightning?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/Lightning-AI/lightning?style=flat) | High-level interface for pretraining/fine-tuning | 9 | DeepSpeed, torchtune |
| **LoRA** | ![Stars](https://img.shields.io/github/stars/microsoft/LoRA?style=flat) ![Issues](https://img.shields.io/github/issues/microsoft/LoRA?style=flat) ![PRs](https://img.shields.io/github/issues-pr/microsoft/LoRA?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/microsoft/LoRA?style=flat) | Low-Rank Adaptation for efficient fine-tuning | 8 | PEFT, QLoRA |
| **QLoRA** | ![Stars](https://img.shields.io/github/stars/artidoro/qlora?style=flat) ![Issues](https://img.shields.io/github/issues/artidoro/qlora?style=flat) ![PRs](https://img.shields.io/github/issues-pr/artidoro/qlora?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/artidoro/qlora?style=flat) | Quantized LoRA for memory-efficient fine-tuning | 8 | LoRA, PEFT |

<div align="right">
  <a href="#-aiml-curated-list-of-aiml-libraries-tools-and-applications">
    <img src="https://img.shields.io/badge/Back_to_top-⬆️-blue" alt="Back to top" />
  </a>
</div>

## 📱 LLM Application Development Frameworks

| Library | GitHub Stats | Description | Score | Alternatives |
|:--------|:------------|:------------|:-----:|:-------------|
| **LangChain** | ![Stars](https://img.shields.io/github/stars/langchain-ai/langchain?style=flat) ![Issues](https://img.shields.io/github/issues/langchain-ai/langchain?style=flat) ![PRs](https://img.shields.io/github/issues-pr/langchain-ai/langchain?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/langchain-ai/langchain?style=flat) | Framework for developing LLM-powered applications | 10 | HayStack, Llama Index |
| **Llama Index** | ![Stars](https://img.shields.io/github/stars/jerryjliu/llama_index?style=flat) ![Issues](https://img.shields.io/github/issues/jerryjliu/llama_index?style=flat) ![PRs](https://img.shields.io/github/issues-pr/jerryjliu/llama_index?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/jerryjliu/llama_index?style=flat) | Data framework for LLM applications | 9 | LangChain, HayStack |
| **HayStack** | ![Stars](https://img.shields.io/github/stars/deepset-ai/haystack?style=flat) ![Issues](https://img.shields.io/github/issues/deepset-ai/haystack?style=flat) ![PRs](https://img.shields.io/github/issues-pr/deepset-ai/haystack?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/deepset-ai/haystack?style=flat) | End-to-end LLM framework with vector search | 9 | LangChain, Llama Index |
| **Prompt Flow** | ![Stars](https://img.shields.io/github/stars/microsoft/promptflow?style=flat) ![Issues](https://img.shields.io/github/issues/microsoft/promptflow?style=flat) ![PRs](https://img.shields.io/github/issues-pr/microsoft/promptflow?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/microsoft/promptflow?style=flat) | Tools to streamline LLM-based AI app development | 8 | LangFlow, Griptape |
| **Griptape** | ![Stars](https://img.shields.io/github/stars/griptape-ai/griptape?style=flat) ![Issues](https://img.shields.io/github/issues/griptape-ai/griptape?style=flat) ![PRs](https://img.shields.io/github/issues-pr/griptape-ai/griptape?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/griptape-ai/griptape?style=flat) | Modular Python framework for AI-powered apps | 7 | Weave, LangChain |
| **Weave** | ![Stars](https://img.shields.io/github/stars/wandb/weave?style=flat) ![Issues](https://img.shields.io/github/issues/wandb/weave?style=flat) ![PRs](https://img.shields.io/github/issues-pr/wandb/weave?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/wandb/weave?style=flat) | Toolkit for developing Generative AI applications | 7 | Griptape, LangChain |

<div align="right">
  <a href="#-aiml-curated-list-of-aiml-libraries-tools-and-applications">
    <img src="https://img.shields.io/badge/Back_to_top-⬆️-blue" alt="Back to top" />
  </a>
</div>

## 🌐 Multi API Access

| Library            | GitHub Stats | Description                                          | Score | Alternatives             |
|--------------------|:------------|:------------|:-----:|:-------------|
| **LiteLLM** | ![Stars](https://img.shields.io/github/stars/BerriAI/litellm?style=flat) ![Issues](https://img.shields.io/github/issues/BerriAI/litellm?style=flat) ![PRs](https://img.shields.io/github/issues-pr/BerriAI/litellm?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/BerriAI/litellm?style=flat) | Call all LLM APIs using OpenAI format | 9 | OpenAI, Anthropic |
| **Embedchain** | ![Stars](https://img.shields.io/github/stars/embedchain/embedchain?style=flat) ![Issues](https://img.shields.io/github/issues/embedchain/embedchain?style=flat) ![PRs](https://img.shields.io/github/issues-pr/embedchain/embedchain?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/embedchain/embedchain?style=flat) | Framework for building RAG applications | 8 | LangChain, Llama Index |
| **Semantic Kernel** | ![Stars](https://img.shields.io/github/stars/microsoft/semantic-kernel?style=flat) ![Issues](https://img.shields.io/github/issues/microsoft/semantic-kernel?style=flat) ![PRs](https://img.shields.io/github/issues-pr/microsoft/semantic-kernel?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/microsoft/semantic-kernel?style=flat) | Integration of LLM capabilities into applications | 8 | LangChain, Llama Index |

<div align="right">
  <a href="#-aiml-curated-list-of-aiml-libraries-tools-and-applications">
    <img src="https://img.shields.io/badge/Back_to_top-⬆️-blue" alt="Back to top" />
  </a>
</div>

## 🛤️ Routers

| Library | GitHub Stats | Description | Score | Alternatives |
|:--------|:------------|:------------|:-----:|:-------------|
| **Router** | ![Stars](https://img.shields.io/github/stars/llmrouter/llm-router?style=flat) ![Issues](https://img.shields.io/github/issues/llmrouter/llm-router?style=flat) ![PRs](https://img.shields.io/github/issues-pr/llmrouter/llm-router?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/llmrouter/llm-router?style=flat) | Route requests between different LLM providers | 8 | LiteLLM, OpenRouter |
| **OpenRouter** | ![Stars](https://img.shields.io/github/stars/openrouter-dev/openrouter?style=flat) ![Issues](https://img.shields.io/github/issues/openrouter-dev/openrouter?style=flat) ![PRs](https://img.shields.io/github/issues-pr/openrouter-dev/openrouter?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/openrouter-dev/openrouter?style=flat) | Single API for 50+ LLMs | 8 | Router, LiteLLM |

<div align="right">
  <a href="#-aiml-curated-list-of-aiml-libraries-tools-and-applications">
    <img src="https://img.shields.io/badge/Back_to_top-⬆️-blue" alt="Back to top" />
  </a>
</div>

## 🧠 Memory

| Library | GitHub Stats | Description | Score | Alternatives |
|:--------|:------------|:------------|:-----:|:-------------|
| **LLMCache** | ![Stars](https://img.shields.io/github/stars/lastmile-ai/llmcache?style=flat) ![Issues](https://img.shields.io/github/issues/lastmile-ai/llmcache?style=flat) ![PRs](https://img.shields.io/github/issues-pr/lastmile-ai/llmcache?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/lastmile-ai/llmcache?style=flat) | Caching for LLM calls | 7 | Redis, Memcached |
| **Semantic Cache** | [WIP - GitHub Stats Not Available] | Cache LLM responses | 7 | LLMCache, Redis |

<div align="right">
  <a href="#-aiml-curated-list-of-aiml-libraries-tools-and-applications">
    <img src="https://img.shields.io/badge/Back_to_top-⬆️-blue" alt="Back to top" />
  </a>
</div>

## 🖥️ Interface

| Library | GitHub Stats | Description | Score | Alternatives |
|:--------|:------------|:------------|:-----:|:-------------|
| **Flowise** | ![Stars](https://img.shields.io/github/stars/FlowiseAI/Flowise?style=flat) ![Issues](https://img.shields.io/github/issues/FlowiseAI/Flowise?style=flat) ![PRs](https://img.shields.io/github/issues-pr/FlowiseAI/Flowise?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/FlowiseAI/Flowise?style=flat) | Drag & drop UI to build LLM flows | 8 | LangFlow, Chainlit |
| **LangFlow** | ![Stars](https://img.shields.io/github/stars/logspace-ai/langflow?style=flat) ![Issues](https://img.shields.io/github/issues/logspace-ai/langflow?style=flat) ![PRs](https://img.shields.io/github/issues-pr/logspace-ai/langflow?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/logspace-ai/langflow?style=flat) | UI for LangChain | 8 | Flowise, Chainlit |
| **Chainlit** | ![Stars](https://img.shields.io/github/stars/Chainlit/chainlit?style=flat) ![Issues](https://img.shields.io/github/issues/Chainlit/chainlit?style=flat) ![PRs](https://img.shields.io/github/issues-pr/Chainlit/chainlit?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/Chainlit/chainlit?style=flat) | Build Python LLM apps | 8 | Flowise, LangFlow |
| **Gradio** | ![Stars](https://img.shields.io/github/stars/gradio-app/gradio?style=flat) ![Issues](https://img.shields.io/github/issues/gradio-app/gradio?style=flat) ![PRs](https://img.shields.io/github/issues-pr/gradio-app/gradio?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/gradio-app/gradio?style=flat) | Create UIs for ML models | 9 | Streamlit, Panel |
| **Streamlit** | ![Stars](https://img.shields.io/github/stars/streamlit/streamlit?style=flat) ![Issues](https://img.shields.io/github/issues/streamlit/streamlit?style=flat) ![PRs](https://img.shields.io/github/issues-pr/streamlit/streamlit?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/streamlit/streamlit?style=flat) | Build data applications | 9 | Gradio, Panel |
| **Panel** | ![Stars](https://img.shields.io/github/stars/holoviz/panel?style=flat) ![Issues](https://img.shields.io/github/issues/holoviz/panel?style=flat) ![PRs](https://img.shields.io/github/issues-pr/holoviz/panel?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/holoviz/panel?style=flat) | Create interactive web apps | 8 | Gradio, Streamlit |

<div align="right">
  <a href="#-aiml-curated-list-of-aiml-libraries-tools-and-applications">
    <img src="https://img.shields.io/badge/Back_to_top-⬆️-blue" alt="Back to top" />
  </a>
</div>

## 🧩 Low Code

| Library | GitHub Stats | Description | Score | Alternatives |
|:--------|:------------|:------------|:-----:|:-------------|
| **Flowise** | ![Stars](https://img.shields.io/github/stars/FlowiseAI/Flowise?style=flat) ![Issues](https://img.shields.io/github/issues/FlowiseAI/Flowise?style=flat) ![PRs](https://img.shields.io/github/issues-pr/FlowiseAI/Flowise?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/FlowiseAI/Flowise?style=flat) | Drag & drop UI to build LLM flows | 8 | LangFlow, Chainlit |
| **LangFlow** | ![Stars](https://img.shields.io/github/stars/logspace-ai/langflow?style=flat) ![Issues](https://img.shields.io/github/issues/logspace-ai/langflow?style=flat) ![PRs](https://img.shields.io/github/issues-pr/logspace-ai/langflow?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/logspace-ai/langflow?style=flat) | UI for LangChain | 8 | Flowise, Chainlit |
| **Chainlit** | ![Stars](https://img.shields.io/github/stars/Chainlit/chainlit?style=flat) ![Issues](https://img.shields.io/github/issues/Chainlit/chainlit?style=flat) ![PRs](https://img.shields.io/github/issues-pr/Chainlit/chainlit?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/Chainlit/chainlit?style=flat) | Build Python LLM apps | 8 | Flowise, LangFlow |

<div align="right">
  <a href="#-aiml-curated-list-of-aiml-libraries-tools-and-applications">
    <img src="https://img.shields.io/badge/Back_to_top-⬆️-blue" alt="Back to top" />
  </a>
</div>


## ⚡ Cache

| Library | GitHub Stats | Description | Score | Alternatives |
|:--------|:------------|:------------|:-----:|:-------------|
| **GPTCache** | ![Stars](https://img.shields.io/github/stars/zilliztech/GPTCache?style=flat) ![Issues](https://img.shields.io/github/issues/zilliztech/GPTCache?style=flat) ![PRs](https://img.shields.io/github/issues-pr/zilliztech/GPTCache?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/zilliztech/GPTCache?style=flat) | Semantic cache for LLMs | 8 | Redis, Memcached |
| **LLMCache** | ![Stars](https://img.shields.io/github/stars/lastmile-ai/llmcache?style=flat) ![Issues](https://img.shields.io/github/issues/lastmile-ai/llmcache?style=flat) ![PRs](https://img.shields.io/github/issues-pr/lastmile-ai/llmcache?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/lastmile-ai/llmcache?style=flat) | Caching for LLM calls | 7 | GPTCache, Redis |

<div align="right">
  <a href="#-aiml-curated-list-of-aiml-libraries-tools-and-applications">
    <img src="https://img.shields.io/badge/Back_to_top-⬆️-blue" alt="Back to top" />
  </a>
</div>


## 📚 LLM RAG (Retrieval-Augmented Generation)

| Library | GitHub Stats | Description | Score | Alternatives |
|:--------|:------------|:------------|:-----:|:-------------|
| **LlamaIndex** | ![Stars](https://img.shields.io/github/stars/jerryjliu/llama_index?style=flat) ![Issues](https://img.shields.io/github/issues/jerryjliu/llama_index?style=flat) ![PRs](https://img.shields.io/github/issues-pr/jerryjliu/llama_index?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/jerryjliu/llama_index?style=flat) | Data framework for LLM applications | 9 | LangChain, HayStack |
| **ChromaDB** | ![Stars](https://img.shields.io/github/stars/chroma-core/chroma?style=flat) ![Issues](https://img.shields.io/github/issues/chroma-core/chroma?style=flat) ![PRs](https://img.shields.io/github/issues-pr/chroma-core/chroma?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/chroma-core/chroma?style=flat) | Open-source embedding database | 9 | Milvus, Weaviate |
| **Weaviate** | ![Stars](https://img.shields.io/github/stars/weaviate/weaviate?style=flat) ![Issues](https://img.shields.io/github/issues/weaviate/weaviate?style=flat) ![PRs](https://img.shields.io/github/issues-pr/weaviate/weaviate?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/weaviate/weaviate?style=flat) | Vector database for scale | 9 | ChromaDB, Milvus |
| **Milvus** | ![Stars](https://img.shields.io/github/stars/milvus-io/milvus?style=flat) ![Issues](https://img.shields.io/github/issues/milvus-io/milvus?style=flat) ![PRs](https://img.shields.io/github/issues-pr/milvus-io/milvus?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/milvus-io/milvus?style=flat) | Vector database for embeddings | 9 | Weaviate, ChromaDB |
| **txtai** | ![Stars](https://img.shields.io/github/stars/neuml/txtai?style=flat) ![Issues](https://img.shields.io/github/issues/neuml/txtai?style=flat) ![PRs](https://img.shields.io/github/issues-pr/neuml/txtai?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/neuml/txtai?style=flat) | Build semantic search applications | 8 | ChromaDB, Weaviate |
| **DocArray** | ![Stars](https://img.shields.io/github/stars/docarray/docarray?style=flat) ![Issues](https://img.shields.io/github/issues/docarray/docarray?style=flat) ![PRs](https://img.shields.io/github/issues-pr/docarray/docarray?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/docarray/docarray?style=flat) | Data structure for multimodal AI | 8 | ChromaDB, txtai |

<div align="right">
  <a href="#-aiml-curated-list-of-aiml-libraries-tools-and-applications">
    <img src="https://img.shields.io/badge/Back_to_top-⬆️-blue" alt="Back to top" />
  </a>
</div>

## ⚙️ LLM Inference

| Library | GitHub Stats | Description | Score | Alternatives |
|:--------|:------------|:------------|:-----:|:-------------|
| **vLLM** | ![Stars](https://img.shields.io/github/stars/vllm-project/vllm?style=flat) ![Issues](https://img.shields.io/github/issues/vllm-project/vllm?style=flat) ![PRs](https://img.shields.io/github/issues-pr/vllm-project/vllm?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/vllm-project/vllm?style=flat) | High-throughput LLM inference & serving engine | 9 | Text Generation Inference |
| **CTranslate2** | ![Stars](https://img.shields.io/github/stars/OpenNMT/CTranslate2?style=flat) ![Issues](https://img.shields.io/github/issues/OpenNMT/CTranslate2?style=flat) ![PRs](https://img.shields.io/github/issues-pr/OpenNMT/CTranslate2?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/OpenNMT/CTranslate2?style=flat) | Optimized inference engine for Transformers | 8 | vLLM, TensorRT-LLM |
| **TensorRT-LLM** | ![Stars](https://img.shields.io/github/stars/NVIDIA/TensorRT-LLM?style=flat) ![Issues](https://img.shields.io/github/issues/NVIDIA/TensorRT-LLM?style=flat) ![PRs](https://img.shields.io/github/issues-pr/NVIDIA/TensorRT-LLM?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/NVIDIA/TensorRT-LLM?style=flat) | Optimize LLM inference on NVIDIA GPUs | 9 | vLLM, CTranslate2 |
| **OpenLLM** | ![Stars](https://img.shields.io/github/stars/bentoml/OpenLLM?style=flat) ![Issues](https://img.shields.io/github/issues/bentoml/OpenLLM?style=flat) ![PRs](https://img.shields.io/github/issues-pr/bentoml/OpenLLM?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/bentoml/OpenLLM?style=flat) | Run inference with any open-source LLMs | 8 | vLLM, Text Generation |
| **Text Generation** | ![Stars](https://img.shields.io/github/stars/huggingface/text-generation-inference?style=flat) ![Issues](https://img.shields.io/github/issues/huggingface/text-generation-inference?style=flat) ![PRs](https://img.shields.io/github/issues-pr/huggingface/text-generation-inference?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/huggingface/text-generation-inference?style=flat) | Large language model inference | 8 | vLLM, OpenLLM |
| **ExLlamaV2** | ![Stars](https://img.shields.io/github/stars/turboderp/exllamav2?style=flat) ![Issues](https://img.shields.io/github/issues/turboderp/exllamav2?style=flat) ![PRs](https://img.shields.io/github/issues-pr/turboderp/exllamav2?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/turboderp/exllamav2?style=flat) | Optimized inference for LLMs | 8 | vLLM, TensorRT-LLM |
| **TorchServe** | ![Stars](https://img.shields.io/github/stars/pytorch/serve?style=flat) ![Issues](https://img.shields.io/github/issues/pytorch/serve?style=flat) ![PRs](https://img.shields.io/github/issues-pr/pytorch/serve?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/pytorch/serve?style=flat) | Model serving framework for PyTorch | 8 | Ray Serve, BentoML |
| **Mosec** | ![Stars](https://img.shields.io/github/stars/mosecorg/mosec?style=flat) ![Issues](https://img.shields.io/github/issues/mosecorg/mosec?style=flat) ![PRs](https://img.shields.io/github/issues-pr/mosecorg/mosec?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/mosecorg/mosec?style=flat) | High-performance model serving framework | 7 | Ray Serve, TorchServe |

<div align="right">
  <a href="#-aiml-curated-list-of-aiml-libraries-tools-and-applications">
    <img src="https://img.shields.io/badge/Back_to_top-⬆️-blue" alt="Back to top" />
  </a>
</div>

## 🌍 LLM Serving

| Library | GitHub Stats | Description | Score | Alternatives |
|:--------|:------------|:------------|:-----:|:-------------|
| **TorchServe** | ![Stars](https://img.shields.io/github/stars/pytorch/serve?style=flat) ![Issues](https://img.shields.io/github/issues/pytorch/serve?style=flat) ![PRs](https://img.shields.io/github/issues-pr/pytorch/serve?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/pytorch/serve?style=flat) | Model serving framework for PyTorch | 8 | Ray Serve, BentoML |
| **Ray Serve** | ![Stars](https://img.shields.io/github/stars/ray-project/ray?style=flat) ![Issues](https://img.shields.io/github/issues/ray-project/ray?style=flat) ![PRs](https://img.shields.io/github/issues-pr/ray-project/ray?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/ray-project/ray?style=flat) | Scalable model serving framework | 9 | TorchServe, BentoML |
| **BentoML** | ![Stars](https://img.shields.io/github/stars/bentoml/BentoML?style=flat) ![Issues](https://img.shields.io/github/issues/bentoml/BentoML?style=flat) ![PRs](https://img.shields.io/github/issues-pr/bentoml/BentoML?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/bentoml/BentoML?style=flat) | Platform for ML model deployment | 8 | TorchServe, Ray Serve |
| **Mosec** | ![Stars](https://img.shields.io/github/stars/mosecorg/mosec?style=flat) ![Issues](https://img.shields.io/github/issues/mosecorg/mosec?style=flat) ![PRs](https://img.shields.io/github/issues-pr/mosecorg/mosec?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/mosecorg/mosec?style=flat) | High-performance model serving framework | 7 | Ray Serve, TorchServe |

<div align="right">
  <a href="#-aiml-curated-list-of-aiml-libraries-tools-and-applications">
    <img src="https://img.shields.io/badge/Back_to_top-⬆️-blue" alt="Back to top" />
  </a>
</div>

## 📜 LLM Data Extraction

| Library | GitHub Stats | Description | Score | Alternatives |
|:--------|:------------|:------------|:-----:|:-------------|
| **Unstructured** | ![Stars](https://img.shields.io/github/stars/Unstructured-IO/unstructured?style=flat) ![Issues](https://img.shields.io/github/issues/Unstructured-IO/unstructured?style=flat) ![PRs](https://img.shields.io/github/issues-pr/Unstructured-IO/unstructured?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/Unstructured-IO/unstructured?style=flat) | Pre-process documents for LLM applications | 8 | Docquery, LangChain |
| **Docquery** | [Private Repository - Stats N/A] | Extract data from documents using LLMs | 7 | Unstructured, LangChain |
| **LlamaParser** | [Repository Archived] | Extract structured data from unstructured text | 7 | Unstructured, Docquery |
| **Nougat** | ![Stars](https://img.shields.io/github/stars/facebookresearch/nougat?style=flat) ![Issues](https://img.shields.io/github/issues/facebookresearch/nougat?style=flat) ![PRs](https://img.shields.io/github/issues-pr/facebookresearch/nougat?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/facebookresearch/nougat?style=flat) | Document understanding system | 8 | Unstructured, Docquery |

## 📦 LLM Data Generation

| Library | GitHub Stats | Description | Score | Alternatives |
|:--------|:------------|:------------|:-----:|:-------------|
| **DataDreamer** | ![Stars](https://img.shields.io/github/stars/datadreamer-dev/DataDreamer?style=flat) ![Issues](https://img.shields.io/github/issues/datadreamer-dev/DataDreamer?style=flat) ![PRs](https://img.shields.io/github/issues-pr/datadreamer-dev/DataDreamer?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/datadreamer-dev/DataDreamer?style=flat) | Python library for synthetic data generation | 8 | fabricator, Promptwright |
| **fabricator** | [Early Development - Stats N/A] | Flexible framework to generate datasets with LLMs | 7 | DataDreamer, Promptwright |
| **Promptwright** | [Closed Source] | Framework for generating synthetic data | 7 | DataDreamer, fabricator |
| **Syntheticpy** | [Repository Discontinued] | Generate synthetic data using LLMs | 6 | DataDreamer, fabricator |
| **Synthetic Data** | Tools for generating synthetic data | 6 | DataDreamer, Syntheticpy |

## 🤖 LLM Agents

| Library | GitHub Stats | Description | Score | Alternatives |
|:--------|:------------|:------------|:-----:|:-------------|
| **AutoGPT** | ![Stars](https://img.shields.io/github/stars/Significant-Gravitas/AutoGPT?style=flat) ![Issues](https://img.shields.io/github/issues/Significant-Gravitas/AutoGPT?style=flat) ![PRs](https://img.shields.io/github/issues-pr/Significant-Gravitas/AutoGPT?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/Significant-Gravitas/AutoGPT?style=flat) | Autonomous GPT-4 experiment | 8 | BabyAGI, AgentGPT |
| **BabyAGI** | ![Stars](https://img.shields.io/github/stars/yoheinakajima/babyagi?style=flat) ![Issues](https://img.shields.io/github/issues/yoheinakajima/babyagi?style=flat) ![PRs](https://img.shields.io/github/issues-pr/yoheinakajima/babyagi?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/yoheinakajima/babyagi?style=flat) | Task-driven autonomous agent | 7 | AutoGPT, AgentGPT |
| **AgentGPT** | ![Stars](https://img.shields.io/github/stars/reworkd/AgentGPT?style=flat) ![Issues](https://img.shields.io/github/issues/reworkd/AgentGPT?style=flat) ![PRs](https://img.shields.io/github/issues-pr/reworkd/AgentGPT?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/reworkd/AgentGPT?style=flat) | Autonomous AI agents in browser | 7 | AutoGPT, BabyAGI |
| **XAgent** | ![Stars](https://img.shields.io/github/stars/OpenBMB/XAgent?style=flat) ![Issues](https://img.shields.io/github/issues/OpenBMB/XAgent?style=flat) ![PRs](https://img.shields.io/github/issues-pr/OpenBMB/XAgent?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/OpenBMB/XAgent?style=flat) | Autonomous LLM agent framework | 7 | AutoGPT, BabyAGI |
| **SuperAGI** | ![Stars](https://img.shields.io/github/stars/TransformerOptimus/SuperAGI?style=flat) ![Issues](https://img.shields.io/github/issues/TransformerOptimus/SuperAGI?style=flat) ![PRs](https://img.shields.io/github/issues-pr/TransformerOptimus/SuperAGI?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/TransformerOptimus/SuperAGI?style=flat) | Dev framework for autonomous AI agents | 7 | AutoGPT, XAgent |
| **ix** | [Private Repository - Stats N/A] | Autonomous GPT-4 agent platform | 7 | AutoGPT, SuperAGI |
| **CrewAI** | ![Stars](https://img.shields.io/github/stars/joaomdmoura/crewAI?style=flat) ![Issues](https://img.shields.io/github/issues/joaomdmoura/crewAI?style=flat) ![PRs](https://img.shields.io/github/issues-pr/joaomdmoura/crewAI?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/joaomdmoura/crewAI?style=flat) | Framework for orchestrating role-playing AI agents | 8 | AutoGPT, SuperAGI |

<div align="right">
  <a href="#-aiml-curated-list-of-aiml-libraries-tools-and-applications">
    <img src="https://img.shields.io/badge/Back_to_top-⬆️-blue" alt="Back to top" />
  </a>
</div>

## 📋 LLM Structured Outputs

| Library | GitHub Stats | Description | Score | Alternatives |
|:--------|:------------|:------------|:-----:|:-------------|
| **Instructor** | ![Stars](https://img.shields.io/github/stars/jxnl/instructor?style=flat) ![Issues](https://img.shields.io/github/issues/jxnl/instructor?style=flat) ![PRs](https://img.shields.io/github/issues-pr/jxnl/instructor?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/jxnl/instructor?style=flat) | Python library for structured outputs using Pydantic | 8 | Outlines, Guidance |
| **Guidance** | ![Stars](https://img.shields.io/github/stars/microsoft/guidance?style=flat) ![Issues](https://img.shields.io/github/issues/microsoft/guidance?style=flat) ![PRs](https://img.shields.io/github/issues-pr/microsoft/guidance?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/microsoft/guidance?style=flat) | Language for controlling LLMs | 8 | LMQL, DSPy |
| **LMQL** | ![Stars](https://img.shields.io/github/stars/eth-sri/lmql?style=flat) ![Issues](https://img.shields.io/github/issues/eth-sri/lmql?style=flat) ![PRs](https://img.shields.io/github/issues-pr/eth-sri/lmql?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/eth-sri/lmql?style=flat) | Programming language for LLM interaction | 8 | Guidance, DSPy |
| **Outlines** | ![Stars](https://img.shields.io/github/stars/normal-computing/outlines?style=flat) ![Issues](https://img.shields.io/github/issues/normal-computing/outlines?style=flat) ![PRs](https://img.shields.io/github/issues-pr/normal-computing/outlines?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/normal-computing/outlines?style=flat) | Type-safe structured generation with LLMs | 7 | Instructor, Guidance |
| **Jsonformer** | ![Stars](https://img.shields.io/github/stars/1rgs/jsonformer?style=flat) ![Issues](https://img.shields.io/github/issues/1rgs/jsonformer?style=flat) ![PRs](https://img.shields.io/github/issues-pr/1rgs/jsonformer?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/1rgs/jsonformer?style=flat) | Structured JSON generation with LLMs | 7 | Instructor, Outlines |
| **Guardrails** | ![Stars](https://img.shields.io/github/stars/ShreyaR/guardrails?style=flat) ![Issues](https://img.shields.io/github/issues/ShreyaR/guardrails?style=flat) ![PRs](https://img.shields.io/github/issues-pr/ShreyaR/guardrails?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/ShreyaR/guardrails?style=flat) | Add structure, type safety, and security to LLM outputs | 8 | Instructor, NeMo Guardrails |

## 📊 LLM Evaluation

| Library | GitHub Stats | Description | Score | Alternatives |
|:--------|:------------|:------------|:-----:|:-------------|
| **Helicone** | ![Stars](https://img.shields.io/github/stars/helicone/helicone?style=flat) ![Issues](https://img.shields.io/github/issues/helicone/helicone?style=flat) ![PRs](https://img.shields.io/github/issues-pr/helicone/helicone?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/helicone/helicone?style=flat) | Open-source LLM observability platform | 8 | LangSmith, Evidently |
| **Evidently** | ![Stars](https://img.shields.io/github/stars/evidentlyai/evidently?style=flat) ![Issues](https://img.shields.io/github/issues/evidentlyai/evidently?style=flat) ![PRs](https://img.shields.io/github/issues-pr/evidentlyai/evidently?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/evidentlyai/evidently?style=flat) | Open-source ML and LLM observability framework | 8 | Phoenix, Observers |
| **Phoenix** | [Early Development - Stats N/A] | Open-source AI observability platform | 8 | Evidently, W&B |
| **Observers** | [Repository Archived] | Lightweight library for AI observability | 6 | Evidently, Phoenix |

<div align="right">
  <a href="#-aiml-curated-list-of-aiml-libraries-tools-and-applications">
    <img src="https://img.shields.io/badge/Back_to_top-⬆️-blue" alt="Back to top" />
  </a>
</div>

## 📝 LLM Prompts

| Library | GitHub Stats | Description | Score | Alternatives |
|:--------|:------------|:------------|:-----:|:-------------|
| **DSPy** | ![Stars](https://img.shields.io/github/stars/stanfordnlp/dspy?style=flat) ![Issues](https://img.shields.io/github/issues/stanfordnlp/dspy?style=flat) ![PRs](https://img.shields.io/github/issues-pr/stanfordnlp/dspy?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/stanfordnlp/dspy?style=flat) | Framework for programming language models | 9 | Guidance, LMQL |
| **Promptify** | ![Stars](https://img.shields.io/github/stars/promptslab/Promptify?style=flat) ![Issues](https://img.shields.io/github/issues/promptslab/Promptify?style=flat) ![PRs](https://img.shields.io/github/issues-pr/promptslab/Promptify?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/promptslab/Promptify?style=flat) | Generate NLP task prompts for generative models | 7 | EasyInstruct, PromptSource |
| **PromptSource** | ![Stars](https://img.shields.io/github/stars/bigscience-workshop/promptsource?style=flat) ![Issues](https://img.shields.io/github/issues/bigscience-workshop/promptsource?style=flat) ![PRs](https://img.shields.io/github/issues-pr/bigscience-workshop/promptsource?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/bigscience-workshop/promptsource?style=flat) | Toolkit for creating/sharing natural language prompts | 7 | Promptify, DSPy |
| **LLMLingua** | ![Stars](https://img.shields.io/github/stars/microsoft/LLMLingua?style=flat) ![Issues](https://img.shields.io/github/issues/microsoft/LLMLingua?style=flat) ![PRs](https://img.shields.io/github/issues-pr/microsoft/LLMLingua?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/microsoft/LLMLingua?style=flat) | Compress prompts to accelerate LLM inference | 8 | Selective Context, PCToolkit |

## 🔒 LLM Safety and Security

| Library | GitHub Stats | Description | Score | Alternatives |
|:--------|:------------|:------------|:-----:|:-------------|
| **Guardrails** | ![Stars](https://img.shields.io/github/stars/ShreyaR/guardrails?style=flat) ![Issues](https://img.shields.io/github/issues/ShreyaR/guardrails?style=flat) ![PRs](https://img.shields.io/github/issues-pr/ShreyaR/guardrails?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/ShreyaR/guardrails?style=flat) | Add guardrails to LLMs for safer interactions | 8 | NeMo Guardrails, LLM Guard |
| **LLM Guard** | ![Stars](https://img.shields.io/github/stars/protectai/llm-guard?style=flat) ![Issues](https://img.shields.io/github/issues/protectai/llm-guard?style=flat) ![PRs](https://img.shields.io/github/issues-pr/protectai/llm-guard?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/protectai/llm-guard?style=flat) | Security toolkit for LLM interactions | 8 | Guardrails, NeMo Guardrails |
| **NeMo Guardrails** | ![Stars](https://img.shields.io/github/stars/NVIDIA/NeMo-Guardrails?style=flat) ![Issues](https://img.shields.io/github/issues/NVIDIA/NeMo-Guardrails?style=flat) ![PRs](https://img.shields.io/github/issues-pr/NVIDIA/NeMo-Guardrails?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/NVIDIA/NeMo-Guardrails?style=flat) | Toolkit for adding programmable guardrails to LLMs | 8 | Guardrails, LLM Guard |
| **Garak** | ![Stars](https://img.shields.io/github/stars/leondz/garak?style=flat) ![Issues](https://img.shields.io/github/issues/leondz/garak?style=flat) ![PRs](https://img.shields.io/github/issues-pr/leondz/garak?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/leondz/garak?style=flat) | LLM vulnerability scanner | 7 | JailbreakEval, AuditNLG |

## 🎁 Others

| Library | GitHub Stats | Description | Score | Alternatives |
|:--------|:------------|:------------|:-----:|:-------------|
| **pandas-ai** | ![Stars](https://img.shields.io/github/stars/gventuri/pandas-ai?style=flat) ![Issues](https://img.shields.io/github/issues/gventuri/pandas-ai?style=flat) ![PRs](https://img.shields.io/github/issues-pr/gventuri/pandas-ai?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/gventuri/pandas-ai?style=flat) | Chat with databases using LLMs | 8 | Vanna, SQLAlchemy |
| **Vanna** | ![Stars](https://img.shields.io/github/stars/vanna-ai/vanna?style=flat) ![Issues](https://img.shields.io/github/issues/vanna-ai/vanna?style=flat) ![PRs](https://img.shields.io/github/issues-pr/vanna-ai/vanna?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/vanna-ai/vanna?style=flat) | Accurate Text-to-SQL generation via LLMs using RAG | 8 | pandas-ai, SQLAgent |
| **mergekit** | ![Stars](https://img.shields.io/github/stars/cg123/mergekit?style=flat) ![Issues](https://img.shields.io/github/issues/cg123/mergekit?style=flat) ![PRs](https://img.shields.io/github/issues-pr/cg123/mergekit?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/cg123/mergekit?style=flat) | Tools for merging pretrained LLMs | 7 | Mergoo, ModelFusion |
| **LLM Reasoner** | ![Stars](https://img.shields.io/github/stars/microsoft/LLMReasoner?style=flat) ![Issues](https://img.shields.io/github/issues/microsoft/LLMReasoner?style=flat) ![PRs](https://img.shields.io/github/issues-pr/microsoft/LLMReasoner?style=flat) ![Last Commit](https://img.shields.io/github/last-commit/microsoft/LLMReasoner?style=flat) | Enhance LLMs to reason like OpenAI o1 or DeepSeek R1 | 7 | DSPy, LLM Reasoners |

<div align="right">
  <a href="#-aiml-curated-list-of-aiml-libraries-tools-and-applications">
    <img src="https://img.shields.io/badge/Back_to_top-⬆️-blue" alt="Back to top" />
  </a>
</div>

## ✨ How to Use

1. **Browse Categories**: Navigate through different sections using the Quick Navigation links
2. **Compare Tools**: Each entry includes:
   - Description of the tool/library
   - Evaluation score (1-10)
   - Alternative options
   - GitHub statistics (stars, issues, PRs, last commit)
3. **Contribute**: Help keep this list current by submitting PRs for new tools or updates

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/add-new-tool`)
3. Add your changes following the existing format
4. Commit your changes (`git commit -m 'Add new tool: ToolName'`)
5. Push to the branch (`git push origin feature/add-new-tool`)
6. Open a Pull Request

### Contribution Guidelines

- Ensure the tool is actively maintained
- Include all required fields (description, score, alternatives)
- Add GitHub stats badges where applicable
- Keep descriptions concise and informative

## 📝 TODO

1. **Library Links Enhancement**
   - [ ] Convert all library names to clickable links
   - [ ] Ensure links open in new tab using `target="_blank"` attribute
   - [ ] Format: `<a href="https://github.com/org/repo" target="_blank">LibraryName</a>`

2. **Score Automation**
   - [ ] Develop script to calculate score based on GitHub statistics:
     - Stars count
     - Issues/PRs ratio
     - Last commit recency
     - Release frequency
     - Contributors count
   - [ ] Automate score updates using GitHub Actions
   - [ ] Add methodology documentation for score calculation

<div align="right">
  <a href="#-aiml-curated-list-of-aiml-libraries-tools-and-applications">
    <img src="https://img.shields.io/badge/Back_to_top-⬆️-blue" alt="Back to top" />
  </a>
</div>


## 📄 License

This project is licensed under the MIT License.

<div align="right">
  <a href="#-aiml-curated-list-of-aiml-libraries-tools-and-applications">
    <img src="https://img.shields.io/badge/Back_to_top-⬆️-blue" alt="Back to top" />
  </a>
</div>
```


