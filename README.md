# 🚀 AIML: Curated List of AI/ML Libraries, Tools, and Applications

---

## 🌟 Overview
Welcome to **AIML**! This repository curates the best libraries, frameworks, and tools for **AI and Machine Learning**, categorized for training, fine-tuning, application development, inference, and more. Each entry includes a description, evaluation score (1-10), and alternatives to help you choose the right tool for your needs.

<div align="center">
  <a href="https://github.com/OSSDeveloper/AIML/stargazers">
    <img src="https://img.shields.io/github/stars/OSSDeveloper/AIML?style=social" alt="Stars">
  </a>
  <a href="https://github.com/OSSDeveloper/AIML/network/members">
    <img src="https://img.shields.io/github/forks/OSSDeveloper/AIML?style=social" alt="Forks">
  </a>
  <a href="https://github.com/OSSDeveloper/AIML/issues">
    <img src="https://img.shields.io/github/issues/OSSDeveloper/AIML?color=red" alt="Issues">
  </a>
  <a href="https://github.com/OSSDeveloper/AIML/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/OSSDeveloper/AIML?color=blue" alt="License">
  </a>
</div>

---

## 📑 Table of Contents
- [LLM Training and Fine-Tuning](#llm-training-and-fine-tuning) 🛠️
- [LLM Application Development Frameworks](#llm-application-development-frameworks) 📱
- [Multi API Access](#multi-api-access) 🌐
- [Routers](#routers) 🛤️
- [Memory](#memory) 🧠
- [Interface](#interface) 🖥️
- [Low Code](#low-code) 🧩
- [Cache](#cache) ⚡
- [LLM RAG (Retrieval-Augmented Generation)](#llm-rag-retrieval-augmented-generation) 📚
- [LLM Inference](#llm-inference) ⚙️
- [LLM Serving](#llm-serving) 🌍
- [LLM Data Extraction](#llm-data-extraction) 📜
- [LLM Data Generation](#llm-data-generation) 📦
- [LLM Agents](#llm-agents) 🤖
- [LLM Evaluation](#llm-evaluation) 📊
- [LLM Monitoring](#llm-monitoring) 📈
- [LLM Prompts](#llm-prompts) ✍️
- [LLM Structured Outputs](#llm-structured-outputs) 📋
- [LLM Safety and Security](#llm-safety-and-security) 🔒
- [LLM Embedding Models](#llm-embedding-models) 📏
- [Others](#others) 🎁

---

## 🛠️ LLM Training and Fine-Tuning

| Library            | Description                                          | Score | Alternatives             | Link       |
|--------------------|------------------------------------------------------|-------|--------------------------|------------|
| **Unsloth**        | Fine-tune LLMs faster with less memory.              | 8     | LoRA, QLoRA              | [Link](#)  |
| **PEFT**           | State-of-the-art Parameter-Efficient Fine-Tuning.    | 9     | Adapters, Prompt Tuning  | [Link](#)  |
| **TRL**            | Train transformer models with reinforcement learning.| 8     | RLHF-Toolkit, DeepRL     | [Link](#)  |
| **Transformers**   | Thousands of pretrained models for various tasks.    | 10    | Fairseq, OpenNMT         | [Link](#)  |
| **Axolotl**        | Streamlines post-training for various AI models.     | 7     | Llama-Factory, XTuring   | [Link](#)  |
| **LLMBox**         | Unified training pipeline and model evaluation.      | 7     | Ludwig, PyTorch Lightning| [Link](#)  |
| **LitGPT**         | Train and fine-tune LLMs lightning fast.             | 8     | PyTorch Lightning, DeepSpeed | [Link](#)  |
| **Mergoo**         | Easily merge multiple LLM experts and train efficiently. | 7 | mergekit, ModelFusion    | [Link](#)  |
| **Llama-Factory**  | Easy and efficient LLM fine-tuning.                  | 8     | Axolotl, Unsloth         | [Link](#)  |
| **Ludwig**         | Low-code framework for custom LLMs and AI models.    | 8     | H2O.ai, AutoKeras        | [Link](#)  |
| **Txtinstruct**    | Framework for training instruction-tuned models.     | 6     | EasyInstruct, Promptify  | [Link](#)  |
| **Lamini**         | Integrated LLM inference and tuning platform.        | 7     | XTuring, DeepSpeed       | [Link](#)  |
| **XTuring**        | Fast, efficient fine-tuning of open-source LLMs.     | 8     | Axolotl, Llama-Factory   | [Link](#)  |
| **RL4LMs**         | Modular RL library to fine-tune LLMs to preferences. | 7     | TRL, DeepRL              | [Link](#)  |
| **DeepSpeed**      | Deep learning optimization for distributed training. | 9     | Megatron-LM, Horovod     | [Link](#)  |
| **torchtune**      | PyTorch-native library for fine-tuning LLMs.         | 8     | PyTorch Lightning, Transformers | [Link](#)  |
| **PyTorch Lightning** | High-level interface for pretraining/fine-tuning. | 9     | DeepSpeed, torchtune     | [Link](#)  |
| **LoRA**           | Low-Rank Adaptation for efficient fine-tuning.       | 8     | PEFT, QLoRA              | [Link](#)  |
| **QLoRA**          | Quantized LoRA for memory-efficient fine-tuning.     | 8     | LoRA, PEFT               | [Link](#)  |

---

## 📱 LLM Application Development Frameworks

| Library            | Description                                          | Score | Alternatives             | Link       |
|--------------------|------------------------------------------------------|-------|--------------------------|------------|
| **LangChain**      | Framework for developing LLM-powered applications.   | 10    | HayStack, Llama Index    | [Link](#)  |
| **Llama Index**    | Data framework for LLM applications.                 | 9     | LangChain, HayStack      | [Link](#)  |
| **HayStack**       | End-to-end LLM framework with vector search.         | 9     | LangChain, Llama Index   | [Link](#)  |
| **Prompt Flow**    | Tools to streamline LLM-based AI app development.    | 8     | LangFlow, Griptape       | [Link](#)  |
| **Griptape**       | Modular Python framework for AI-powered apps.        | 7     | Weave, LangChain         | [Link](#)  |
| **Weave**          | Toolkit for developing Generative AI applications.   | 7     | Griptape, LangChain      | [Link](#)  |
| **Llama Stack**    | Build Llama-based applications.                      | 6     | Llama Index, LangChain   | [Link](#)  |

---

## 🌐 Multi API Access

| Library            | Description                                          | Score | Alternatives             | Link       |
|--------------------|------------------------------------------------------|-------|--------------------------|------------|
| **LiteLLM**        | Call 100+ LLM APIs in OpenAI format.                 | 8     | AI Gateway, OpenRouter   | [Link](#)  |
| **AI Gateway**     | Fast AI gateway with guardrails for 200+ LLMs.       | 7     | LiteLLM, OpenLLM         | [Link](#)  |

---

## 🛤️ Routers

| Library            | Description                                          | Score | Alternatives             | Link       |
|--------------------|------------------------------------------------------|-------|--------------------------|------------|
| **RouteLLM**       | Framework for serving/evaluating LLM routers.        | 7     | LiteLLM, Custom Router   | [Link](#)  |

---

## 🧠 Memory

| Library            | Description                                          | Score | Alternatives             | Link       |
|--------------------|------------------------------------------------------|-------|--------------------------|------------|
| **mem0**           | Memory layer for AI apps.                            | 7     | Memoripy, LangChain Memory | [Link](#)  |
| **Memoripy**       | AI memory layer with semantic clustering.            | 6     | mem0, Memary             | [Link](#)  |

---

## 🖥️ Interface

| Library            | Description                                          | Score | Alternatives             | Link       |
|--------------------|------------------------------------------------------|-------|--------------------------|------------|
| **Streamlit**      | Build and share interactive data apps quickly.       | 9     | Gradio, Dash             | [Link](#)  |
| **Gradio**         | Build and share ML apps in Python.                   | 9     | Streamlit, Chainlit      | [Link](#)  |
| **AI SDK UI**      | Build chat and generative UIs.                       | 7     | Gradio, Streamlit        | [Link](#)  |
| **AI-Gradio**      | Create AI apps with various AI providers.            | 6     | Gradio, Chainlit         | [Link](#)  |
| **Simpleaichat**   | Simple package for interfacing with chat apps.       | 6     | Chainlit, LangChain      | [Link](#)  |
| **Chainlit**       | Build production-ready conversational AI apps.       | 8     | Gradio, Streamlit        | [Link](#)  |

---

## 🧩 Low Code

| Library            | Description                                          | Score | Alternatives             | Link       |
|--------------------|------------------------------------------------------|-------|--------------------------|------------|
| **LangFlow**       | Low-code builder for RAG and multi-agent AI apps.    | 8     | Prompt Flow, Ludwig      | [Link](#)  |

---

## ⚡ Cache

| Library            | Description                                          | Score | Alternatives             | Link       |
|--------------------|------------------------------------------------------|-------|--------------------------|------------|
| **GPTCache**       | Semantic cache for LLM queries to reduce costs.      | 8     | Redis, Custom Cache      | [Link](#)  |

---

## 📚 LLM RAG (Retrieval-Augmented Generation)

| Library            | Description                                          | Score | Alternatives             | Link       |
|--------------------|------------------------------------------------------|-------|--------------------------|------------|
| **FastGraph RAG**  | Promptable framework for high-precision RAG.         | 7     | fastRAG, BeyondLLM       | [Link](#)  |
| **Chonkie**        | Lightweight, fast RAG chunking library.              | 6     | Llmware, RAG to Riches   | [Link](#)  |
| **RAGChecker**     | Fine-grained framework for diagnosing RAG systems.   | 7     | Ragas, Trulens           | [Link](#)  |
| **RAG to Riches**  | Build, scale, and deploy state-of-the-art RAG apps.  | 7     | BeyondLLM, fastRAG       | [Link](#)  |
| **BeyondLLM**      | All-in-one toolkit for RAG experimentation.          | 8     | fastRAG, Llmware         | [Link](#)  |
| **SQLite-Vec**     | Vector search SQLite extension for RAG.              | 7     | Chroma, FAISS            | [Link](#)  |
| **fastRAG**        | Research framework for efficient RAG pipelines.      | 8     | BeyondLLM, FlashRAG      | [Link](#)  |
| **FlashRAG**       | Python toolkit for efficient RAG research.           | 7     | fastRAG, BeyondLLM       | [Link](#)  |
| **Llmware**        | Framework for enterprise RAG pipelines.              | 8     | fastRAG, Vectara         | [Link](#)  |
| **Rerankers**      | Lightweight unified API for reranking models in RAG. | 7     | ColBERT, Cross-Encoder   | [Link](#)  |
| **Vectara**        | Build agentic RAG applications.                      | 8     | Llmware, fastRAG         | [Link](#)  |
| **Chroma**         | Open-source vector database for RAG.                 | 8     | SQLite-Vec, FAISS        | [Link](#)  |

---

## ⚙️ LLM Inference

| Library            | Description                                          | Score | Alternatives             | Link       |
|--------------------|------------------------------------------------------|-------|--------------------------|------------|
| **LLM Compressor** | Apply compression algorithms to LLMs.                | 7     | vLLM, TensorRT-LLM       | [Link](#)  |
| **LightLLM**       | Lightweight, scalable, high-speed LLM inference.     | 7     | vLLM, torchchat          | [Link](#)  |
| **vLLM**           | High-throughput, memory-efficient LLM inference.     | 9     | TensorRT-LLM, LightLLM   | [Link](#)  |
| **torchchat**      | Run PyTorch LLMs locally on servers/desktop/mobile.  | 8     | vLLM, WebLLM             | [Link](#)  |
| **TensorRT-LLM**   | Optimize LLM inference with TensorRT.                | 9     | vLLM, DeepSpeed          | [Link](#)  |
| **WebLLM**         | High-performance in-browser LLM inference engine.    | 7     | torchchat, vLLM          | [Link](#)  |

---

## 🌍 LLM Serving

| Library            | Description                                          | Score | Alternatives             | Link       |
|--------------------|------------------------------------------------------|-------|--------------------------|------------|
| **Langcorn**       | Serve LangChain LLM apps and agents with FastAPI.    | 7     | LitServe, FastAPI        | [Link](#)  |
| **LitServe**       | Fast serving engine for AI models with GPU autoscaling. | 8  | vLLM, Langcorn           | [Link](#)  |

---

## 📜 LLM Data Extraction

| Library            | Description                                          | Score | Alternatives             | Link       |
|--------------------|------------------------------------------------------|-------|--------------------------|------------|
| **Crawl4AI**       | LLM-friendly web crawler and scraper.                | 7     | Crawlee, ScrapeGraphAI   | [Link](#)  |
| **ScrapeGraphAI**  | Web scraping with LLMs and graph logic.              | 8     | Crawl4AI, Llama Parse    | [Link](#)  |
| **Docling**        | Fast document parsing and export for LLM use cases.  | 7     | PyMuPDF4LLM, MegaParse   | [Link](#)  |
| **Llama Parse**    | GenAI-native document parser for complex data.       | 8     | Docling, PyMuPDF4LLM     | [Link](#)  |
| **PyMuPDF4LLM**    | Extract PDF content for LLM and RAG environments.    | 7     | Docling, Llama Parse     | [Link](#)  |
| **Crawlee**        | Web scraping and browser automation library.         | 8     | Crawl4AI, ScrapeGraphAI  | [Link](#)  |
| **MegaParse**      | Parser for all document types.                       | 7     | Llama Parse, Docling     | [Link](#)  |
| **ExtractThinker** | Document intelligence library for LLMs.              | 6     | Llama Parse, MegaParse   | [Link](#)  |

---

## 📦 LLM Data Generation

| Library            | Description                                          | Score | Alternatives             | Link       |
|--------------------|------------------------------------------------------|-------|--------------------------|------------|
| **DataDreamer**    | Python library for synthetic data generation.        | 8     | fabricator, Promptwright | [Link](#)  |
| **fabricator**     | Flexible framework to generate datasets with LLMs.   | 7     | DataDreamer, Promptwright | [Link](#)  |
| **Promptwright**   | Synthetic dataset generation library.                | 6     | DataDreamer, fabricator  | [Link](#)  |
| **EasyInstruct**   | Easy-to-use instruction processing framework.        | 7     | Txtinstruct, Promptify   | [Link](#)  |

---

## 🤖 LLM Agents

| Library            | Description                                          | Score | Alternatives             | Link       |
|--------------------|------------------------------------------------------|-------|--------------------------|------------|
| **CrewAI**         | Orchestrate role-playing, autonomous AI agents.      | 8     | AutoGen, LangGraph       | [Link](#)  |
| **LangGraph**      | Build resilient language agents as graphs.           | 9     | CrewAI, AutoGen          | [Link](#)  |
| **Agno**           | Build AI agents with memory, knowledge, and tools.   | 7     | AutoGen, Smolagents      | [Link](#)  |
| **AutoGen**        | Framework for building AI agent systems.             | 9     | CrewAI, LangGraph        | [Link](#)  |
| **Smolagents**     | Build powerful agents in few lines of code.          | 6     | Agno, Lagent             | [Link](#)  |
| **Pydantic AI**    | Python agent framework for production-grade apps.    | 7     | AutoGen, CrewAI          | [Link](#)  |
| **gradio-tools**   | Convert Gradio apps into tools for LLM agents.       | 6     | Composio, Browser Use    | [Link](#)  |
| **Composio**       | Production-ready toolset for AI agents.              | 7     | gradio-tools, Atomic Agents | [Link](#)  |
| **Atomic Agents**  | Build AI agents modularly.                           | 6     | Composio, Smolagents     | [Link](#)  |
| **Memary**         | Open-source memory layer for autonomous agents.      | 7     | mem0, Memoripy           | [Link](#)  |
| **Browser Use**    | Make websites accessible for AI agents.              | 6     | OpenWebAgent, Crawlee    | [Link](#)  |
| **OpenWebAgent**   | Toolkit to enable web agents on LLMs.                | 7     | Browser Use, Crawlee     | [Link](#)  |
| **Lagent**         | Lightweight framework for building LLM-based agents. | 7     | Smolagents, AutoGen      | [Link](#)  |
| **LazyLLM**        | Low-code tool for building multi-agent LLM apps.     | 7     | LangFlow, Swarms         | [Link](#)  |
| **Swarms**         | Enterprise-grade multi-agent orchestration framework.| 8     | AutoGen, CrewAI          | [Link](#)  |
| **ChatArena**      | Multi-agent language game environments for research. | 7     | Swarm, Agentarium        | [Link](#)  |
| **Swarm**          | Lightweight educational framework for multi-agents.  | 7     | Swarms, ChatArena        | [Link](#)  |
| **AgentStack**     | Fast way to build robust AI agents.                  | 6     | AutoGen, CrewAI          | [Link](#)  |
| **Archgw**         | Intelligent gateway for agents.                      | 6     | AI Gateway, Composio     | [Link](#)  |
| **Flow**           | Lightweight task engine for building AI agents.      | 6     | LangGraph, AutoGen       | [Link](#)  |
| **AgentOps**       | Python SDK for AI agent monitoring.                  | 7     | LangSmith, Helicone      | [Link](#)  |
| **Langroid**       | Multi-agent framework.                               | 7     | AutoGen, CrewAI          | [Link](#)  |
| **Agentarium**     | Framework for simulations with AI-powered agents.    | 6     | ChatArena, Swarm         | [Link](#)  |
| **Upsonic**        | Reliable AI agent framework with MCP support.        | 6     | Swarms, AutoGen          | [Link](#)  |

---

## 📊 LLM Evaluation

| Library            | Description                                          | Score | Alternatives             | Link       |
|--------------------|------------------------------------------------------|-------|--------------------------|------------|
| **Ragas**          | Toolkit for evaluating and optimizing LLM apps.      | 8     | DeepEval, Trulens        | [Link](#)  |
| **Giskard**        | Open-source evaluation/testing for ML/LLM systems.   | 8     | Ragas, LangTest          | [Link](#)  |
| **DeepEval**       | LLM evaluation framework.                            | 8     | Ragas, Trulens           | [Link](#)  |
| **Lighteval**      | All-in-one toolkit for evaluating LLMs.              | 7     | PromptBench, EvalPlus    | [Link](#)  |
| **Trulens**        | Evaluation and tracking for LLM experiments.         | 8     | Ragas, DeepEval          | [Link](#)  |
| **PromptBench**    | Unified evaluation framework for LLMs.               | 7     | Lighteval, EvalPlus      | [Link](#)  |
| **LangTest**       | 60+ test types for comparing LLMs on accuracy/bias.  | 8     | Giskard, Ragas           | [Link](#)  |
| **EvalPlus**       | Rigorous evaluation framework for LLM4Code.          | 7     | PromptBench, Lighteval   | [Link](#)  |
| **FastChat**       | Platform for training/serving/evaluating chatbots.   | 8     | Evals, Trulens           | [Link](#)  |
| **judges**         | Small library of LLM judges for evaluation.          | 6     | Trulens, Ragas           | [Link](#)  |
| **Evals**          | Framework for evaluating LLMs with benchmarks.       | 8     | FastChat, Trulens        | [Link](#)  |
| **AgentEvals**     | Evaluators and utilities for agent performance.      | 7     | Trulens, Ragas           | [Link](#)  |
| **LLMBox**         | Unified training and evaluation pipelines.           | 7     | Ludwig, DeepEval         | [Link](#)  |
| **Opik**           | End-to-end LLM development platform with evaluation. | 8     | LangSmith, Trulens       | [Link](#)  |

---

## 📈 LLM Monitoring

| Library            | Description                                          | Score | Alternatives             | Link       |
|--------------------|------------------------------------------------------|-------|--------------------------|------------|
| **Opik**           | End-to-end LLM platform with monitoring features.    | 8     | LangSmith, Helicone      | [Link](#)  |
| **LangSmith**      | Tools for logging, monitoring, and improving LLM apps.| 9     | Opik, W&B                | [Link](#)  |
| **Weights & Biases (W&B)** | Features for tracking LLM performance.       | 9     | LangSmith, Phoenix       | [Link](#)  |
| **Helicone**       | Open-source LLM observability platform.              | 8     | LangSmith, Evidently     | [Link](#)  |
| **Evidently**      | Open-source ML and LLM observability framework.      | 8     | Phoenix, Observers       | [Link](#)  |
| **Phoenix**        | Open-source AI observability platform.               | 8     | Evidently, W&B           | [Link](#)  |
| **Observers**      | Lightweight library for AI observability.            | 6     | Evidently, Phoenix       | [Link](#)  |

---

## ✍️ LLM Prompts

| Library            | Description                                          | Score | Alternatives             | Link       |
|--------------------|------------------------------------------------------|-------|--------------------------|------------|
| **PCToolkit**      | Plug-and-play prompt compression toolkit for LLMs.   | 7     | LLMLingua, Selective Context | [Link](#)  |
| **Selective Context** | Compress prompts/context to process 2x more content. | 7     | LLMLingua, PCToolkit     | [Link](#)  |
| **LLMLingua**      | Compress prompts to accelerate LLM inference.        | 8     | Selective Context, PCToolkit | [Link](#)  |
| **betterprompt**   | Test suite for LLM prompts before production.        | 6     | Promptify, PromptSource  | [Link](#)  |
| **Promptify**      | Generate NLP task prompts for generative models.     | 7     | EasyInstruct, PromptSource | [Link](#)  |
| **PromptSource**   | Toolkit for creating/sharing natural language prompts. | 7     | Promptify, DSPy          | [Link](#)  |
| **DSPy**           | Framework for programming language models.           | 9     | Guidance, LMQL           | [Link](#)  |
| **Py-priompt**     | Prompt design library.                               | 6     | Promptify, PromptSource  | [Link](#)  |
| **Promptimizer**   | Prompt optimization library.                         | 6     | DSPy, Guidance           | [Link](#)  |

---

## 📋 LLM Structured Outputs

| Library            | Description                                          | Score | Alternatives             | Link       |
|--------------------|------------------------------------------------------|-------|--------------------------|------------|
| **Instructor**     | Python library for structured outputs using Pydantic.| 8     | Outlines, Guidance       | [Link](#)  |
| **XGrammar**       | Efficient, flexible structured generation library.   | 7     | Instructor, LMQL         | [Link](#)  |
| **Outlines**       | Robust structured text generation.                   | 8     | Instructor, Guidance     | [Link](#)  |
| **Guidance**       | Efficient programming paradigm for steering LLMs.    | 9     | DSPy, LMQL               | [Link](#)  |
| **LMQL**           | Language for constraint-guided LLM programming.      | 8     | Guidance, DSPy           | [Link](#)  |
| **Jsonformer**     | Generate structured JSON from LLMs reliably.         | 7     | Instructor, Outlines     | [Link](#)  |

---

## 🔒 LLM Safety and Security

| Library            | Description                                          | Score | Alternatives             | Link       |
|--------------------|------------------------------------------------------|-------|--------------------------|------------|
| **JailbreakEval**  | Automated evaluators for assessing jailbreak attempts.| 7     | EasyJailbreak, Garak     | [Link](#)  |
| **EasyJailbreak**  | Framework to generate adversarial jailbreak prompts. | 7     | JailbreakEval, Garak     | [Link](#)  |
| **Guardrails**     | Add guardrails to LLMs for safer interactions.       | 8     | NeMo Guardrails, LLM Guard | [Link](#)  |
| **LLM Guard**      | Security toolkit for LLM interactions.               | 8     | Guardrails, NeMo Guardrails | [Link](#)  |
| **AuditNLG**       | Reduce risks in generative AI systems for language.  | 7     | Garak, LLM Guard         | [Link](#)  |
| **NeMo Guardrails**| Toolkit for adding programmable guardrails to LLMs.  | 8     | Guardrails, LLM Guard    | [Link](#)  |
| **Garak**          | LLM vulnerability scanner.                           | 7     | JailbreakEval, AuditNLG  | [Link](#)  |

---

## 📏 LLM Embedding Models

| Library            | Description                                          | Score | Alternatives             | Link       |
|--------------------|------------------------------------------------------|-------|--------------------------|------------|
| **Sentence-Transformers** | State-of-the-art text embeddings.             | 9     | Hugging Face Embeddings, FastText | [Link](#)  |
| **Model2Vec**      | Fast state-of-the-art static embeddings.             | 7     | Sentence-Transformers, Word2Vec | [Link](#)  |
| **Text Embedding Inference** | High-performance inference for text embeddings. | 8     | Sentence-Transformers, Model2Vec | [Link](#)  |

---

## 🎁 Others

| Library            | Description                                          | Score | Alternatives             | Link       |
|--------------------|------------------------------------------------------|-------|--------------------------|------------|
| **Text Machina**   | Framework for creating unbiased datasets for MGT.    | 7     | DataDreamer, fabricator  | [Link](#)  |
| **LLM Reasoners**  | Library for advanced LLM reasoning.                  | 7     | DSPy, Guidance           | [Link](#)  |
| **EasyEdit**       | Easy-to-use knowledge editing framework for LLMs.    | 7     | mergekit, Promptimizer   | [Link](#)  |
| **CodeTF**         | One-stop transformer library for code LLMs.          | 7     | EvalPlus, Transformers   | [Link](#)  |
| **spacy-llm**      | Integrates LLMs into spaCy for NLP tasks.            | 8     | Transformers, NLTK       | [Link](#)  |
| **pandas-ai**      | Chat with databases (SQL, CSV, etc.) using LLMs.     | 8     | Vanna, SQLAlchemy        | [Link](#)  |
| **LLM Transparency Tool** | Interactive toolkit for analyzing LLMs.       | 7     | Trulens, DeepEval        | [Link](#)  |
| **Vanna**          | Accurate Text-to-SQL generation via LLMs using RAG.  | 8     | pandas-ai, SQLAgent      | [Link](#)  |
| **mergekit**       | Tools for merging pretrained LLMs.                   | 7     | Mergoo, ModelFusion      | [Link](#)  |
| **MarkLLM**        | Open-source toolkit for LLM watermarking.            | 6     | Garak, AuditNLG          | [Link](#)  |
| **LLMSanitize**    | Contamination detection in NLP datasets and LLMs.    | 6     | AuditNLG, Giskard        | [Link](#)  |
| **Annotateai**     | Automatically annotate papers using LLMs.            | 6     | DataDreamer, Promptwright | [Link](#)  |
| **LLM Reasoner**   | Enhance LLMs to reason like OpenAI o1 or DeepSeek R1.| 7     | DSPy, LLM Reasoners      | [Link](#)  |

---

## ✨ How to Use

1. **Clone the repo**:  
   ```bash
   git clone https://github.com/OSSDeveloper/AIML.git


 Contributing
We welcome contributions! Please follow these steps:
Fork the repo.

Add your library/tool with a description, score, and alternatives.

Submit a pull request with a clear description of your changes.

 License
This project is licensed under the MIT License - see the LICENSE file for details.


