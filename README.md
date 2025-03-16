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

<style>
table {
    width: 100%;
    table-layout: fixed;
}
table th:nth-child(1) { width: 20%; }
table th:nth-child(2) { width: 50%; }
table th:nth-child(3) { width: 10%; }
table th:nth-child(4) { width: 20%; }
</style>

---



## 🛠️ LLM Training and Fine-Tuning

| Library | Description | Score | Alternatives |
|:--------|:------------|:-----:|:-------------|
| **Unsloth** | Fine-tune LLMs faster with less memory | 8 | LoRA, QLoRA |
| **PEFT** | State-of-the-art Parameter-Efficient Fine-Tuning | 9 | Adapters, Prompt Tuning |
| **TRL** | Train transformer models with reinforcement learning | 8 | RLHF-Toolkit, DeepRL |
| **Transformers** | Thousands of pretrained models for various tasks | 10 | Fairseq, OpenNMT |
| **Axolotl** | Streamlines post-training for various AI models | 7 | Llama-Factory, XTuring |
| **LLMBox** | Unified training pipeline and model evaluation | 7 | Ludwig, PyTorch Lightning |
| **LitGPT** | Train and fine-tune LLMs lightning fast | 8 | PyTorch Lightning, DeepSpeed |
| **Mergoo** | Easily merge multiple LLM experts and train efficiently | 7 | mergekit, ModelFusion |
| **Llama-Factory** | Easy and efficient LLM fine-tuning | 8 | Axolotl, Unsloth |
| **Ludwig** | Low-code framework for custom LLMs and AI models | 8 | H2O.ai, AutoKeras |
| **Txtinstruct** | Framework for training instruction-tuned models | 6 | EasyInstruct, Promptify |
| **Lamini** | Integrated LLM inference and tuning platform | 7 | XTuring, DeepSpeed |
| **XTuring** | Fast, efficient fine-tuning of open-source LLMs | 8 | Axolotl, Llama-Factory |
| **RL4LMs** | Modular RL library to fine-tune LLMs to preferences | 7 | TRL, DeepRL |
| **DeepSpeed** | Deep learning optimization for distributed training | 9 | Megatron-LM, Horovod |
| **torchtune** | PyTorch-native library for fine-tuning LLMs | 8 | PyTorch Lightning, Transformers |
| **PyTorch Lightning** | High-level interface for pretraining/fine-tuning | 9 | DeepSpeed, torchtune |
| **LoRA** | Low-Rank Adaptation for efficient fine-tuning | 8 | PEFT, QLoRA |
| **QLoRA** | Quantized LoRA for memory-efficient fine-tuning | 8 | LoRA, PEFT |

<div align="right">
  <a href="#-aiml-curated-list-of-aiml-libraries-tools-and-applications">
    <img src="https://img.shields.io/badge/Back_to_top-⬆️-blue" alt="Back to top" />
  </a>
</div>

## 📱 LLM Application Development Frameworks

| Library            | Description                                          | Score | Alternatives             |
|--------------------|------------------------------------------------------|-------|--------------------------|
| **LangChain**      | Framework for developing LLM-powered applications    | 10    | HayStack, Llama Index    |
| **Llama Index**    | Data framework for LLM applications                  | 9     | LangChain, HayStack      |
| **HayStack**       | End-to-end LLM framework with vector search         | 9     | LangChain, Llama Index   |
| **Prompt Flow**    | Tools to streamline LLM-based AI app development    | 8     | LangFlow, Griptape       |
| **Griptape**       | Modular Python framework for AI-powered apps        | 7     | Weave, LangChain         |
| **Weave**          | Toolkit for developing Generative AI applications   | 7     | Griptape, LangChain      |

<div align="right">
  <a href="#-aiml-curated-list-of-aiml-libraries-tools-and-applications">
    <img src="https://img.shields.io/badge/Back_to_top-⬆️-blue" alt="Back to top" />
  </a>
</div>

## 🌐 Multi API Access

| Library            | Description                                          | Score | Alternatives             |
|--------------------|------------------------------------------------------|-------|--------------------------|
| **LiteLLM**        | Call all LLM APIs using OpenAI format               | 9     | OpenAI, Anthropic        |
| **Embedchain**     | Framework for building RAG applications             | 8     | LangChain, Llama Index   |
| **Semantic Kernel** | Integration of LLM capabilities into applications   | 8     | LangChain, Llama Index   |

<div align="right">
  <a href="#-aiml-curated-list-of-aiml-libraries-tools-and-applications">
    <img src="https://img.shields.io/badge/Back_to_top-⬆️-blue" alt="Back to top" />
  </a>
</div>

## 🛤️ Routers

| Library | Description | Score | Alternatives |
|:--------|:------------|:-----:|:-------------|
| **Router** | Route requests between different LLM providers | 8 | LiteLLM, OpenRouter |
| **OpenRouter** | Single API for 50+ LLMs | 8 | Router, LiteLLM |

<div align="right">
  <a href="#-aiml-curated-list-of-aiml-libraries-tools-and-applications">
    <img src="https://img.shields.io/badge/Back_to_top-⬆️-blue" alt="Back to top" />
  </a>
</div>

## 🧠 Memory

| Library | Description | Score | Alternatives |
|:--------|:------------|:-----:|:-------------|
| **LLMCache** | Caching for LLM calls | 7 | Redis, Memcached |
| **Semantic Cache** | Cache LLM responses | 7 | LLMCache, Redis |

<div align="right">
  <a href="#-aiml-curated-list-of-aiml-libraries-tools-and-applications">
    <img src="https://img.shields.io/badge/Back_to_top-⬆️-blue" alt="Back to top" />
  </a>
</div>

## 🖥️ Interface

| Library            | Description                                          | Score | Alternatives             |
|--------------------|------------------------------------------------------|-------|--------------------------|
| **Gradio**         | Create UIs for ML models                            | 9     | Streamlit, Panel        |
| **Streamlit**      | Build data applications                             | 9     | Gradio, Panel           |
| **Panel**          | Create interactive web apps                         | 8     | Gradio, Streamlit       |



## 🧩 Low Code

| Library            | Description                                          | Score | Alternatives             |
|--------------------|------------------------------------------------------|-------|--------------------------|
| **Flowise**        | Drag & drop UI to build LLM flows                   | 8     | LangFlow, Chainlit      |
| **LangFlow**       | UI for LangChain                                    | 8     | Flowise, Chainlit       |
| **Chainlit**       | Build Python LLM apps                               | 8     | Flowise, LangFlow       |



## ⚡ Cache

| Library            | Description                                          | Score | Alternatives             |
|--------------------|------------------------------------------------------|-------|--------------------------|
| **LLMCache**       | Caching for LLM calls                               | 7     | Redis, Memcached        |
| **Semantic Cache** | Cache LLM responses                                 | 7     | LLMCache, Redis         |



## 📚 LLM RAG (Retrieval-Augmented Generation)

| Library | Description | Score | Alternatives |
|:--------|:------------|:-----:|:-------------|
| **FastGraph RAG** | Promptable framework for high-precision RAG | 7 | fastRAG, BeyondLLM |
| **Chonkie** | Lightweight, fast RAG chunking library | 6 | Llmware, RAG to Riches |
| **RAGChecker** | Fine-grained framework for diagnosing RAG systems | 7 | Ragas, Trulens |
| **RAG to Riches** | Build, scale, and deploy state-of-the-art RAG apps | 7 | BeyondLLM, fastRAG |
| **BeyondLLM** | All-in-one toolkit for RAG experimentation | 8 | fastRAG, Llmware |
| **SQLite-Vec** | Vector search SQLite extension for RAG | 7 | Chroma, FAISS |
| **fastRAG** | Research framework for efficient RAG pipelines | 8 | BeyondLLM, FlashRAG |
| **FlashRAG** | Python toolkit for efficient RAG research | 7 | fastRAG, BeyondLLM |
| **Llmware** | Framework for enterprise RAG pipelines | 8 | fastRAG, Vectara |
| **Rerankers** | Lightweight unified API for reranking models in RAG | 7 | ColBERT, Cross-Encoder |
| **Vectara** | Build agentic RAG applications | 8 | Llmware, fastRAG |
| **Chroma** | Open-source vector database for RAG | 8 | SQLite-Vec, FAISS |

<div align="right">
  <a href="#-aiml-curated-list-of-aiml-libraries-tools-and-applications">
    <img src="https://img.shields.io/badge/Back_to_top-⬆️-blue" alt="Back to top" />
  </a>
</div>

## ⚙️ LLM Inference

| Library | Description | Score | Alternatives |
|:--------|:------------|:-----:|:-------------|
| **vLLM** | High-throughput LLM inference & serving engine | 9 | Text Generation Inference |
| **CTranslate2** | Optimized inference engine for Transformers | 8 | vLLM, TensorRT-LLM |
| **TensorRT-LLM** | Optimize LLM inference on NVIDIA GPUs | 9 | vLLM, CTranslate2 |
| **OpenLLM** | Run inference with any open-source LLMs | 8 | vLLM, Text Generation |
| **Text Generation** | Large language model inference | 8 | vLLM, OpenLLM |
| **ExLlamaV2** | Optimized inference for LLMs | 8 | vLLM, TensorRT-LLM |
| **TorchServe** | Model serving framework for PyTorch | 8 | Ray Serve, BentoML |
| **Mosec** | High-performance model serving framework | 7 | Ray Serve, TorchServe |

<div align="right">
  <a href="#-aiml-curated-list-of-aiml-libraries-tools-and-applications">
    <img src="https://img.shields.io/badge/Back_to_top-⬆️-blue" alt="Back to top" />
  </a>
</div>

## 🌍 LLM Serving

| Library            | Description                                          | Score | Alternatives             |
|--------------------|------------------------------------------------------|-------|--------------------------|
| **Ray Serve**      | Scalable model serving framework                    | 9     | FastAPI, TorchServe     |
| **BentoML**        | Platform for ML model deployment                    | 8     | Ray Serve, TorchServe   |
| **TorchServe**     | Model serving framework for PyTorch                | 8     | Ray Serve, BentoML      |
| **Mosec**          | High-performance model serving framework            | 7     | Ray Serve, TorchServe   |


## 📜 LLM Data Extraction

| Library | Description | Score | Alternatives |
|:--------|:------------|:-----:|:-------------|
| **Unstructured** | Pre-process documents for LLM applications | 8 | Docquery, LangChain |
| **Docquery** | Extract data from documents using LLMs | 7 | Unstructured, LangChain |
| **LlamaParser** | Extract structured data from unstructured text | 7 | Unstructured, Docquery |
| **Nougat** | Document understanding system | 8 | Unstructured, Docquery |

<div align="right">
  <a href="#-aiml-curated-list-of-aiml-libraries-tools-and-applications">
    <img src="https://img.shields.io/badge/Back_to_top-⬆️-blue" alt="Back to top" />
  </a>
</div>

## 📦 LLM Data Generation

| Library | Description | Score | Alternatives |
|:--------|:------------|:-----:|:-------------|
| **DataDreamer** | Python library for synthetic data generation | 8 | fabricator, Promptwright |
| **fabricator** | Flexible framework to generate datasets with LLMs | 7 | DataDreamer, Promptwright |
| **Promptwright** | Synthetic dataset generation library | 6 | DataDreamer, fabricator |
| **EasyInstruct** | Easy-to-use instruction processing framework | 7 | Txtinstruct, Promptify |

<div align="right">
  <a href="#-aiml-curated-list-of-aiml-libraries-tools-and-applications">
    <img src="https://img.shields.io/badge/Back_to_top-⬆️-blue" alt="Back to top" />
  </a>
</div>

## 🤖 LLM Agents

| Library | Description | Score | Alternatives |
|:--------|:------------|:-----:|:-------------|
| **CrewAI** | Orchestrate role-playing, autonomous AI agents | 8 | AutoGen, LangGraph |
| **AutoGen** | Framework for developing LLM applications | 9 | CrewAI, LangGraph |
| **LangGraph** | Build stateful, multi-agent applications | 8 | CrewAI, AutoGen |

<div align="right">
  <a href="#-aiml-curated-list-of-aiml-libraries-tools-and-applications">
    <img src="https://img.shields.io/badge/Back_to_top-⬆️-blue" alt="Back to top" />
  </a>
</div>

## 📊 LLM Evaluation

| Library | Description | Score | Alternatives |
|:--------|:------------|:-----:|:-------------|
| **Ragas** | Framework for evaluating RAG systems | 8 | DeepEval, Trulens |
| **DeepEval** | Evaluation framework for LLMs | 7 | Ragas, Trulens |
| **Giskard** | Testing framework for ML models | 8 | DeepEval, Ragas |
| **Trulens** | Evaluation and tracking for LLM experiments | 8 | Ragas, DeepEval |
| **PromptBench** | Unified evaluation framework for LLMs | 7 | Lighteval, EvalPlus |
| **LangTest** | 60+ test types for comparing LLMs on accuracy/bias | 8 | Giskard, Ragas |
| **EvalPlus** | Rigorous evaluation framework for LLM4Code | 7 | PromptBench, Lighteval |
| **FastChat** | Platform for training/serving/evaluating chatbots | 8 | Evals, Trulens |
| **judges** | Small library of LLM judges for evaluation | 6 | Trulens, Ragas |
| **Evals** | Framework for evaluating LLMs with benchmarks | 8 | FastChat, Trulens |
| **AgentEvals** | Evaluators and utilities for agent performance | 7 | Trulens, Ragas |
| **LLMBox** | Unified training and evaluation pipelines | 7 | Ludwig, DeepEval |
| **Opik** | End-to-end LLM development platform with evaluation | 8 | LangSmith, Trulens |

<div align="right">
  <a href="#-aiml-curated-list-of-aiml-libraries-tools-and-applications">
    <img src="https://img.shields.io/badge/Back_to_top-⬆️-blue" alt="Back to top" />
  </a>
</div>

## 📈 LLM Monitoring

| Library | Description | Score | Alternatives |
|:--------|:------------|:-----:|:-------------|
| **Opik** | End-to-end LLM platform with monitoring features | 8 | LangSmith, Helicone |
| **LangSmith** | Tools for logging, monitoring, and improving LLM apps | 9 | Opik, W&B |
| **Weights & Biases** | Features for tracking LLM performance | 9 | LangSmith, Phoenix |
| **Helicone** | Open-source LLM observability platform | 8 | LangSmith, Evidently |
| **Evidently** | Open-source ML and LLM observability framework | 8 | Phoenix, Observers |
| **Phoenix** | Open-source AI observability platform | 8 | Evidently, W&B |
| **Observers** | Lightweight library for AI observability | 6 | Evidently, Phoenix |

<div align="right">
  <a href="#-aiml-curated-list-of-aiml-libraries-tools-and-applications">
    <img src="https://img.shields.io/badge/Back_to_top-⬆️-blue" alt="Back to top" />
  </a>
</div>

## ✍️ LLM Prompts

| Library | Description | Score | Alternatives |
|:--------|:------------|:-----:|:-------------|
| **PCToolkit** | Plug-and-play prompt compression toolkit for LLMs | 7 | LLMLingua, Selective Context |
| **Selective Context** | Compress prompts/context to process 2x more content | 7 | LLMLingua, PCToolkit |
| **LLMLingua** | Compress prompts to accelerate LLM inference | 8 | Selective Context, PCToolkit |
| **betterprompt** | Test suite for LLM prompts before production | 6 | Promptify, PromptSource |
| **Promptify** | Generate NLP task prompts for generative models | 7 | EasyInstruct, PromptSource |
| **PromptSource** | Toolkit for creating/sharing natural language prompts | 7 | Promptify, DSPy |
| **DSPy** | Framework for programming language models | 9 | Guidance, LMQL |
| **Py-priompt** | Prompt design library | 6 | Promptify, PromptSource |
| **Promptimizer** | Prompt optimization library | 6 | DSPy, Guidance |

<div align="right">
  <a href="#-aiml-curated-list-of-aiml-libraries-tools-and-applications">
    <img src="https://img.shields.io/badge/Back_to_top-⬆️-blue" alt="Back to top" />
  </a>
</div>

## 📋 LLM Structured Outputs

| Library | Description | Score | Alternatives |
|:--------|:------------|:-----:|:-------------|
| **Instructor** | Python library for structured outputs using Pydantic | 8 | Outlines, Guidance |
| **Outlines** | Type-safe structured generation | 8 | Instructor, Guidance |
| **Guidance** | Language for controlling text generation | 8 | LMQL, DSPy |
| **LMQL** | Programming language for LLM interaction | 8 | Guidance, DSPy |
| **XGrammar** | Efficient, flexible structured generation library | 7 | Instructor, LMQL |

<div align="right">
  <a href="#-aiml-curated-list-of-aiml-libraries-tools-and-applications">
    <img src="https://img.shields.io/badge/Back_to_top-⬆️-blue" alt="Back to top" />
  </a>
</div>

## 🔒 LLM Safety and Security

| Library | Description | Score | Alternatives |
|:--------|:------------|:-----:|:-------------|
| **EasyJailbreak** | Framework to generate adversarial jailbreak prompts | 7 | JailbreakEval, Garak |
| **Guardrails** | Add guardrails to LLMs for safer interactions | 8 | NeMo Guardrails, LLM Guard |
| **LLM Guard** | Security toolkit for LLM interactions | 8 | Guardrails, NeMo Guardrails |
| **AuditNLG** | Reduce risks in generative AI systems for language | 7 | Garak, LLM Guard |
| **NeMo Guardrails** | Toolkit for adding programmable guardrails to LLMs | 8 | Guardrails, LLM Guard |
| **Garak** | LLM vulnerability scanner | 7 | JailbreakEval, AuditNLG |

<div align="right">
  <a href="#-aiml-curated-list-of-aiml-libraries-tools-and-applications">
    <img src="https://img.shields.io/badge/Back_to_top-⬆️-blue" alt="Back to top" />
  </a>
</div>

## 🎁 Others

| Library | Description | Score | Alternatives |
|:--------|:------------|:-----:|:-------------|
| **LLM Reasoners** | Library for advanced LLM reasoning | 7 | DSPy, Guidance |
| **EasyEdit** | Easy-to-use knowledge editing framework for LLMs | 7 | mergekit, Promptimizer |
| **CodeTF** | One-stop transformer library for code LLMs | 7 | EvalPlus, Transformers |
| **spacy-llm** | Integrates LLMs into spaCy for NLP tasks | 8 | Transformers, NLTK |
| **pandas-ai** | Chat with databases (SQL, CSV, etc.) using LLMs | 8 | Vanna, SQLAlchemy |
| **Vanna** | Accurate Text-to-SQL generation via LLMs using RAG | 8 | pandas-ai, SQLAgent |
| **mergekit** | Tools for merging pretrained LLMs | 7 | Mergoo, ModelFusion |
| **MarkLLM** | Open-source toolkit for LLM watermarking | 6 | Garak, AuditNLG |
| **LLMSanitize** | Contamination detection in NLP datasets and LLMs | 6 | AuditNLG, Giskard |
| **Annotateai** | Automatically annotate papers using LLMs | 6 | DataDreamer, Promptwright |
| **LLM Reasoner** | Enhance LLMs to reason like OpenAI o1 or DeepSeek R1 | 7 | DSPy, LLM Reasoners |

<div align="right">
  <a href="#-aiml-curated-list-of-aiml-libraries-tools-and-applications">
    <img src="https://img.shields.io/badge/Back_to_top-⬆️-blue" alt="Back to top" />
  </a>
</div>

## ✨ How to Use

1. **Clone the repo**:  
   ```bash
   git clone https://github.com/OSSDeveloper/AIML.git
   ```

<div align="right">
  <a href="#-aiml-curated-list-of-aiml-libraries-tools-and-applications">
    <img src="https://img.shields.io/badge/Back_to_top-⬆️-blue" alt="Back to top" />
  </a>
</div>

## 🤝 Contributing

We welcome contributions! Please follow these steps:
1. Fork the repo
2. Add your library/tool with a description, score, and alternatives
3. Submit a pull request with a clear description of your changes

<div align="right">
  <a href="#-aiml-curated-list-of-aiml-libraries-tools-and-applications">
    <img src="https://img.shields.io/badge/Back_to_top-⬆️-blue" alt="Back to top" />
  </a>
</div>

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

<div align="right">
  <a href="#-aiml-curated-list-of-aiml-libraries-tools-and-applications">
    <img src="https://img.shields.io/badge/Back_to_top-⬆️-blue" alt="Back to top" />
  </a>
This project is licensed under the MIT License - see the LICENSE file for details.
