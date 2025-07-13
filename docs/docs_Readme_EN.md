<div align='center'>
    <img src="./images/head.jpg" alt="Happy-LLM Header Image" width="100%">
    <h1>Happy-LLM</h1>
</div>

<div align="center">
  <img src="https://img.shields.io/github/stars/datawhalechina/happy-llm?style=flat&logo=github" alt="GitHub stars"/>
  <img src="https://img.shields.io/github/forks/datawhalechina/happy-llm?style=flat&logo=github" alt="GitHub forks"/>
  <!-- <img src="https://img.shields.io/badge/language-Chinese-brightgreen?style=flat" alt="Language"/> --> <!-- Update this to "English" or add a dual-language badge -->
  <a href="https://github.com/datawhalechina/happy-llm"><img src="https://img.shields.io/badge/GitHub-Project-blue?style=flat&logo=github" alt="GitHub Project"></a>
  <a href="https://swanlab.cn/@kmno4/Happy-LLM/overview"><img src="https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg" alt="SwanLab"></a>
</div>

<div align="center">
  <a href="https://trendshift.io/repositories/14175" target="_blank"><img src="https://trendshift.io/api/badge/repositories/14175" alt="datawhalechina%2Fhappy-llm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

<div align="center">
  <h3>📚 A Tutorial on Large Language Models from Scratch: Principles and Practice</h3>
  <p><em>Deeply Understand LLM Core Principles and Hands-On Implement Your First Large Model</em></p>
</div>

---

## 🎯 Project Introduction

>   *Many folks, after finishing Datawhale's open-source project [self-llm: Guide to Using Open-Source Large Models](https://github.com/datawhalechina/self-llm), felt like they wanted more and desired a deeper dive into the principles and training processes of large language models. So, we (Datawhale) decided to launch the "Happy-LLM" project to help everyone deeply understand the principles and training of large language models.*

   This project is a **systematic LLM learning tutorial**, starting from basic NLP research methods and progressively delving into LLM ideas and principles. It breaks down the foundational architecture and training processes of LLMs for readers. At the same time, we'll combine the most mainstream code frameworks in the LLM field to demonstrate how to build and train an LLM from scratch, aiming to teach you to fish rather than just giving you a fish. We hope this book helps you step into the vast world of LLMs and explore their limitless possibilities.

### ✨ What You'll Gain?

- 📚 **Datawhale Open-Source and Free** Completely free access to all project content
- 🔍 **Deep Understanding** of Transformer architecture and attention mechanisms
- 📚 **Mastery** of pre-trained language model basics
- 🧠 **Knowledge** of existing large model structures
- 🏗️ **Hands-On Implementation** of a complete LLaMA2 model
- ⚙️ **Training Mastery** from pre-training to fine-tuning
- 🚀 **Practical Applications** like RAG, Agents, and other cutting-edge tech

## 📖 Content Navigation

| Chapter | Key Content | Status |
| --- | --- | --- |
| [Preface](./Foreword.md) | Project origins, background, and reader suggestions | ✅ |
| [Chapter 1: NLP Basics](./chapter1/Chapter 1 %20NLP Basic Concept.md) | What is NLP, development history, task classification, text representation evolution | ✅ |
| [Chapter 2: Transformer Architecture](./chapter2/Chapter 2%20Transformer architecture.md) | Attention mechanisms, Encoder-Decoder, hands-on building a Transformer | ✅ |
| [Chapter 3: Pre-Trained Language Models](./chapter3/Chapter 3%20 Pre-trained Language Model.md) | Comparison of Encoder-only, Encoder-Decoder, Decoder-Only models | ✅ |
| [Chapter 4: Large Language Models](./chapter4/Chapter 4%20 language models.md) | LLM definition, training strategies, emergenciency ability analysis | ✅ |
| [Chapter 5: Hands-On Building a Large Model](./chapter5/Chapter 5%20 Hands-on to build a big model.md) | Implementing LLaMA2, training Tokenizer, pre-training a small LLM | ✅ |
| [Chapter 6: Large Model Training Practice](./chapter6/Chapter 6%20 Big Model Training Process Practice.md) | Pre-training, supervised fine-tuning, efficient fine-tuning with LoRA/QLoRA | 🚧 |
| [Chapter 7: Large Model Applications](./chapter7/Chapter 7%20 Large Model Application.md) | Model evaluation, RAG retrieval augmentation, Agent intelligence | ✅ |

### Model Downloads

| Model Name | Download Link |
| --- | --- |
| Happy-LLM-Chapter5-Base-215M | [🤖 ModelScope](https://www.modelscope.cn/models/kmno4zx/happy-llm-215M-base) |
| Happy-LLM-Chapter5-SFT-215M | [🤖 ModelScope](https://www.modelscope.cn/models/kmno4zx/happy-llm-215M-sft) |

> *ModelScope Demo Space: [🤖 Demo Space](https://www.modelscope.cn/studios/kmno4zx/happy_llm_215M_sft)*

### PDF Version Download

   ***This Happy-LLM PDF tutorial is completely open-source and free. To prevent various marketers from adding watermarks and selling it to LLM beginners, we've pre-added a non-intrusive Datawhale open-source watermark to the PDF—please understand～***

> *Happy-LLM PDF: https://github.com/datawhalechina/happy-llm/releases/tag/PDF*  
> *Happy-LLM PDF Domestic Download: https://www.datawhale.cn/learn/summary/179*

## 💡 How to Learn

   This project is suitable for university students, researchers, and LLM enthusiasts. Before learning this project, it's recommended to have some programming experience, especially familiarity with the Python programming language. It's best to have knowledge of deep learning and understand related concepts and terminology in the NLP field to make learning this project easier.

   This project is divided into two parts: basic knowledge and practical applications. Chapters 1–4 cover the basic knowledge, introducing LLM principles from shallow to deep. Chapter 1 briefly introduces basic NLP tasks and development for non-NLP researchers; Chapter 2 introduces the core LLM architecture—Transformer, including principles and code implementation as the most important theoretical foundation for LLMs; Chapter 3 provides an overview of classic PLMs, including Encoder-Only, Encoder-Decoder, and Decoder-Only architectures, and also introduces the architectures and ideas of some current mainstream LLMs; Chapter 4 formally enters the LLM section, detailing LLM features, capabilities, and overall training processes. Chapters 5–7 are the practical application parts, gradually leading everyone into the underlying details of LLMs. Chapter 5 guides you to build an LLM from scratch based on PyTorch layers and implement the full process of pre-training and supervised fine-tuning; Chapter 6 introduces the current industry-standard LLM training framework Transformers, guiding learners to quickly and efficiently implement LLM training based on this framework; Chapter 7 introduces various LLM-based applications to complete learners' understanding of the LLM system, including LLM evaluation, Retrieval-Augmented Generation (RAG), and the concepts and simple implementations of Agents. You can selectively read relevant chapters based on personal interests and needs.

   While reading this book, it's recommended to combine theory with practice. LLM is a rapidly developing field that emphasizes practice, so we suggest investing more in hands-on work, reproducing the various codes provided in the book, and actively participating in LLM-related projects and competitions to truly dive into the wave of LLM development. We encourage you to follow Datawhale and other LLM-related open-source communities, and when you encounter problems, you can ask questions at any time in this project's issue area.

   Finally, we welcome every reader to join the ranks of LLM developers after learning this project. As a domestic AI open-source community, we hope to fully gather co-creators to enrich this open-source LLM world and create more comprehensive and distinctive LLM tutorials. Sparks gather into a sea. We hope to become the ladder between LLMs and the general public, embracing a more magnificent and vast LLM world with the spirit of free and equal open source.

## 🤝 How to Contribute

We welcome contributions in any form!

- 🐛 **Report Bugs** - Submit an Issue if you find a problem
- 💡 **Feature Suggestions** - Tell us if you have good ideas
- 📝 **Content Improvements** - Help refine the tutorial content
- 🔧 **Code Optimizations** - Submit a Pull Request

## 🙏 Acknowledgments

### Core Contributors
- [Song Zhixue - Project Lead](https://github.com/KMnO4-zx) (Datawhale Member - China University of Mining and Technology (Beijing))
- [Zou Yuheng - Project Lead](https://github.com/logan-zou) (Datawhale Member - University of International Business and Economics)
- [Zhu Xinzhong - Expert Advisor](https://xinzhongzhu.github.io/) (Datawhale Chief Scientist - Professor at Zhejiang Normal University Hangzhou Institute of Artificial Intelligence)

### Special Thanks
- Thanks to [@Sm1les](https://github.com/Sm1les) for help and support on this project
- Thanks to all developers who contributed to this project ❤️

<div align=center style="margin-top: 30px;">
  <a href="https://github.com/datawhalechina/happy-llm/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=datawhalechina/happy-llm" />
  </a>
</div>

## Star History

<div align='center'>
    <img src="./images/star-history-2025710.png" alt="Star History" width="90%">
</div>

<div align="center">
  <p>⭐ If this project helps you, please give us a Star!</p>
</div>

## About Datawhale

<div align='center'>
    <img src="./images/datawhale.png" alt="Datawhale Logo" width="30%">
    <p>Scan the QR code to follow Datawhale's official account and get more high-quality open-source content</p>
</div>

---

## 📜 Open-Source License

This work is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).