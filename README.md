---
title: ProPrepPal
emoji: 📉
colorFrom: pink
colorTo: yellow
sdk: docker
pinned: false
app_port: 7860
---

<p align = "center" draggable=”false” ><img src="https://drive.google.com/file/d/1zbvsMo0jpCk_3VWhhlL3NMKhjHsfM3BU"
     width="200px"
     height="auto"/>
</p>

## <h1 align="center" id="heading">🔍 AI Makerspace Demo Day - ProPrepPal</h1>

### Outline:

#### Problem

Professionals often face difficulties in adequately preparing for meetings where active participation is crucial. The complexity and diversity of information required for effective participation can make preparation time-consuming and overwhelming.

#### Why it’s a Problem

Engagement in company meetings, events, and interviews demands varying depths of knowledge, ranging from detailed specific information to a broad understanding of multiple topics. This necessitates extensive research and preparation, which can be particularly challenging due to the time required and the organizational skills needed. With increasing professional demands, the ability to efficiently prepare for engagements is compromised, leading to decreased job satisfaction and productivity.

#### Solution

ProPrepPal is designed to be your personal assistant, dedicated to ensuring you are comprehensively prepared for any meeting or event. It functions by monitoring your schedule, identifying upcoming engagements, and automatically compiling crucial summaries, key information, pending issues, and relevant background details. Initially, Prepr will operate as a chatbot, providing essential information for your appointments. Future versions will evolve into a more dynamic AI agent capable of offering not only information but also coordination support.

#### Target Audience

The primary users of Prepr are professionals who frequently attend meetings, conferences, and other professional gatherings. Success Metrics The effectiveness of Prepr will be measured by the enhanced productivity of meetings and the increased satisfaction users experience with their time management and goal achievement. Integral to Prepr will be built-in metrics that provide feedback on the utility of the meeting preparations, the relevance of the information provided, and the overall efficacy of the tool in enhancing meeting outcomes.

# Build 🏗️

- The python notebook prepr.ipynb - this includes the RAG system, tests of the prompts, RAGAS and LangSmith
- The python script with RAG Pipeline with Chainlit App: app.py

# Ship 🚢

- The notebook was merged and integrated into the Chainlit App.
- To host this on Hugging Face, a Space was created, the application code and associated files were copied to the cloned directory
- The space is a Docker container securely hosted by Hugging Face.

# Share 🚀

- Enjoy the app hosted on Hugging Face
- https://huggingface.co/spaces/shoshana-levitt/pro-prep-pal
- https://huggingface.co/spaces/MikeCraBash/prepr
