---
title: Diabetes Expert SLM GUI
emoji: ⚕️
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
---

# 🩺 Diabetes Expert GUI

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/udaybhan10/DiabetesExpertGUI)

A professional Gradio interface for the **Diabetes Expert SLM**. This GUI allows users to chat with the model about 2026 ADA Standards of Care and deep pathophysiology logic locally or via the cloud.

## 💻 Hardware Requirements

- **Local Development (Python/MLX)**: Requires an **Apple Silicon (M1/M2/M3/M4/M5)** Mac. This version is hardware-accelerated via MLX for near-instant clinical reasoning.
- **Global Deployment (Docker)**: Cross-platform. Can run on Linux, Windows, or macOS (Intel/Apple) as it utilizes the `transformers` backend.

## 📥 Setup & Usage

### 🍎 Local Preview (Apple Silicon Only)
To run with hardware acceleration on your Mac:
1. **Clone and Install**:
   ```bash
   git clone https://github.com/udaybhan10/DiabetesExpertGUI.git
   cd DiabetesExpertGUI
   pip install -r requirements.txt
   ```
2. **Run MLX Demo**:
   ```bash
   python local_demo_mlx.py
   ```

### 🐳 Global/Docker Execution (Cross-Platform)
For Hugging Face Spaces or non-Mac systems:
1. **Build and Run**:
   ```bash
   docker build -t diabetes-expert-gui .
   docker run -p 7860:7860 diabetes-expert-gui
   ```

## ⚖️ Disclaimer
This model is for **educational and research purposes only**. It is not a medical device and should not be used for diagnosis or treatment decisions. Always consult a qualified healthcare professional.

## 📜 License
MIT License
