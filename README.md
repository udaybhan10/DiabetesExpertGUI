# 🩺 Diabetes Expert GUI

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/udaybhan10/DiabetesExpertGUI)

A professional Gradio interface for the **Diabetes Expert SLM**. This GUI allows users to chat with the model about 2026 ADA Standards of Care and deep pathophysiology logic locally or via the cloud.

## 🚀 Features
- **Medical Expertise**: Fine-tuned on distilled PubMed research and ADA 2026 revisions.
- **Cross-Platform**: Uses the `transformers` backend to run on standard CPUs and NVIDIA GPUs.
- **Dockerized**: Easy deployment to Hugging Face Spaces or private servers.

## 📥 Setup & Usage

### Local Execution (Python)
1. **Clone the repository**:
   ```bash
   git clone https://github.com/udaybhan10/DiabetesExpertGUI.git
   cd DiabetesExpertGUI
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**:
   ```bash
   python app.py
   ```

### Local Execution (Docker)
1. **Build the image**:
   ```bash
   docker build -t diabetes-expert-gui .
   ```

2. **Run the container**:
   ```bash
   docker run -p 7860:7860 diabetes-expert-gui
   ```

## ⚖️ Disclaimer
This model is for **educational and research purposes only**. It is not a medical device and should not be used for diagnosis or treatment decisions. Always consult a qualified healthcare professional.

## 📜 License
MIT License
