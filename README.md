# 🔍 Nexus Vision - Smart Navigation Assistant

![Nexus Vision](https://img.shields.io/badge/Nexus%20Vision-AI%20Navigation-blue?style=for-the-badge&logo=eye&logoColor=white)

**AI-Powered Smart Navigation Assistant for Visually Impaired Users**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nexus-vision-app.streamlit.app)  
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)  
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)](https://ultralytics.com/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🌟 Overview

Nexus Vision is an advanced AI-powered navigation assistance system designed for visually impaired users. Powered by YOLOv8 and Streamlit, it provides:

- 🔍 **Real-time object detection** with 22 specialized navigation classes  
- 🔊 **Audio navigation alerts** via text-to-speech  
- 🚨 **Priority-based warnings** (Critical / Warning / Safe)  
- ♿ **Accessibility-first UI** optimized for ease of use  

---

## 🚀 Live Demo

[Try Nexus Vision on Streamlit Cloud »](https://nexus-vision-app.streamlit.app)

---

## 🎯 Specialized Classes

**🚨 Critical Hazards**  
stairs • curb • car • bus • truck • bicycle  

**⚠️ Common Objects**  
person • stop_sign • traffic_light • bench • fire_hydrant • pole  

**📍 Contextual & Other**  
bus_stop • tree • crutch • dog • motorcycle • spherical_roadblock • train • warning_column • waste_container  

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit with custom CSS  
- **AI/ML:** YOLOv8 (Ultralytics)  
- **Vision:** OpenCV for image handling  
- **Audio:** Google Text-to-Speech (gTTS)  
- **Deployment:** Streamlit Cloud  
- **Language:** Python 3.8+  

---

## 📦 Installation

```
git clone https://github.com/yourusername/nexus-vision-streamlit.git
cd nexus-vision-streamlit
pip install -r requirements.txt
# Place your trained model as best.pt or use pretrained fallback
streamlit run streamlit_app.py
```

---

## 🎮 Usage

1. 🎥 Allow camera access and click **Take Picture**  
2. 🤖 Wait for AI to analyze and annotate  
3. 🔊 Listen to audio alerts (Critical/Warning/Safe)  
4. 🚶‍♂️ Navigate safely based on guidance  

---

## 🔧 Configuration

- Place custom model as `best.pt` (fallback to `yolov8n.pt`)  
- Adjust confidence threshold in `safe_detect()` (default: 0.25)  
- Change TTS language in `generate_audio_alert()`  

---

## 🤝 Contributing

Contributions are welcome!  
1. Fork & branch  
2. Commit & push  
3. Open a PR  

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
