# smart-meeting-notetaker

## Description
NoteFlow is a fully offline edge AI meeting note-taker built on Snapdragon X Elite. 
It records meeting audio, transcribes it using Whisper ONNX via QNN EP on the NPU, 
and generates structured notes using Phi-3-mini. Everything runs on-device with zero 
cloud dependency.

## Team
| Name | Email |
|------|-------|
| Manthan Limbachiya | 25022776@studentmail.ul.ie |
| Anshul Keng | 25028367@studentmail.ul.ie |
| Paridhi Bisht | 25068857@studentmail.ul.ie |
| Govind Pathak | 25020749@studentmail.ul.ie |

## Setup Instructions

### Prerequisites
- Python 3.11
- Windows OS

### Installation
1. Clone the repository:
   git clone https://github.com/mrjuniorlimbachiya/smart-meeting-notetaker.git
   cd smart-meeting-notetaker

2. Install dependencies:
   python -m pip install flask flask-cors onnxruntime onnxruntime-qnn onnxruntime-genai soundfile torch openai-whisper scipy numpy

3. Download Whisper ONNX models from Qualcomm AI Hub and place in /models:
   - WhisperEncoder.onnx
   - WhisperDecoder.onnx
   - mel_filters.npz

4. Download Phi-3-mini ONNX from https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx
   Place in: backend/phi3_model/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/
   
## Run Instructions
python backend/app.py
Browser opens automatically at http://localhost:5000

## Usage
1. Click Record and speak your meeting
2. Click Stop when done
3. Click Whisper NPU to transcribe
4. Click Phi-3 Insight to generate structured notes
5. Export as .txt or .md
   
## Notes
- Requires Snapdragon X Elite or X Plus for NPU acceleration via QNN EP
- Falls back to CPU on non-Snapdragon hardware automatically
- All processing is fully offline — no data leaves the device
- Phi-3 model loading takes approximately 30 seconds on first run
  
## References
- Qualcomm AI Hub: https://aihub.qualcomm.com
- Simple Whisper Transcription: https://github.com/thatrandomfrenchdude/simple-whisper-transcription
- Simple NPU Chatbot: https://github.com/thatrandomfrenchdude/simple_npu_chatbot
- ONNX Runtime QNN EP: https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html
- Phi-3 ONNX: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx
