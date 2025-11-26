# Kilinda-Sauti Implementation Plan

## Phase 1: Data Preparation (Weeks 1-2)

### Audio Data Collection
- [ ] Collect 500+ hours of Kenyan-accented English speech
  - Radio broadcasts (Citizen TV, KBC, Capital FM)
  - Political speeches and press conferences
  - Podcast archives
- [ ] Collect 200+ hours of Swahili native speakers
  - News broadcasts in Swahili
  - Parliamentary proceedings
  - Community radio stations
- [ ] Generate 100+ hours of synthetic voice samples
  - Use ElevenLabs, Play.ht for voice cloning
  - Clone known Kenyan public figures (with permission)
  - Create adversarial examples

### Video Data Collection
- [ ] Collect 1000+ hours of authentic Kenyan political footage
  - TV news archives (2020-2024)
  - Government press briefings
  - Campaign rallies and public events
- [ ] Create synthetic video dataset
  - Generate 200+ hours using First Order Motion Model
  - Create face-swap examples of Kenyan politicians
  - Use Wav2Lip for lip-sync manipulation
- [ ] Compile media branding database
  - Extract logos from all major Kenyan TV stations
  - Document government seal variations
  - Create augmented versions for robustness

### Text Data Collection
- [ ] PolitiKweli dataset integration
  - 10,000+ annotated social media posts
  - Swahili-English code-switching examples
  - Labeled for propaganda, misinformation, incitement
- [ ] X/Twitter Kenya archive
  - Scrape political discourse (2022-2024)
  - Focus on election periods
  - Include fact-checked posts from Africa Check
- [ ] News article corpus
  - Collect from Daily Nation, The Standard, Citizen Digital
  - Include verified and debunked stories
  - Create ground truth labels

### Data Preprocessing
```bash
# Audio preprocessing
python scripts/preprocess_audio.py \
  --input_dir data/raw/audio \
  --output_dir data/processed/audio \
  --sample_rate 16000 \
  --normalize \
  --augment

# Video preprocessing
python scripts/preprocess_video.py \
  --input_dir data/raw/video \
  --output_dir data/processed/video \
  --resolution 720p \
  --fps 25 \
  --extract_audio

# Text preprocessing
python scripts/preprocess_text.py \
  --input_file data/raw/text/politikweli.csv \
  --output_dir data/processed/text \
  --language_detection \
  --clean_text
```

## Phase 2: Model Development (Weeks 3-6)

### Audio Module Development

**Week 3: Feature Engineering**
```python
# scripts/audio/extract_features.py
import librosa
import numpy as np

def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    
    # MFCC features (40 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    
    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    
    # Pitch features (YIN algorithm)
    f0 = librosa.yin(y, fmin=50, fmax=500)
    
    # Phoneme timing (using Kenyan accent model)
    phonemes = extract_phonemes_kenyan(y, sr)
    
    return {
        'mfcc': mfcc,
        'spectral': (spectral_centroids, spectral_rolloff),
        'pitch': f0,
        'phonemes': phonemes
    }
```

**Week 4: Model Training**
```python
# models/audio_classifier.py
import torch
import torch.nn as nn

class AudioDeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN for spectral features
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # LSTM for temporal patterns
        self.lstm = nn.LSTM(128, 256, num_layers=2, batch_first=True)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (batch, 1, freq, time)
        x = self.cnn(x)
        x = x.mean(dim=2)  # Global average pooling
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Last hidden state
        return self.classifier(x)

# Training script
python train_audio.py \
  --data_dir data/processed/audio \
  --model_config configs/audio_model.yaml \
  --epochs 50 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --gpu 0
```

### Visual Module Development

**Week 4: Temporal Feature Extraction**
```python
# models/video_classifier.py
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class VideoDeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # Spatial feature extractor
        self.spatial = EfficientNet.from_pretrained('efficientnet-b0')
        
        # Temporal convolutional network
        self.tcn = nn.Sequential(
            nn.Conv1d(1280, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Lip-sync analyzer
        self.lipsync = LipSyncAnalyzer()
        
        # Fusion and classification
        self.classifier = nn.Sequential(
            nn.Linear(256 + 64, 128),  # TCN + lip-sync features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, frames, audio):
        # Extract spatial features per frame
        spatial_features = []
        for frame in frames:
            feat = self.spatial.extract_features(frame)
            spatial_features.append(feat)
        
        # Temporal analysis
        temporal = torch.stack(spatial_features, dim=1)
        temporal = self.tcn(temporal.transpose(1, 2))
        
        # Lip-sync analysis
        lipsync_feat = self.lipsync(frames, audio)
        
        # Combine and classify
        combined = torch.cat([temporal.mean(dim=2), lipsync_feat], dim=1)
        return self.classifier(combined)
```

**Week 5: Branding Detection**
```python
# models/branding_detector.py
import cv2
import torch

class KenyanMediaBrandingDetector:
    def __init__(self):
        self.logo_templates = self.load_templates([
            'citizen_tv', 'kbc', 'ntv', 'govt_seal'
        ])
    
    def detect_tampering(self, frame):
        # Template matching for logos
        matches = []
        for template_name, template in self.logo_templates.items():
            result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
            if result.max() > 0.8:
                matches.append(template_name)
        
        # Check for compression artifacts near logos
        if matches:
            for match in matches:
                region = self.extract_logo_region(frame, match)
                artifacts = self.analyze_compression(region)
                if artifacts['suspicious']:
                    return True, f"Tampering detected in {match}"
        
        return False, "Authentic branding"
```

### Text Module Development

**Week 5-6: mBERT Fine-tuning**
```python
# models/text_classifier.py
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class MisinformationDetector:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.model = AutoModelForSequenceClassification.from_pretrained(
            'bert-base-multilingual-cased',
            num_labels=4  # propaganda, misinfo, incitement, manipulation
        )
    
    def preprocess(self, text):
        # Handle code-switching
        text = self.normalize_code_switching(text)
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        return inputs
    
    def predict(self, text):
        inputs = self.preprocess(text)
        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        
        return {
            'propaganda': probs[0][0].item(),
            'misinformation': probs[0][1].item(),
            'incitement': probs[0][2].item(),
            'manipulation': probs[0][3].item()
        }

# Training
python train_text.py \
  --data_file data/processed/text/politikweli_train.csv \
  --model_name bert-base-multilingual-cased \
  --epochs 10 \
  --batch_size 16 \
  --learning_rate 2e-5
```

## Phase 3: Fusion & API Development (Weeks 7-8)

### Fusion Core Implementation
```python
# models/fusion.py
import numpy as np
from sklearn.calibration import CalibratedClassifierCV

class FusionCore:
    def __init__(self):
        self.weights = {
            'audio': 0.35,
            'visual': 0.40,
            'text': 0.25
        }
        self.calibrator = self.load_calibrator()
    
    def fuse(self, audio_score, visual_score, text_score):
        # Weighted combination
        raw_score = (
            self.weights['audio'] * audio_score +
            self.weights['visual'] * visual_score +
            self.weights['text'] * text_score
        )
        
        # Temperature scaling calibration
        calibrated_score = self.calibrator.predict_proba([[raw_score]])[0][1]
        
        # Risk categorization
        risk_level = self.categorize_risk(calibrated_score)
        
        return {
            'overall_confidence': calibrated_score * 100,
            'risk_level': risk_level,
            'raw_scores': {
                'audio': audio_score,
                'visual': visual_score,
                'text': text_score
            }
        }
    
    def categorize_risk(self, score):
        if score > 0.8:
            return 'high'
        elif score > 0.6:
            return 'medium'
        else:
            return 'low'
```

### API Implementation
```python
# api/main.py
from fastapi import FastAPI, UploadFile, BackgroundTasks
from redis import Redis
import uuid

app = FastAPI()
redis_client = Redis(host='localhost', port=6379)

@app.post("/api/v1/analyze/video")
async def analyze_video(
    file: UploadFile,
    priority: str = "normal",
    background_tasks: BackgroundTasks = None
):
    # Generate job ID
    job_id = f"klnd_{uuid.uuid4().hex[:12]}"
    
    # Save file
    file_path = f"/tmp/{job_id}.mp4"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Queue for processing
    redis_client.lpush("video_queue", json.dumps({
        "job_id": job_id,
        "file_path": file_path,
        "priority": priority
    }))
    
    return {"job_id": job_id, "status": "queued"}

@app.get("/api/v1/results/{job_id}")
async def get_results(job_id: str):
    # Check Redis for results
    result = redis_client.get(f"result:{job_id}")
    if result:
        return json.loads(result)
    return {"status": "processing"}
```

## Phase 4: Dashboard & HITL (Week 9)

### Streamlit Dashboard
```python
# dashboard/app.py
import streamlit as st
import requests

st.set_page_config(page_title="Kilinda-Sauti", layout="wide")

st.title("🛡️ Kilinda-Sauti Detection System")

# Upload section
uploaded_file = st.file_uploader("Upload content for analysis", 
    type=['mp4', 'mp3', 'wav', 'txt'])

if uploaded_file and st.button("Analyze"):
    # Call API
    response = requests.post(
        "http://localhost:8000/api/v1/analyze/multimodal",
        files={"file": uploaded_file}
    )
    job_id = response.json()['job_id']
    
    # Poll for results
    with st.spinner("Analyzing..."):
        result = poll_results(job_id)
    
    # Display results
    st.metric("Overall Confidence", f"{result['overall_confidence']}%")
    st.metric("Risk Level", result['risk_level'])
    
    # Module breakdown
    col1, col2, col3 = st.columns(3)
    col1.metric("Audio", f"{result['modules']['audio']['confidence']}%")
    col2.metric("Visual", f"{result['modules']['visual']['confidence']}%")
    col3.metric("Text", f"{result['modules']['text']['confidence']}%")
    
    # HITL button
    if st.button("Send for Expert Review"):
        requests.post(f"/api/v1/hitl/submit", json={"job_id": job_id})
        st.success("Sent for human review")
```

## Phase 5: Testing & Optimization (Week 10)

### Performance Testing
```bash
# Load testing
locust -f tests/load_test.py --host=http://localhost:8000

# Accuracy evaluation
python evaluate.py \
  --test_set data/test \
  --models_dir models/checkpoints \
  --output_file results/evaluation.json
```

### Model Optimization
- [ ] Quantization (INT8) for faster inference
- [ ] ONNX export for cross-platform deployment
- [ ] TensorRT optimization for GPU inference
- [ ] Model pruning to reduce size

## Hackathon Roadmap (3 Days)

### Day 1: Core Infrastructure
- [x] Setup FastAPI backend with Redis queue
- [x] Implement basic file upload and storage
- [x] Create mock models (random predictions for demo)
- [x] Build basic Streamlit dashboard

### Day 2: Integration
- [x] Connect all three modules to API
- [x] Implement fusion logic
- [x] Add real-time feed scanner (mock X API)
- [x] Create results visualization

### Day 3: Polish & Demo
- [x] Add HITL workflow
- [x] Create demo video with test cases
- [x] Prepare presentation slides
- [x] Deploy to cloud (AWS/GCP)
- [x] Load testing and bug fixes

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Audio Accuracy | > 85% | Test set evaluation |
| Video Accuracy | > 80% | Test set evaluation |
| Text Accuracy | > 88% | Test set evaluation |
| Inference Time | < 2s | 95th percentile latency |
| False Positive Rate | < 15% | Manual review of flagged content |
| User Satisfaction | > 4/5 | Expert reviewer feedback |
