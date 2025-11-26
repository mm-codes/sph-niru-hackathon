# Kilinda-Sauti System Architecture

## Executive Summary

Kilinda-Sauti is a multi-modal AI system designed to detect localized misinformation and deepfakes in the Kenyan context. The system integrates three specialized detection modules with a fusion core to provide comprehensive threat assessment.

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend Dashboard                        │
│     (Real-time monitoring + Analysis interface)              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                      API Gateway                             │
│           (Load balancing + Authentication)                  │
└────┬───────────────┬────────────────┬───────────────────────┘
     │               │                │
     ▼               ▼                ▼
┌──────────┐  ┌──────────┐    ┌──────────┐
│  Audio   │  │  Visual  │    │   Text   │
│  Module  │  │  Module  │    │  Module  │
└────┬─────┘  └────┬─────┘    └────┬─────┘
     │             │               │
     └─────────────┼───────────────┘
                   ▼
           ┌──────────────┐
           │ Fusion Core  │
           └──────┬───────┘
                  │
                  ▼
         ┌────────────────┐
         │  Result Store  │
         │  + HITL Queue  │
         └────────────────┘
```

## Module Architecture

### 1. Audio Module (Voice Clone Detection)

**Purpose**: Detect AI-generated voices in Kenyan-accented English and Swahili speech

**Technical Stack**:
- Model: WaveNet-based spectrogram classifier
- Input: 16kHz audio (mono)
- Features: MFCCs, spectral centroids, pitch contours, phoneme timing

**Detection Pipeline**:
```
Audio Input → Preprocessing → Feature Extraction → Neural Network → Confidence Score
   ↓              ↓                   ↓                  ↓              ↓
16kHz WAV    Normalize/          MFCC (40 dim)      CNN+LSTM      0-100% synthetic
             Denoise            Spectral flux       ResNet18       likelihood
```

**Key Components**:
- Spectral Analysis: FFT-based frequency domain analysis
- Pitch Detection: YIN algorithm for fundamental frequency
- Phoneme Timing: Kenyan English + Swahili phonetic models
- Accent Verification: Trained on local speaker profiles

**Training Data Requirements**:
- 500+ hours Kenyan-accented speech
- 200+ hours Swahili native speakers
- 100+ hours synthetic voice samples (ElevenLabs, Play.ht, etc.)
- Radio broadcasts from Citizen TV, KBC, Capital FM

### 2. Visual Module (Deepfake Detection)

**Purpose**: Detect manipulated video content in Kenyan media

**Technical Stack**:
- Model: Temporal Convolutional Network (TCN) + EfficientNet
- Input: 720p video, 25fps
- Features: Facial landmarks, temporal inconsistencies, metadata analysis

**Detection Pipeline**:
```
Video Input → Frame Extraction → Face Detection → TCN Analysis → Confidence Score
   ↓               ↓                   ↓              ↓              ↓
MP4/MOV      Every 5 frames      MediaPipe        3D Conv       0-100% fake
             (5fps sample)       FaceMesh         Layers         likelihood
```

**Detection Techniques**:
1. **Lip-Sync Analysis**: Audio-visual correspondence checking
2. **Temporal Consistency**: Frame-to-frame coherence analysis
3. **Compression Artifacts**: JPEG/H.264 artifact patterns
4. **Branding Verification**: Logo detection for Kenyan media (Citizen TV watermarks, government seals)

**Training Data Requirements**:
- 1000+ hours authentic Kenyan political footage
- 200+ hours synthetic video (face-swap, lip-sync)
- Media branding database (TV stations, government)
- FaceForensics++ adapted for Kenyan faces

### 3. Text Module (Misinformation Detection)

**Purpose**: Detect propaganda, false narratives, and incitement in Swahili-English text

**Technical Stack**:
- Model: mBERT fine-tuned on PolitiKweli dataset
- Input: UTF-8 text (Swahili + English code-switching)
- Features: Named entities, sentiment, propaganda patterns

**Detection Pipeline**:
```
Text Input → Preprocessing → mBERT Encoding → Classification → Confidence Score
   ↓             ↓                  ↓              ↓              ↓
Raw text    Language ID      768-dim vectors   Multi-label    0-100% risk
            Tokenization     Contextual repr.   classifier     per category
```

**Detection Categories**:
1. **Propaganda**: Loaded language, flag-waving, emotional appeals
2. **Misinformation**: False claims, conspiracy theories
3. **Incitement**: Hate speech, call to violence
4. **Manipulation**: Cherry-picking, whataboutism

**Training Data Requirements**:
- PolitiKweli dataset (10,000+ annotated posts)
- X/Twitter Kenya archive (political discourse)
- Fact-checked articles from Africa Check, PesaCheck
- Code-switching corpus (Sheng, Swahili-English mix)

## Fusion Core

**Purpose**: Combine module outputs into unified threat assessment

**Algorithm**: Weighted ensemble with confidence calibration

```python
fusion_score = (
    w_audio * audio_confidence +
    w_visual * visual_confidence +
    w_text * text_confidence
) * calibration_factor

# Weights learned from validation set
w_audio = 0.35
w_visual = 0.40
w_text = 0.25
```

**Calibration**:
- Temperature scaling on validation set
- Conformal prediction for uncertainty quantification
- Kenyan-specific threshold tuning

## API Design

### Endpoints

```
POST /api/v1/analyze/audio
POST /api/v1/analyze/video
POST /api/v1/analyze/text
POST /api/v1/analyze/multimodal

GET  /api/v1/results/{job_id}
POST /api/v1/hitl/submit
GET  /api/v1/feed/realtime

POST /api/v1/scan/social-media
```

### Example Request

```json
POST /api/v1/analyze/video
Content-Type: multipart/form-data

{
  "video": "<binary>",
  "priority": "high",
  "callback_url": "https://webhook.example.com/results"
}
```

### Example Response

```json
{
  "job_id": "klnd_2024_001234",
  "status": "completed",
  "timestamp": "2024-11-25T10:30:00Z",
  "results": {
    "overall_confidence": 87,
    "risk_level": "high",
    "modules": {
      "audio": {
        "confidence": 92,
        "findings": ["Synthetic voice detected", "Pitch anomalies at 2.3s"]
      },
      "visual": {
        "confidence": 84,
        "findings": ["Lip-sync errors", "Frame inconsistencies"]
      },
      "text": {
        "confidence": 85,
        "findings": ["Propaganda language detected"]
      }
    },
    "metadata": {
      "processing_time": "1.8s",
      "model_versions": {
        "audio": "v1.2.3",
        "visual": "v2.0.1",
        "text": "v1.5.0"
      }
    }
  }
}
```

## Infrastructure Requirements

### Compute
- GPU: 4x NVIDIA A100 (40GB) or equivalent
- CPU: 64 cores for preprocessing
- RAM: 256GB minimum
- Storage: 2TB NVMe SSD

### Deployment
- Kubernetes cluster (3 nodes minimum)
- Redis for job queue
- PostgreSQL for result storage
- S3-compatible object storage for media files

### Scaling Strategy
- Horizontal scaling for API gateway
- GPU auto-scaling based on queue depth
- CDN for static assets and results

## Security & Privacy

### Data Protection
- End-to-end encryption for uploads
- Automatic content deletion after 30 days
- Role-based access control (RBAC)
- Audit logging for all operations

### Model Security
- Model versioning and rollback
- Adversarial robustness testing
- Regular security audits
- Watermarking of analyzed content

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Audio analysis | < 0.5s | TBD |
| Video analysis | < 2.0s | TBD |
| Text analysis | < 0.2s | TBD |
| Overall latency | < 2.0s | TBD |
| Throughput | 100 req/min | TBD |
| Accuracy | > 85% | TBD |
