# Kilinda-Sauti: AI Misinformation Detection System

**Multi-modal deepfake and misinformation detection for Kenya**

## Overview

Kilinda-Sauti (Swahili: "Guardian Voice") is a comprehensive AI system designed to detect synthetic media, manipulated content, and propaganda in the Kenyan context. The system combines three specialized detection modules with a fusion core to provide real-time threat assessment.

## Key Features

- 🎤 **Audio Detection**: Voice clone identification using spectral analysis
- 🎥 **Video Detection**: Deepfake detection with Temporal Convolutional Networks
- 📝 **Text Detection**: Misinformation and propaganda detection in Swahili-English
- 🔄 **Fusion Core**: Unified confidence scoring across all modalities
- 📊 **Real-time Dashboard**: Live threat monitoring and analysis interface
- 👥 **Human-in-the-Loop**: Expert review workflow for high-stakes content
- 🌐 **Social Media Scanner**: X/TikTok feed monitoring

## System Architecture

```
Frontend (React + Tailwind)
       ↓
  API Gateway (FastAPI)
       ↓
  ┌────┴────┬────────┬────────┐
  │         │        │        │
Audio     Visual    Text    Fusion
Module    Module   Module    Core
  │         │        │        │
  └─────────┴────────┴────────┘
              ↓
       Result Storage
       + HITL Queue
```

## Tech Stack

### Frontend
- React + TypeScript
- Tailwind CSS (dark theme)
- Shadcn UI components
- Lucide icons

### Backend (Planned)
- FastAPI (Python)
- PyTorch for model inference
- Redis for job queue
- PostgreSQL for results
- S3 for media storage

### ML Models
- **Audio**: WaveNet-based spectrogram classifier
- **Visual**: TCN + EfficientNet
- **Text**: mBERT fine-tuned on PolitiKweli

## Quick Start

### Frontend Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

### Environment Setup

```bash
# Required environment variables
VITE_API_URL=http://localhost:8000
VITE_ENABLE_DEMO_MODE=true
```

## Documentation

- [**ARCHITECTURE.md**](./ARCHITECTURE.md) - Detailed system design and module specifications
- [**IMPLEMENTATION_PLAN.md**](./IMPLEMENTATION_PLAN.md) - Development roadmap and training scripts
- [**DEPLOYMENT.md**](./DEPLOYMENT.md) - Infrastructure, CI/CD, and scaling strategy
- [**ETHICS_SAFEGUARDS.md**](./ETHICS_SAFEGUARDS.md) - Privacy, bias mitigation, and governance

## Project Structure

```
kilinda-sauti/
├── src/
│   ├── components/
│   │   └── dashboard/
│   │       ├── DashboardHeader.tsx
│   │       ├── StatsOverview.tsx
│   │       ├── ThreatFeed.tsx
│   │       ├── AnalysisPanel.tsx
│   │       └── AnalysisResults.tsx
│   ├── pages/
│   │   └── Index.tsx
│   └── index.css
├── docs/
│   ├── ARCHITECTURE.md
│   ├── IMPLEMENTATION_PLAN.md
│   ├── DEPLOYMENT.md
│   └── ETHICS_SAFEGUARDS.md
└── README.md
```

## Training Data Requirements

### Audio Module
- 500+ hours Kenyan-accented English
- 200+ hours Swahili native speakers
- 100+ hours synthetic voice samples

### Visual Module
- 1000+ hours authentic Kenyan political footage
- 200+ hours synthetic video (face-swap, lip-sync)
- Media branding database (TV stations, govt)

### Text Module
- PolitiKweli dataset (10,000+ posts)
- X/Twitter Kenya archive
- Fact-checked articles from Africa Check

## Performance Targets

| Metric | Target |
|--------|--------|
| Audio Accuracy | > 85% |
| Video Accuracy | > 80% |
| Text Accuracy | > 88% |
| Overall Latency | < 2 seconds |
| False Positive Rate | < 15% |

## Ethical Safeguards

- ✅ Human-in-the-loop review for high-risk content
- ✅ Privacy-first design (no user tracking)
- ✅ Bias testing across demographics
- ✅ Transparent quarterly reporting
- ✅ Independent oversight board
- ✅ User appeals process

See [ETHICS_SAFEGUARDS.md](./ETHICS_SAFEGUARDS.md) for details.

## Deployment Strategy

### Development
- Cost: ~$500/month
- 1 GPU worker (g4dn.xlarge)
- Small database and cache

### Production
- Cost: ~$2,500/month
- 2+ GPU workers with auto-scaling
- High-availability database
- CDN for global access

See [DEPLOYMENT.md](./DEPLOYMENT.md) for full infrastructure specs.

## Hackathon Roadmap (3 Days)

### Day 1: Core Infrastructure
- [x] Frontend dashboard with mock data
- [x] API structure and endpoints
- [x] Basic model integration

### Day 2: Integration
- [ ] Connect all three modules
- [ ] Implement fusion logic
- [ ] Add social media scanner
- [ ] Results visualization

### Day 3: Polish & Demo
- [ ] HITL workflow
- [ ] Demo video with test cases
- [ ] Presentation materials
- [ ] Deploy to cloud

## API Endpoints (Planned)

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

## Contributing

We welcome contributions from:
- ML researchers (model improvements)
- Data scientists (bias testing)
- Frontend developers (UI/UX)
- Security experts (vulnerability testing)
- Civil society (ethics review)

Please read our ethics guidelines before contributing.

## License

[To be determined - likely GPL or AGPL for transparency]

## Contact

- **Technical Lead**: [Your Name]
- **Ethics Officer**: [Name]
- **Data Protection**: [Name]
- **General Inquiries**: info@kilinda-sauti.ke

## Acknowledgments

- PolitiKweli dataset creators
- Africa Check and PesaCheck for fact-checking expertise
- Kenyan media houses for data partnerships
- Civil society organizations for oversight

---

**⚠️ Important Notice**: This system is designed to detect manipulated media, not to determine truth or censor speech. All high-risk detections undergo human expert review. The system operates with transparency and respects press freedom and civil liberties.
