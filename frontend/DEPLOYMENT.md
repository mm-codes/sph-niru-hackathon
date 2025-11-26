# Kilinda-Sauti Deployment Strategy

## Infrastructure Overview

### Cloud Provider Options

#### Option 1: AWS (Recommended)
**Pros**: Mature ML services, strong GPU support, global infrastructure
**Cost**: ~$2,500-$5,000/month for production

```yaml
# AWS Architecture
Services:
  - EC2 P3.2xlarge (V100 GPU): Model inference
  - ECS Fargate: API containers
  - RDS PostgreSQL: Result storage
  - ElastiCache Redis: Job queue
  - S3: Media file storage
  - CloudFront: CDN
  - Route 53: DNS management
  - Certificate Manager: SSL/TLS
```

#### Option 2: Google Cloud Platform
**Pros**: Strong AI/ML tooling, good pricing for startups
**Cost**: ~$2,000-$4,000/month

#### Option 3: Azure
**Pros**: Government contract-friendly, strong compliance
**Cost**: ~$2,500-$4,500/month

### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kilinda-sauti-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: kilinda-api
  template:
    metadata:
      labels:
        app: kilinda-api
    spec:
      containers:
      - name: api
        image: kilinda/api:v1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_HOST
          value: redis-service
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kilinda-sauti-worker
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: worker
        image: kilinda/worker:v1.0.0
        resources:
          requests:
            nvidia.com/gpu: 1
          limits:
            nvidia.com/gpu: 1
---
apiVersion: v1
kind: Service
metadata:
  name: kilinda-api
spec:
  selector:
    app: kilinda-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Docker Configuration

### API Container
```dockerfile
# Dockerfile.api
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY models/ ./models/

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Worker Container (GPU)
```dockerfile
# Dockerfile.worker
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3.10 python3-pip
COPY requirements-worker.txt .
RUN pip3 install --no-cache-dir -r requirements-worker.txt

# Copy models
COPY models/ ./models/
COPY src/workers/ ./src/workers/

CMD ["python3", "src/workers/inference_worker.py"]
```

## Database Schema

```sql
-- PostgreSQL schema
CREATE TABLE analysis_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id VARCHAR(50) UNIQUE NOT NULL,
    content_type VARCHAR(20) NOT NULL,
    overall_confidence DECIMAL(5,2),
    risk_level VARCHAR(10),
    audio_score DECIMAL(5,2),
    visual_score DECIMAL(5,2),
    text_score DECIMAL(5,2),
    findings JSONB,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE hitl_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id VARCHAR(50) REFERENCES analysis_results(job_id),
    assigned_to VARCHAR(100),
    status VARCHAR(20) DEFAULT 'pending',
    expert_decision VARCHAR(20),
    expert_notes TEXT,
    submitted_at TIMESTAMP DEFAULT NOW(),
    reviewed_at TIMESTAMP
);

CREATE TABLE social_media_scans (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    platform VARCHAR(20) NOT NULL,
    post_id VARCHAR(100) UNIQUE,
    content_url TEXT,
    scan_result UUID REFERENCES analysis_results(id),
    flagged BOOLEAN DEFAULT FALSE,
    scanned_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_results_job_id ON analysis_results(job_id);
CREATE INDEX idx_results_created_at ON analysis_results(created_at DESC);
CREATE INDEX idx_hitl_status ON hitl_queue(status);
CREATE INDEX idx_scans_platform ON social_media_scans(platform, scanned_at DESC);
```

## CI/CD Pipeline

### GitHub Actions Workflow
```yaml
# .github/workflows/deploy.yml
name: Deploy Kilinda-Sauti

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    - name: Run tests
      run: pytest tests/ --cov=src/
    - name: Upload coverage
      uses: codecov/codecov-action@v3

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Build API image
      run: docker build -f Dockerfile.api -t kilinda/api:${{ github.sha }} .
    - name: Build worker image
      run: docker build -f Dockerfile.worker -t kilinda/worker:${{ github.sha }} .
    - name: Push to registry
      run: |
        echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
        docker push kilinda/api:${{ github.sha }}
        docker push kilinda/worker:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/kilinda-api api=kilinda/api:${{ github.sha }}
        kubectl set image deployment/kilinda-worker worker=kilinda/worker:${{ github.sha }}
        kubectl rollout status deployment/kilinda-api
```

## Monitoring & Observability

### Prometheus Metrics
```python
# src/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
requests_total = Counter(
    'kilinda_requests_total',
    'Total requests',
    ['method', 'endpoint', 'status']
)

inference_duration = Histogram(
    'kilinda_inference_duration_seconds',
    'Inference duration',
    ['model_type']
)

queue_depth = Gauge(
    'kilinda_queue_depth',
    'Current queue depth',
    ['priority']
)

# Detection metrics
detections_total = Counter(
    'kilinda_detections_total',
    'Total detections',
    ['risk_level', 'content_type']
)
```

### Grafana Dashboard
```json
{
  "dashboard": {
    "title": "Kilinda-Sauti Operations",
    "panels": [
      {
        "title": "Requests per Second",
        "targets": [{
          "expr": "rate(kilinda_requests_total[5m])"
        }]
      },
      {
        "title": "Inference Latency (p95)",
        "targets": [{
          "expr": "histogram_quantile(0.95, kilinda_inference_duration_seconds)"
        }]
      },
      {
        "title": "Queue Depth",
        "targets": [{
          "expr": "kilinda_queue_depth"
        }]
      },
      {
        "title": "Detection Rate by Risk",
        "targets": [{
          "expr": "kilinda_detections_total"
        }]
      }
    ]
  }
}
```

### Logging Strategy
```python
# src/logging_config.py
import logging
import json

class StructuredLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
    
    def log_analysis(self, job_id, content_type, result):
        self.logger.info(json.dumps({
            "event": "analysis_complete",
            "job_id": job_id,
            "content_type": content_type,
            "confidence": result['overall_confidence'],
            "risk_level": result['risk_level'],
            "processing_time": result['metadata']['processing_time']
        }))
    
    def log_error(self, job_id, error, traceback):
        self.logger.error(json.dumps({
            "event": "analysis_error",
            "job_id": job_id,
            "error": str(error),
            "traceback": traceback
        }))
```

## Security Configuration

### API Authentication
```python
# src/auth.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(
            credentials.credentials,
            SECRET_KEY,
            algorithms=["HS256"]
        )
        return payload
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
```

### Network Security
```yaml
# Security groups (AWS)
Ingress:
  - Port 443 (HTTPS): 0.0.0.0/0
  - Port 80 (HTTP): 0.0.0.0/0 (redirect to 443)
  - Port 22 (SSH): VPN IP range only
  
Egress:
  - All traffic: 0.0.0.0/0

# WAF rules
Rules:
  - Rate limiting: 100 req/min per IP
  - SQL injection protection
  - XSS protection
  - Bot detection
```

## Backup & Disaster Recovery

### Database Backups
```bash
# Daily automated backups
0 2 * * * pg_dump kilinda_db | gzip > /backups/kilinda_$(date +\%Y\%m\%d).sql.gz

# Retention policy: 30 days
find /backups -name "kilinda_*.sql.gz" -mtime +30 -delete
```

### Model Versioning
```bash
# Store model checkpoints in S3
aws s3 sync models/ s3://kilinda-models/versions/$(date +%Y%m%d)/

# Rollback procedure
aws s3 sync s3://kilinda-models/versions/20241115/ models/
kubectl rollout restart deployment/kilinda-worker
```

## Scaling Strategy

### Horizontal Pod Autoscaler
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: kilinda-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: kilinda-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: requests_per_second
      target:
        type: AverageValue
        averageValue: "1000"
```

### GPU Worker Scaling
```python
# scripts/scale_workers.py
import boto3

def scale_gpu_workers(queue_depth):
    """Scale GPU workers based on queue depth"""
    if queue_depth > 100:
        desired_count = 5
    elif queue_depth > 50:
        desired_count = 3
    else:
        desired_count = 2
    
    ecs = boto3.client('ecs')
    ecs.update_service(
        cluster='kilinda-cluster',
        service='kilinda-worker',
        desiredCount=desired_count
    )
```

## Cost Optimization

### Resource Allocation
```yaml
# Development environment
API: 2 replicas × t3.medium ($0.04/hr) = $60/month
Workers: 1 × g4dn.xlarge ($0.52/hr) = $380/month
Database: db.t3.small ($0.03/hr) = $22/month
Total: ~$500/month

# Production environment
API: 3 replicas × t3.large ($0.08/hr) = $180/month
Workers: 2 × g4dn.2xlarge ($1.04/hr) = $1,500/month
Database: db.r5.large ($0.24/hr) = $175/month
Redis: cache.r5.large ($0.22/hr) = $160/month
Storage: 2TB S3 = $46/month
Data transfer: ~$200/month
Total: ~$2,500/month
```

### Spot Instance Strategy
```python
# Use spot instances for batch processing
spot_config = {
    'InstanceType': 'g4dn.xlarge',
    'MaxPrice': '0.35',  # 33% discount
    'SpotFleetRequestConfig': {
        'AllocationStrategy': 'lowestPrice',
        'IamFleetRole': 'arn:aws:iam::account:role/spot-fleet',
        'TargetCapacity': 2
    }
}
```

## Launch Checklist

### Pre-Launch
- [ ] Load testing completed (1000 req/min sustained)
- [ ] Security audit passed
- [ ] Backup procedures tested
- [ ] Monitoring dashboards configured
- [ ] API documentation published
- [ ] User training materials created
- [ ] Incident response plan documented

### Launch Day
- [ ] Deploy to production
- [ ] Verify all services healthy
- [ ] Test end-to-end workflows
- [ ] Monitor error rates
- [ ] Prepare rollback plan

### Post-Launch
- [ ] Monitor performance for 48 hours
- [ ] Gather user feedback
- [ ] Address critical bugs
- [ ] Plan first optimization cycle
