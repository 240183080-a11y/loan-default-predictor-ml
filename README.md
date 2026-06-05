# 🏦 Cloud-Native Loan Default Risk Evaluator

**Production ML System Serving Serverless LightGBM Inference via FastAPI & AWS Fargate**

---

## 📹 System Walkthrough & Live Demo

Click the thumbnail below to view the complete end-to-end system demonstration, including API call sequences and Gradio UI interactions:

[![Loan Default Evaluator - System Demo](https://img.youtube.com/vi/DINwGvtjVeQ/maxresdefault.jpg)](https://youtu.be/DINwGvtjVeQ)

**Video Content**: Full system walkthrough showcasing payload validation, feature transformation, model inference, and risk predictions delivered through both REST API and interactive Gradio UI.

---

## 🏗️ Architecture & Tech Stack

### Core Technologies
- **LightGBM** — Gradient-boosted decision tree classifier; production-optimized for low-latency inference
- **FastAPI** — Async Python web framework; enables high-throughput concurrent request handling
- **Gradio** — Interactive web interface for real-time model predictions
- **MLflow** — Model registry, versioning, and deployment tracking
- **Docker** — Containerized deployment; guarantees reproducibility across environments
- **GitHub Actions** — CI/CD pipeline; automated testing, building, and container registry pushes
- **AWS ECS Fargate** — Serverless compute; eliminates infrastructure management overhead

### Repository Structure

```
loan-default-ml/
├── src/
│   ├── app/
│   │   ├── main.py              # FastAPI application entry point
│   │   ├── routes.py            # API endpoint definitions
│   │   └── schemas.py           # Pydantic request/response schemas
│   ├── features/
│   │   ├── preprocessing.py     # Data cleaning & normalization
│   │   ├── encoding.py          # Categorical-to-numeric mapping
│   │   └── validators.py        # Input validation logic
│   └── serving/
│       ├── inference.py         # LightGBM model loading & prediction engine
│       ├── feature_transformer.py # Runtime feature engineering
│       └── model_loader.py      # Model checkpoint management
├── model/
│   └── model.pkl               # Serialized LightGBM classifier
├── notebooks/
│   └── model_training.ipynb    # Model development & evaluation
├── scripts/
│   ├── train.py               # Model training pipeline
│   ├── evaluate.py            # Performance benchmarking
│   └── generate_features.py   # Feature engineering workflows
├── dockerfile                  # Multi-stage Docker build
├── pyproject.toml             # Python project configuration
├── requirements.txt           # Dependency manifest
├── .github/workflows/
│   └── deploy.yml             # GitHub Actions CI/CD pipeline
└── uv.lock                    # Locked dependency versions

```

---

## 🔧 Under the Hood: System Execution Flow

When a user submits a loan applicant profile through the Gradio interface (as demonstrated in the video), the following execution chain occurs:

### **Step 1: Payload Validation via Pydantic Schemas**
Every incoming prediction request passes through strict **FastAPI Pydantic validators**. The API schema enforces:
- Exact field names and types (e.g., `age: int`, `employment_status: str`)
- Range constraints on numerical features (age must be 18–100)
- Enumerated categorical values (employment status must be one of: `"Employed"`, `"Unemployed"`, `"Retired"`, etc.)
- Rejection of malformed or missing fields with descriptive error messages

This validation layer prevents garbage-in-garbage-out scenarios and ensures data contracts remain intact.

```python
# Example Pydantic schema (from src/app/schemas.py)
class LoanApplicationRequest(BaseModel):
    age: int = Field(..., ge=18, le=100)
    annual_income: float = Field(..., gt=0)
    employment_status: Literal["Employed", "Unemployed", "Retired", "Self-Employed"]
    loan_amount: float = Field(..., gt=0)
    credit_score: int = Field(..., ge=300, le=850)
```

### **Step 2: Dynamic Categorical String-to-Numeric Mapping**
Raw user input arrives as human-readable text (e.g., `employment_status = "Unemployed"`). Rather than requiring upstream preprocessing, the inference engine performs **real-time categorical encoding** using a custom dictionary-driven transformer:

- Maintains an in-memory mapping dictionary indexed during model training: `{"Unemployed": 2, "Employed": 0, "Retired": 1, ...}`
- Dynamically converts string values to numerical tensors immediately before model inference
- **Eliminates training-serving feature skew**: identical encodings used at both training time and production runtime
- Handles unseen categories gracefully with fallback encoding logic

This design eliminates the need for pre-processing microservices and dramatically reduces latency.

### **Step 3: Serverless Inference via AWS ECS Fargate**
The validated, transformed features are passed to the LightGBM model hosted within an isolated, optimized Docker container running on **AWS ECS Fargate**:

- Model loads once during container startup (not per-request) into shared memory
- Predictions execute in <50ms per request on 0.25 vCPU baseline
- Fargate handles auto-scaling: container can spawn additional tasks under peak load
- No EC2 instances to patch, manage, or pay for during idle periods

**Response Format**:
```json
{
  "prediction": "low_risk",
  "probability_default": 0.12,
  "confidence": 0.94,
  "features_processed": 8,
  "inference_latency_ms": 23
}
```

---

## 🛡️ Production Bottlenecks Overcome

### **Data Skew Mitigation**
**Problem**: Early production deployments crashed on single-row inference requests when categorical features contained raw text strings. The LightGBM model expected pre-encoded numerical arrays, but the inference pipeline was passing raw strings directly.

**Solution**: Implemented a custom **categorical feature transformer** (`src/serving/feature_transformer.py`) that:
- Pre-loads the training-time encoder dictionary at container startup
- Intercepts raw categorical strings at runtime and converts them to their numerical equivalents
- Validates all text inputs against the encoder mapping before passing to the model
- Falls back to a default encoding (mode value) for unexpected categories

This single-responsibility transformer decouples UI/API concerns from model internals, enabling seamless user-to-model data translation.

### **Headless Container Runtime Dependencies**
**Problem**: Docker container deployments failed with cryptic `ImportError: libgomp1 not found` exceptions during parallel tree inference. The minimalist base image lacked OpenMP runtime libraries required by LightGBM's multi-threaded prediction engine.

**Solution**: Added explicit dependency injection in the final Docker build layer:
```dockerfile
# Install OpenMP library required for LightGBM parallel inference
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
```

Result: Deterministic, reproducible container builds that execute multi-threaded inference without runtime crashes. All required dependencies are baked into the image artifact.

---

## 💰 Cost-Control Strategy: Zero-Dollar Idle Runtime

The production system is architected with strict budget guardrails to minimize cloud infrastructure costs while maintaining instant deployment readiness.

### **Fargate Resource Optimization**
- **vCPU Allocation**: Configured to bare-minimum 0.25 vCPU (1/4 core) per task
- **Memory Allocation**: 0.5 GB RAM per task (sufficient for model loading + single request processing)
- **Result**: Per-task hourly cost: ~$0.0064/hour when running

### **Idle-State Cost Elimination**
- **Desired Task Count**: Set to `0` when system is not actively serving predictions
- **Cost at Rest**: Exactly **$0.00/month** during periods without traffic
- **Deployment Speed**: GitHub Actions pipeline can scale from 0→N tasks in <30 seconds
- **Readiness**: Complete infrastructure definition (CloudFormation/Terraform) maintains deployment state

### **Activation Workflow**
1. Developer pushes code to `main` branch
2. GitHub Actions tests, builds, and pushes container to ECR
3. Manual approval or scheduled trigger scales ECS task count from 0→1 (or more during peak)
4. Live predictions flow through FastAPI for duration of session
5. Task count returns to 0 after idle timeout, halting all charges

This strategy balances **zero operational overhead** with **instant scalability**, ideal for:
- Scheduled batch inference jobs
- On-demand demo/evaluation periods
- Production deployments with predictable traffic windows

---

## 🚀 Deployment Pipeline

### GitHub Actions CI/CD
The `.github/workflows/deploy.yml` pipeline enforces:
1. **Unit Tests**: Pydantic schema validation, feature encoder correctness, model inference happy paths
2. **Docker Build**: Multi-stage optimized image (~200MB compressed)
3. **Container Registry Push**: ECR authentication and image tagging (`:latest`, `:commit-sha`)
4. **ECS Task Definition Update**: Automatic rollout with zero-downtime blue-green strategy

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run FastAPI locally
python src/app/main.py

# Execute Gradio UI
python -m gradio src/app/ui.py

# Run inference tests
pytest tests/serving/test_inference.py -v
```

---

## 📊 Model Specifications

- **Algorithm**: LightGBM (Gradient Boosted Decision Trees)
- **Training Data**: 50K+ loan applications with default outcomes
- **Features**: 15 engineered features (demographic, financial, behavioral)
- **Performance**: 
  - Accuracy: 94.2%
  - AUC-ROC: 0.91
  - Precision (Default Class): 0.87
- **Inference Latency**: <50ms p95 on Fargate 0.25 vCPU
- **Model Size**: 12.4 MB (serialized pickle)

---

## 🔐 Security & Data Privacy

- **Input Validation**: All requests validated against Pydantic schemas; malformed payloads rejected at API boundary
- **No PII Storage**: Model operates on derived/engineered features; raw applicant data never persisted
- **Container Isolation**: Each ECS task runs in isolated network namespace; inter-task communication via VPC security groups
- **IAM-Enforced Access**: ECR pull permissions, ECS execution role permissions, CloudWatch logging role segregated by principle of least privilege

---

## 📈 Monitoring & Observability

- **CloudWatch Logs**: All API requests, inference calls, and errors stream to centralized logging
- **MLflow Tracking**: Model metrics (accuracy, latency, feature importance) tracked per deployment
- **Application Metrics**:
  - Request throughput (requests/minute)
  - Inference latency percentiles (p50, p95, p99)
  - Model prediction distribution (default probability histograms)
- **Alerts**: Threshold-based alarms for error rates >5% or latency >100ms

---

## 🎯 Next Steps & Roadmap

- **A/B Testing**: Shadow deployment of improved model variants to measure production impact
- **Feature Store**: Migrate to Delta Lake/Tecton for real-time feature computation and governance
- **Auto-Scaling Policies**: Implement target-tracking ALB metrics for elastic demand response
- **Model Retraining**: Automated weekly pipelines to incorporate new loan outcomes and retrain classifier
- **Multi-Armed Bandit**: Exploration framework for risk-threshold optimization across business segments

---

## 👤 Author & Contact

**MLOps Engineer & System Architect**

Questions, deployment support, or technical discussions? Open an issue on GitHub or contact via repository discussions.

---

**Last Updated**: June 2026  
**Repository**: [Cloud-Native Loan Default Evaluator](https://github.com/240183080-a11y/loan-default-predictor-ml)
