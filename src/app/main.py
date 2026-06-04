from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
from src.serving.inference import predict  # Core ML inference engine block

# Initialize FastAPI application
app = FastAPI(
    title="Loan Default Prediction API",
    description="Production-grade ML API for calculating customer loan default probabilities",
    version="1.0.0"
)

@app.get("/")
def root():
    """Health check route for orchestrator validation."""
    return {"status": "ok"}

# === REQUEST DATA SCHEMA ===
class LoanApplicantData(BaseModel):
    """
    Pydantic model that validates features matching your exact categorical lists.
    """
    # Categorical fields
    Education: str          # "High School", "Bachelor's", "Master's", "PhD"
    EmploymentType: str     # "Full-time", "Part-time", "Self-employed", "Unemployed"
    MaritalStatus: str      # "Single", "Married", "Divorced"
    HasMortgage: str        # "Yes" or "No"
    HasDependents: str      # "Yes" or "No"
    LoanPurpose: str        # "Home", "Auto", "Business", "Education", "Other"
    HasCoSigner: str        # "Yes" or "No"
    
    # Numeric continuous fields (Adjust these names to match your precise columns!)
    Age: int
    Income: float
    LoanAmount: float
    CreditScore: int
    MonthsInJob: int
    InterestRate: float

# === MAIN PREDICTION API ENDPOINT ===
@app.post("/predict")
def get_prediction(data: LoanApplicantData):
    """
    Validates incoming request payload schemas before feeding to LightGBM.
    """
    try:
        # Note: dict() is deprecated in newer Pydantic versions; model_dump() is preferred
        payload = data.model_dump() if hasattr(data, "model_dump") else data.dict()
        result = predict(payload)
        return {"prediction": result}
    except Exception as e:
        return {"error": str(e)}

# === GRADIO INTERFACE LOGIC ===
def gradio_interface(
    Education, EmploymentType, MaritalStatus, HasMortgage, HasDependents,
    LoanPurpose, HasCoSigner, Age, Income, LoanAmount, CreditScore, MonthsInJob, InterestRate
):
    # Construct schema matching LoanApplicantData requirements
    data = {
        "Education": Education,
        "EmploymentType": EmploymentType,
        "MaritalStatus": MaritalStatus,
        "HasMortgage": HasMortgage,
        "HasDependents": HasDependents,
        "LoanPurpose": LoanPurpose,
        "HasCoSigner": HasCoSigner,
        "Age": int(Age),
        "Income": float(Income),
        "LoanAmount": float(LoanAmount),
        "CreditScore": int(CreditScore),
        "MonthsInJob": int(MonthsInJob),
        "InterestRate": float(InterestRate)
    }
    
    result = predict(data)
    return str(result)

# === GRADIO UI CONFIGURATION ===
demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        # Categorical Inputs
        gr.Dropdown(["High School", "Bachelor's", "Master's", "PhD"], label="Education Level", value="Bachelor's"),
        gr.Dropdown(["Full-time", "Part-time", "Self-employed", "Unemployed"], label="Employment Status", value="Full-time"),
        gr.Dropdown(["Single", "Married", "Divorced"], label="Marital Status", value="Single"),
        gr.Dropdown(["Yes", "No"], label="Has Mortgage?", value="No"),
        gr.Dropdown(["Yes", "No"], label="Has Dependents?", value="No"),
        gr.Dropdown(["Home", "Auto", "Business", "Education", "Other"], label="Loan Purpose", value="Home"),
        gr.Dropdown(["Yes", "No"], label="Has Co-Signer?", value="No"),
        
        # Numeric Inputs
        gr.Number(label="Age", value=30, minimum=18, maximum=100),
        gr.Number(label="Annual Income ($)", value=50000.0, minimum=0),
        gr.Number(label="Requested Loan Amount ($)", value=15000.0, minimum=0),
        gr.Number(label="Credit Score", value=650, minimum=300, maximum=850),
        gr.Number(label="Months in Current Job", value=24, minimum=0),
        gr.Number(label="Interest Rate (%)", value=7.5, minimum=0, maximum=100)
    ],
    outputs=gr.Textbox(label="Risk Assessment Assessment", lines=2),
    title="🏦 Loan Default Risk Evaluator",
    description="""
    **Predict credit delinquency and loan application defaults instantly using LightGBM.**
    
    Fill out the financial assessment details above to gauge performance variables. 
    The core prediction leverages a production-grade classifier backend evaluated over historical default parameters.
    """,
    examples=[
        # High Risk Example (Unemployed, low credit score, no co-signer)
        ["High School", "Unemployed", "Single", "Yes", "No", "Other", "No", 25, 20000.0, 30000.0, 510, 3, 14.5],
        # Low Risk Example (Highly educated, high income, strong credit)
        ["Master's", "Full-time", "Married", "No", "Yes", "Home", "Yes", 42, 110000.0, 25000.0, 780, 72, 5.2]
    ],
    theme=gr.themes.Soft()
)

# === MOUNT GRADIO UI INTO FASTAPI ===
app = gr.mount_gradio_app(app, demo, path="/ui")