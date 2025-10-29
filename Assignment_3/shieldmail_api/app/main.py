from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import PredictRequest, PredictResponse, FeedbackRequest
from app.model import MODEL

app = FastAPI(title="ShieldMail API", version="1.0.0")

# CORS: adjust for your Vite dev port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def _load_model():
    MODEL.load_or_bootstrap()

@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": MODEL.pipeline is not None, "version": "1.0.0"}

@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text is required and must be 1..5000 chars")
    if len(text) > 5000:
        raise HTTPException(status_code=422, detail="text too long")
    try:
        label, prob, top, vsize = MODEL.predict(text)
        return PredictResponse(label=label, probability=prob, top_tokens=top, vector_size=vsize)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inference_error: {e}")

@app.get("/api/tokens")
def tokens(limit: int = 15):
    # Static showcase for front-end (replace with real global stats if you have them)
    return {"tokens": [["win", 0.5], ["free", 0.48], ["urgent", 0.44], ["click", 0.42]][:limit]}

@app.post("/api/feedback")
def feedback(req: FeedbackRequest):
    # Intentionally not storing raw text—just acknowledge
    return {"status": "recorded", "was_correct": req.was_correct, "label": req.label}
