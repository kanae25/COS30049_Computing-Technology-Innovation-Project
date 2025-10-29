from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)

class PredictResponse(BaseModel):
    label: str
    probability: float
    top_tokens: list[tuple[str, float]]
    vector_size: int
    model: str = "multinomial_nb"

class FeedbackRequest(BaseModel):
    was_correct: bool
    label: str
