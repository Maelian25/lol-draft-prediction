from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel

from src.ML_models.draft_MLP import DraftMLPModel
from src.ML_models.draft_transformer import DraftTransformer


# Models available
MODELS = {1: DraftMLPModel, 2: DraftTransformer}


class DraftInput(BaseModel):
    model_id: int
    current_picks: list[int]
    current_bans: list[int]
    current_step: int
    current_phase: int
    current_side: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Server startup")
    # Load available models to cache it and use it on the fly

    yield
    print("Server closed")


app = FastAPI(title="Draft predictor API", lifespan=lifespan)


@app.get("/models")
def model_list():
    return [model.__name__ for _, model in MODELS.items()]


@app.post("/predict")
async def predict_draft(data: DraftInput):
    return
