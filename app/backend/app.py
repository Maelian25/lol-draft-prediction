"""
FastAPI application for League of Legends draft prediction.

This module provides REST API endpoints for:
- Listing available prediction models
- Making draft predictions using loaded models

Models are loaded at startup and cached for efficient prediction requests.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
import torch

from app.backend import predictor
from src.utils.logger_config import get_logger

# Initialize logger for the application
logger = get_logger(__name__, "draft_api.log")

# Model instances cache (populated at startup and reused for all predictions)
model_instances: dict[int, torch.nn.Module] = {}


class DraftInput(BaseModel):
    """Input model for draft prediction requests.

    Attributes:
        model_id: ID of the model to use for prediction (1=MLP, 2=Transformer)
        current_blue_picks: List of champion IDs picked by blue team
        current_blue_bans: List of champion IDs banned by blue team
        current_red_picks: List of champion IDs picked by red team
        current_red_bans: List of champion IDs banned by red team
        blue_roles_available: List of available role IDs for blue team
        red_roles_available: List of available role IDs for red team
        step: Current step in the draft (0-19)
        next_phase: Next phase happening (0=pick, 1=ban)
        next_side: Next side taking action (0=blue, 1=red)
    """

    model_id: int
    current_blue_picks: list[str]
    current_blue_bans: list[str]
    current_red_picks: list[str]
    current_red_bans: list[str]
    blue_roles_available: list[int]
    red_roles_available: list[int]
    step: int
    next_phase: int
    next_side: int


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    """Manage the application lifecycle: load models at startup, cleanup at shutdown.

    On startup:
    - Detects available GPU device (CUDA or CPU)
    - Loads all pre-trained models from disk
    - Caches models in memory for fast inference

    On shutdown:
    - Releases resources and performs cleanup
    """
    # Startup: Load models into cache
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    models = predictor.load_models(device=device)
    model_instances.update(models)
    logger.info(f"Successfully loaded {len(model_instances)} model(s) into cache")

    yield  # Server runs here

    # Shutdown: Cleanup
    logger.info("Server shutdown initiated. Releasing resources.")


fastapi_app = FastAPI(title="Draft predictor API", lifespan=lifespan)


@fastapi_app.get("/models")
def model_list():
    """Retrieve the list of available models.

    Returns:
        list[str]: List of model class names currently loaded in the cache.
    """
    model_names = [m.__class__.__name__ for m in model_instances.values()]
    logger.debug(f"Models list requested. Available models: {model_names}")
    return model_names


@fastapi_app.post("/predict")
async def predict_draft(data: DraftInput):
    """Generate draft predictions based on the current game state.

    This endpoint performs the following steps:
    1. Retrieves the requested model from cache
    2. Preprocesses input data into model tensors
    3. Runs model inference
    4. Postprocesses output to human-readable format

    Args:
        data: DraftInput containing current game state information

    Returns:
        dict: Prediction results containing champion, role, and win rate predictions,
              or error dict if model not found or prediction fails.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Retrieve the requested model from cache
    model = model_instances.get(data.model_id)
    if model is None:
        logger.warning(
            f"Prediction requested with non-existent model ID: {data.model_id}"
        )
        return {"error": "Model not found"}

    logger.debug(
        f"Processing prediction request with model ID {data.model_id} "
        f"on device {device}"
    )

    with torch.no_grad():
        # Preprocess input data into model tensors
        tensors, id2idx = predictor.preprocess_input(data, device=device)

        try:
            # Run model inference
            champ_logits, role_logits, wr_logits = predictor.predict_with_model(
                model, tensors
            )
            logger.debug("Model inference completed successfully")
        except Exception as e:
            logger.error(f"Model prediction failed: {str(e)}", exc_info=True)
            return {"error": f"Model prediction failed: {str(e)}"}

        # Postprocess model outputs to human-readable format
        result = predictor.postprocess_output(
            champ_logits, role_logits, wr_logits, id2idx
        )
        logger.info("Prediction request completed successfully")
        return result
