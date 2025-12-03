import os

import boto3
from dotenv import load_dotenv

from src.utils.logger_config import get_logger

logger = get_logger(__name__, "draft_api.log")


def create_client_s3():
    """Create and return a Boto3 S3 client using
    credentials from environment variables."""
    # Load AWS credentials from .env file
    load_dotenv()
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    region_name = os.getenv("AWS_REGION")

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name,
    )
    return s3_client


def download_models_from_s3(
    model_types: list[str] = ["transformer", "mlp"],
    download_dir: str = "./app/backend/models",
):
    """Download model files from S3 to the specified local directory.

    Args:
        model_type: Name of the model type ('transformer' or 'mlp')
        download_dir: Local directory to save downloaded models
    """
    os.makedirs(download_dir, exist_ok=True)

    bucket_name = os.getenv("S3_BUCKET_NAME", "lol-bot-models")
    s3_client = create_client_s3()

    for model_type in model_types:
        model_filename = f"{model_type}_model.pth"
        local_path = os.path.join(download_dir, model_filename)
        try:
            logger.info(f"Downloading model {model_type} from S3...")
            s3_client.download_file(bucket_name, model_filename, local_path)
            logger.info(f"Downloaded model {model_type} from S3 to {local_path}")
        except Exception as e:
            logger.error(f"Failed to download model {model_type} from S3: {e}")
            raise
