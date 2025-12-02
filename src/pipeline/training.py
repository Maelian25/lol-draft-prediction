from src.ML_models.draft_MLP import DraftMLPModel
from src.ML_models.draft_transformer import DraftTransformer
from src.ML_training.trainer import TrainerClass
from src.ML_training.utils import preprocess_and_save
from src.utils.constants import MLP_CHECKPOINTS, TRANSFORMER_CHECKPOINTS
from src.utils.logger_config import get_logger

logger = get_logger("training", "training.log")


def run_training(
    matches_states, rebuild: bool = True, model_choice: str = "transformer"
):
    """Prepare models and run training pipelines.

    `model_choice` must be either 'transformer' or 'mlp'. Default is 'transformer'.
    """
    preprocess_and_save(matches_states, rebuild=rebuild)

    if model_choice == "mlp":
        mlp_model = DraftMLPModel(
            num_champions=171,
            num_roles=5,
            mode="learnable",
            embed_size=96,
            hidden_dim=1024,
            num_res_blocks=4,
            dropout=0.4,
        )

        mlp_trainer = TrainerClass(
            model=mlp_model,
            batch_size=512,
            num_epochs=20,
            base_lr=4e-4,
            weight_decay=4e-3,
            loss_weights_init=[-0.8, 0.0, 0.5],
            experiment_name="draft_MLP_v1",
            patience=3,
            save_dir=MLP_CHECKPOINTS,
        )

        mlp_trainer.train()
        return

    # default: transformer
    transformer_model = DraftTransformer(
        num_champions=171,
        num_roles=5,
        dim_feedforward=1024,
        nhead=8,
        d_model=256,
        num_layers=5,
        dropout=0.2,
    )

    transformer_trainer = TrainerClass(
        model=transformer_model,
        batch_size=512,
        num_epochs=40,
        base_lr=5e-5,
        weight_decay=1e-3,
        loss_weights_init=[0.0, 0.0, 0.0],
        grads_for_weights=False,
        experiment_name="draft_transformer_v2",
        patience=7,
        save_dir=TRANSFORMER_CHECKPOINTS,
        load_from_checkpoint=True,
    )

    transformer_trainer.train()
