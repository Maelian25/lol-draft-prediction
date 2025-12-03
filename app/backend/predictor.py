from typing import Dict, Any, Tuple
import os
import re
import torch

from app.backend.utils import download_models_from_s3
from src.ML_models.draft_MLP import DraftMLPModel
from src.ML_models.draft_transformer import DraftTransformer
from src.ML_training.utils import expand_mask
from src.utils.champions_helper import champ_id_to_idx_map
from src.utils.logger_config import get_logger

# Initialize module logger
logger = get_logger(__name__, "predictor.log")


def get_best_checkpoint(folder: str, prefix: str) -> str | None:
    """
    Get the best checkpoint file from the specified folder, local development version.
    """
    try:
        files = [
            f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith(".pth")
        ]
    except Exception:
        logger.exception("Unable to list checkpoint folder: %s", folder)
        return None

    best_file = None
    best_score = -1
    for f in files:
        m = re.search(r"(\d+)\.pth$", f)
        if m:
            score = int(m.group(1))
            if score > best_score:
                best_score = score
                best_file = f

    return os.path.join(folder, best_file) if best_file else None


def get_checkpoint(
    model_type: str = "transformer",
    folder: str = "./app/backend/models",
) -> str:
    if os.path.exists(os.path.join(folder, f"{model_type}_model.pth")):
        return os.path.join(folder, f"{model_type}_model.pth")
    else:
        download_models_from_s3([model_type], folder)
        return os.path.join(folder, f"{model_type}_model.pth")


def load_models(device: str = "cpu") -> Dict[int, torch.nn.Module]:
    models: Dict[int, torch.nn.Module] = {}

    logger.info("Loading models on device: %s", device)

    # MLP
    mlp_ckpt = get_checkpoint("mlp", "./app/backend/models")
    model_mlp = DraftMLPModel(
        num_champions=171,
        num_roles=5,
        mode="learnable",
        embed_size=96,
        hidden_dim=1024,
        num_res_blocks=4,
        dropout=0.4,
    )
    if mlp_ckpt:
        ck = torch.load(mlp_ckpt, map_location=device)
        if "model_state_dict" in ck:
            model_mlp.load_state_dict(ck["model_state_dict"])
        logger.info("MLP checkpoint loaded from %s", mlp_ckpt)
    else:
        logger.info(
            "No MLP checkpoint found in %s; using init weights",
            "data/MLP_checkpoints",
        )
    model_mlp.to(device)
    model_mlp.eval()
    models[1] = model_mlp
    logger.info("MLP model ready and registered under id 1")

    # Transformer
    tr_ckpt = get_checkpoint("transformer", "./app/backend/models")
    model_tr = DraftTransformer(
        num_champions=171,
        num_roles=5,
        dim_feedforward=1024,
        nhead=8,
        d_model=256,
        num_layers=5,
        dropout=0.2,
    )
    if tr_ckpt:
        ck = torch.load(tr_ckpt, map_location=device)
        if "model_state_dict" in ck:
            model_tr.load_state_dict(ck["model_state_dict"])
        logger.info("Transformer checkpoint loaded from %s", tr_ckpt)
    else:
        logger.info(
            "No Transformer checkpoint found in %s; using init weights",
            "data/TRANSFORMER_checkpoints",
        )
    model_tr.to(device)
    model_tr.eval()
    models[2] = model_tr
    logger.info("Transformer model ready and registered under id 2")

    return models


def preprocess_input(
    data: Any, device: str, num_champions: int = 171
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """Convert incoming JSON-like data (champion IDs) into model tensors.

    Returns a tuple (tensors_dict, id2idx_map)
    """
    id2idx = champ_id_to_idx_map()
    pad_idx = id2idx.get("PAD", num_champions)

    def _to_idx_list(lst):
        res = []
        for cid in lst:
            idx = id2idx.get(str(cid))
            if idx is None:
                idx = pad_idx
                logger.debug(
                    "Unknown champion id %s encountered during preprocessing; "
                    "using PAD index %s",
                    cid,
                    pad_idx,
                )
            res.append(int(idx))
        return res

    bp_idx = _to_idx_list(data.current_blue_picks)
    bb_idx = _to_idx_list(data.current_blue_bans)
    rp_idx = _to_idx_list(data.current_red_picks)
    rb_idx = _to_idx_list(data.current_red_bans)

    batch_size = 1

    tensors = {}
    tensors["bp"] = torch.tensor([bp_idx], dtype=torch.long, device=device)
    tensors["bb"] = torch.tensor([bb_idx], dtype=torch.long, device=device)
    tensors["rp"] = torch.tensor([rp_idx], dtype=torch.long, device=device)
    tensors["rb"] = torch.tensor([rb_idx], dtype=torch.long, device=device)

    # champ_mask: ones with zeros for used champions
    champ_availability = torch.ones(
        (batch_size, num_champions), dtype=torch.long, device=device
    )
    used = set()
    for lst in (bp_idx, bb_idx, rp_idx, rb_idx):
        for v in lst:
            try:
                vi = int(v)
            except Exception:
                continue
            if 0 <= vi < num_champions:
                used.add(vi)
    for ui in used:
        champ_availability[0, ui] = 0

    champ_availability = expand_mask(champ_availability)
    tensors["champ_availability"] = champ_availability

    # Role masks: accept provided lists (should be length=num_roles)
    tensors["blue_roles_available"] = torch.tensor(
        [data.blue_roles_available], dtype=torch.long, device=device
    )
    tensors["red_roles_available"] = torch.tensor(
        [data.red_roles_available], dtype=torch.long, device=device
    )

    # X placeholder (unused for learnable embedding mode)
    tensors["X"] = torch.zeros((batch_size, 10), dtype=torch.float32, device=device)

    tensors["step"] = torch.tensor([data.step], dtype=torch.long, device=device)
    tensors["next_side"] = torch.tensor(
        [data.next_side], dtype=torch.long, device=device
    ).unsqueeze(1)
    tensors["next_phase"] = torch.tensor(
        [data.next_phase], dtype=torch.long, device=device
    )

    return tensors, id2idx


def postprocess_output(
    champ_logits: torch.Tensor,
    role_logits: torch.Tensor,
    wr_logits: torch.Tensor,
    id2idx: Dict[str, int],
) -> Dict[str, Any]:
    # top champion indices
    topk = 5
    vals = torch.topk(champ_logits, topk).indices.squeeze().tolist()
    if isinstance(vals, int):
        vals = [vals]

    # inverse mapping
    idx2id = {v: int(k) for k, v in id2idx.items() if k != "PAD"}

    top_champs = [idx2id.get(int(i)) for i in vals]

    # roles
    top_roles = torch.topk(role_logits, topk).indices.squeeze().tolist()
    if isinstance(top_roles, int):
        top_roles = [top_roles]

    # winrate
    winrate = torch.sigmoid(wr_logits).squeeze().tolist()
    logger.debug(
        "Postprocess result -> top_champs: %s | top_roles: %s | winrate: %s",
        top_champs,
        top_roles,
        winrate,
    )

    return {
        "top_champions": top_champs,
        "top_roles": top_roles,
        "winrate": winrate,
    }


def predict_with_model(
    model: torch.nn.Module, tensors: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logger.debug("Running inference with model %s", model.__class__.__name__)
    outputs = model(
        tensors["X"],
        tensors["bp"],
        tensors["rp"],
        tensors["bb"],
        tensors["rb"],
        tensors["champ_availability"],
        tensors["blue_roles_available"],
        tensors["red_roles_available"],
        tensors["step"],
        tensors["next_side"],
        tensors["next_phase"],
    )
    logger.debug("Inference completed for model %s", model.__class__.__name__)
    return outputs
