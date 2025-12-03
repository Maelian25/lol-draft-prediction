from app.backend.src import predictor


def test_model_shapes_mlp_and_transformer():
    """Check that models accept preprocessed input and return expected shapes."""

    # Build a dummy data object
    class DummyData:
        def __init__(self):
            # Try to pick 3rd red pick
            self.current_blue_picks = [
                "59",
                "61",
                "266",
                "PAD",
                "PAD",
            ]  # Jarvan, Orianna, Aatrox
            self.current_blue_bans = [
                "134",
                "141",
                "64",
                "PAD",
                "PAD",
            ]  # Syndra, Kayn, Lee Sin
            self.current_red_picks = [
                "112",
                "113",
                "PAD",
                "PAD",
                "PAD",
            ]  # Viktor, Sejuani
            self.current_red_bans = [
                "78",
                "236",
                "526",
                "PAD",
                "PAD",
            ]  # Poppy, Lucian, Rell
            self.blue_roles_available = [0, 0, 0, 1, 1]
            self.red_roles_available = [1, 0, 0, 1, 1]
            self.step = 11  # 11th pick (3rd red pick)
            self.next_phase = 1  # 1 is pick, 0 is ban
            self.next_side = 0  # 0 is red, 1 is blue

    data = DummyData()
    device = "cpu"

    tensors, _ = predictor.preprocess_input(data, device=device)

    # Instantiate MLP and Transformer with same args as predictor
    models = predictor.load_models(device=device)
    mlp = models[1]
    tr = models[2]

    # Run forward passes and assert shapes
    for model in (mlp, tr):
        try:
            out = predictor.predict_with_model(model, tensors)
        except Exception as e:
            raise AssertionError(f"Model forward failed: {e}")

        champ_logits, role_logits, wr_logits = out

        assert champ_logits.dim() == 2 and champ_logits.size(1) == 171 + 1  # +1 for PAD
        assert role_logits.dim() == 2 and role_logits.size(1) == 5
        # winrate logits should be (batch, 1) or (batch,)
        assert wr_logits.dim() in (1, 2)
