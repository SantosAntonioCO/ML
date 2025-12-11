# Tabnet more range

def build_space():
    space = {
        # Dimensões da rede (mais amplas e mais poderosas)
        "n_d": hp.choice("n_d", [0, 1, 2, 3]),    # [8, 16, 24, 32]
        "n_a": hp.choice("n_a", [0, 1, 2, 3]),    # [8, 16, 24, 32]

        # Mais steps (pesquisa real do modelo)
        "n_steps": hp.choice("n_steps", [0, 1, 2]),  # [3, 4, 5]

        # Attention relaxation
        "gamma": hp.uniform("gamma", 1.0, 2.5),   # maior espaço → captura mais relação

        # Regularização clássica TabNet
        "lambda_sparse": hp.loguniform("lambda_sparse", np.log(1e-6), np.log(1e-2)),

        # Aprendizado – espaço ampliado
        "lr": hp.loguniform("lr", np.log(3e-5), np.log(2e-3)),  # mais baixo → mais estável
        "weight_decay": hp.loguniform("weight_decay", np.log(1e-9), np.log(1e-4)),

        # Batch sizes mais variados (melhor resultado)
        "batch_size": hp.choice("batch_size", [0, 1, 2]),     # [512, 1024, 2048]
        "virtual_batch_size": hp.choice("virtual_batch_size", [0, 1, 2]),  # [32, 64, 128]

        # Epochs aumentados
        "max_epochs": hp.choice("max_epochs", [0, 1]),    # [50, 100]
        "patience": hp.choice("patience", [0, 1]),        # [12, 20]

        # workers
        "num_workers": hp.choice("num_workers", [0, 1, 2]),  # [0, 2, 4]

        # scale_pos_weight — calibrado para fraude
        "scale_pos_weight_choice": hp.choice(
            "scale_pos_weight_choice",
            [
                {"type": "fixed", "value": 1.0},
                {"type": "fixed", "value": 5.0},
                {"type": "fixed", "value": 10.0},
                {"type": "auto"},     # neg/pos limitado internamente
            ]
        )
    }
    return space


#####

CHOICE_MAP = {
    "n_d": [8, 16, 24, 32],
    "n_a": [8, 16, 24, 32],
    "n_steps": [3, 4, 5],
    "batch_size": [512, 1024, 2048],
    "virtual_batch_size": [32, 64, 128],
    "max_epochs": [50, 100],
    "patience": [12, 20],
    "num_workers": [0, 2, 4],
}
