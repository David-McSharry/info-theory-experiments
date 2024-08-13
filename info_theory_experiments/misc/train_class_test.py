
from info_theory_experiments.custom_datasets import BitStringDataset
import torch
from info_theory_experiments.models import SkipConnectionSupervenientFeatureNetwork
from info_theory_experiments.misc.trainer_class import EmergentFeatureTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for gamma in [0.99]:
    seed = 0
    torch.manual_seed(seed)
    bits_config_train = {
            "gamma_parity": gamma,
            "gamma_extra": gamma,
            "dataset_length": 1000000,
            "torch_seed": seed,
            "dataset_type": "bits",
            "num_atoms": 6,
            "batch_size": 1000,
            "train_mode": True,
            "adjust_Psi": False,
            "clip": 5,
            "feature_size": 1,
            "start_updating_f_after": 300,
            "update_f_every_N_steps": 5,
            "minimize_neg_terms_until": 0,
            "downward_critics_config": {
                "hidden_sizes_v_critic": [512, 512, 512, 256],
                "hidden_sizes_xi_critic": [512, 512, 512, 256],
                "critic_output_size": 32,
                "lr": 1e-3,
                "bias": True,
                "weight_decay": 0,
            },
            
            "decoupled_critic_config": {
                "hidden_sizes_encoder_1": [512, 512, 512],
                "hidden_sizes_encoder_2": [512, 512, 512],
                "critic_output_size": 32,
                "lr": 1e-3,
                "bias": True,
                "weight_decay": 0,
            },
            "feature_network_config": {
                "hidden_sizes": [256, 256],
                "lr": 1e-4,
                "bias": True,
                "weight_decay": 0.00001,
            }
    }

    bits_config_test = {
            "gamma_parity": gamma,
            "gamma_extra": gamma,
            "dataset_length": 1000000,
            "torch_seed": seed,
            "dataset_type": "bits",
            "num_atoms": 6,
            "batch_size": 1000,
            "adjust_Psi": True,
            "clip": 5,
            "feature_size": 1,
            "downward_critics_config": {
                "hidden_sizes_v_critic": [512, 512, 512, 256],
                "hidden_sizes_xi_critic": [512, 512, 512, 256],
                "critic_output_size": 32,
                "lr": 1e-3,
                "bias": True,
                "weight_decay": 0,
            },
            "decoupled_critic_config": {
                "hidden_sizes_encoder_1": [512, 512, 512],
                "hidden_sizes_encoder_2": [512, 512, 512],
                "critic_output_size": 32,
                "lr": 1e-3,
                "bias": True,
                "weight_decay": 0,
            },
            "feature_network_config": bits_config_train["feature_network_config"]
    }

    dataset = BitStringDataset(
        gamma_extra=bits_config_train["gamma_extra"],
        gamma_parity=bits_config_train["gamma_parity"],
        length=bits_config_train["dataset_length"],
    )

    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=bits_config_train["batch_size"], shuffle=True
    )

    skip_model = SkipConnectionSupervenientFeatureNetwork(
        num_atoms=bits_config_train['num_atoms'],
        feature_size=bits_config_train['feature_size'],
        hidden_sizes=bits_config_train['feature_network_config']['hidden_sizes'],
        include_bias=bits_config_train['feature_network_config']['bias'],
    ).to(device)

    trainer = EmergentFeatureTrainer(
        train_config=bits_config_train,
        test_config=bits_config_test,
        train_loader=trainloader,
        feature_network=skip_model,
        project_name="testing-refactored-code",
        path_where_to_save_model=None,
    )

    trainer.train(epochs=2)

    trainer.test(epochs=2)

    trainer.finish_run()


