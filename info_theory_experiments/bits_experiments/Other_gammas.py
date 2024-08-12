
from custom_datasets import BitStringDataset
import torch
from models import SkipConnectionSupervenientFeatureNetwork
from trainers import train_feature_network

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for gamma in [0.5, 0.6, 0.7, 0.8, 0.9]:
    seed = 2
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
            "train_model_B": False,
            "adjust_Psi": False,
            "clip": 5,
            "feature_size": 1,
            "epochs": 5,
            "start_updating_f_after": 100,
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

    project_name_train = "NEURIPS-other-gammas-for-bits-" + str(gamma)

    skip_model, name = train_feature_network(
            config=bits_config_train,
            trainloader=trainloader,
            feature_network_training=skip_model,
            project_name=project_name_train,
            model_dir_prefix=project_name_train
    )

    bits_config_test = {
            "gamma_parity": gamma,
            "gamma_extra": gamma,
            "dataset_length": 1000000,
            "torch_seed": seed,
            "dataset_type": "bits",
            "num_atoms": 6,
            "batch_size": 1000,
            "train_mode": False,
            "train_model_B": False,
            "adjust_Psi": False,
            "clip": 5,
            "feature_size": 1,
            "epochs": 2,
            "start_updating_f_after": 1000,
            "update_f_every_N_steps": 5,
            "minimize_neg_terms_until": 0,
            "downward_critics_config": {
                "hidden_sizes_v_critic": [256, 256, 256],
                "hidden_sizes_xi_critic": [256, 256, 256],
                "critic_output_size": 32,
                "lr": 1e-3,
                "bias": True,
                "weight_decay": 0,
            },
            
            "decoupled_critic_config": {
                "hidden_sizes_encoder_1": [256, 256],
                "hidden_sizes_encoder_2": [256, 256],
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
            },
        "name": name
    }

    project_name_test = project_name_train + "-validation"

    skil_model, _ = train_feature_network(
            config=bits_config_test,
            trainloader=trainloader,
            feature_network_training=skip_model,
            project_name=project_name_test,
            model_dir_prefix=None
    )



