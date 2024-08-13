from info_theory_experiments.custom_datasets import GameOfLifeDatasetNoLoop
import torch
from info_theory_experiments.trainers_MSE_loss_minimization import train_feature_network
from info_theory_experiments.models import GameOfLifeCNN


device = 'cuda'

seed=0
torch.manual_seed(seed)

config = {
    "torch_seed": seed,
    "dataset_type": "gol",
    "grid_size": 15,
    "num_atoms": 225,
    "batch_size": 1000,
    "train_model_B": False,
    "adjust_Psi": False,
    "clip": 5,
    "feature_size": 1,
    "epochs": 10,
    "start_updating_f_after": 150,
    "update_f_every_N_steps": 5,
    "minimize_neg_terms_until": 0,
    "mse_coeff": -1,
    "decoupled_critic_config": {
        "hidden_sizes_encoder_1": [512, 512, 512],
        "hidden_sizes_encoder_2": [512, 512, 512],
        "critic_output_size": 32,
        "lr": 1e-3,
        "bias": True,
        "weight_decay": 0,
    },
    "MSE_predictor_config": {
        "lr": 1e-3,
        "weight_decay": 1e-6,
    },
    "feature_network_config": {
        "hidden_sizes": [256, 256, 256, 256, 256],
        "lr": 1e-4,
        "bias": True,
        "weight_decay": 1e-6,
    }
}

dataset = GameOfLifeDatasetNoLoop(
    prepare = True,
    normalize=True,
    num_simulations=1000,
    time_steps=100,
    grid_size=config['grid_size'],
)

print("dataset size:", dataset.data.size())

trainloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

skip_model = GameOfLifeCNN(
    output_dim=config['feature_size'],
).to(device)

project_name = "GOL-emergence-MSE-minimization"

skip_model, name = train_feature_network(
    config=config,
    trainloader=trainloader,
    emergenct_feature_encoder=skip_model,
    project_name=project_name,
    model_dir_prefix=project_name
)



    # config_test = {    
    #     "torch_seed": seed,
    #     "dataset_type": "gol",
    #     "num_atoms": 225,
    #     "batch_size": 1000,
    #     "train_mode": False,
    #     "train_model_B": False,
    #     "adjust_Psi": True,
    #     "clip": 5,
    #     "feature_size": 1,
    #     "epochs": 3,
    #     "start_updating_f_after": 300,
    #     "update_f_every_N_steps": 5,
    #     "minimize_neg_terms_until": 0,
    #     "downward_critics_config": {
    #         "hidden_sizes_v_critic": [512, 1024, 1024, 512],
    #         "hidden_sizes_xi_critic": [512, 512, 512],
    #         "critic_output_size": 32,
    #         "lr": 1e-3,
    #         "bias": True,
    #         "weight_decay": 0,
    #     },
        
    #     "decoupled_critic_config": {
    #         "hidden_sizes_encoder_1": [512, 512, 512],
    #         "hidden_sizes_encoder_2": [512, 512, 512],
    #         "critic_output_size": 32,
    #         "lr": 1e-3,
    #         "bias": True,
    #         "weight_decay": 0,
    #     },
    #     "feature_network_config": {
    #         "hidden_sizes": [256, 256, 256, 256, 256],
    #         "lr": 1e-4,
    #         "bias": True,
    #         "weight_decay": 1e-6,
    #     },
    #     "training-name": name
    # }

    # project_name_test = project_name + "-verification"

    # skil_model, _ = train_feature_network(
    #         config=config_test,
    #         trainloader=trainloader,
    #         feature_network_training=skip_model,
    #         project_name=project_name_test,
    #         model_dir_prefix=None
    # )




