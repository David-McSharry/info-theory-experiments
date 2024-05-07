
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import tqdm
import wandb
from models import DecoupledSmileMIEstimator, DownwardSmileMIEstimator, GeneralSmileMIEstimator
from einops import reduce


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_feature_network(
    config,
    trainloader,
    feature_network_training,
    feature_network_A = None,
):
    """
    - Main function for training and evaluating the feature_network.
    - In training mode, the function will train either a model_A or a model_B.
    - In evaluation mode, the feature network will be frozen and adjusted_Psi will be calculated for the network.
    - If in eval_mode, no feature_network_A should be provided because Psi eval_mode is just for making sure
        the feature learned is emergent, and the MI with another feature_network is not relevant to this.

    - train_model_B mode is a subset of train_mode
    - Psi can optionally be replaced with adjusted_Psi in any mode
    """
    
    valid_dataset_types = ['bits', 'ecog']
    if config["dataset_type"] not in valid_dataset_types:
        raise ValueError(f"dataset_type must be one of {valid_dataset_types}")
    # make sure that if config['train_model_B'] is True, then feature_network_A is not None, and if False, then it is None, raise error otherwise
    if config['train_model_B']:
        if feature_network_A is None:
            raise ValueError("feature_network_A must be provided if training model B")
    else:
        if feature_network_A is not None:
            raise ValueError("feature_network_A must be None if not training model B")
    if not config['train_mode']: # if eval_mode => then should be no feature_network_A
        if feature_network_A is not None:
            raise ValueError("feature_network_A must be None if not training model B")

    wandb.init(project="bits-dataset-neurips", config=config)

    decoupled_MI_estimator = DecoupledSmileMIEstimator(
        feature_size=config['feature_size'],
        critic_output_size=config['critic_output_size'],
        hidden_sizes_1=config['decoupled_critic_hidden_sizes_1'],
        hidden_sizes_2=config['decoupled_critic_hidden_sizes_2'],
        clip=config['clip'],
        include_bias=config['bias']
        ).to(device)
    downward_MI_estimators = [
        DownwardSmileMIEstimator(
            feature_size=config['feature_size'],
            critic_output_size=config['critic_output_size'],
            hidden_sizes_v_critic=config['downward_hidden_sizes_v_critic'],
            hidden_sizes_xi_critic=config['downward_hidden_sizes_xi_critic'],
            clip=config['clip'],
            include_bias=config['bias']
            ).to(device) 
        for _ in range(config['num_atoms'])
    ]

    decoupled_optimizer = torch.optim.Adam(
        decoupled_MI_estimator.parameters(),
        lr=config["decoupled_critic_lr"],
        weight_decay=config["weight_decay"]
    )
    downward_optims = [
        torch.optim.Adam(
            dc.parameters(),
            lr=config["downward_lr"],
            weight_decay=config["weight_decay"]
        ) 
        for dc in downward_MI_estimators
    ]


    # init bit specific estimators
    if config["dataset_type"] == 'bits':
        xor_estimator = GeneralSmileMIEstimator(
            x_dim=config['feature_size'],
            y_dim=1,
            critic_output_size=config['critic_output_size'],
            x_critics_hidden_sizes=[512, 512, 128],
            y_critics_hidden_sizes=[512, 512, 128],
            clip=config['clip'],
            include_bias=config['bias']
        ).to(device)
        extra_bit_estimator = GeneralSmileMIEstimator(
            x_dim=config['feature_size'],
            y_dim=1,
            critic_output_size=config['critic_output_size'],
            x_critics_hidden_sizes=[512, 512, 128],
            y_critics_hidden_sizes=[512, 512, 128],
            clip=config['clip'],
            include_bias=config['bias']
        ).to(device)
        bonus_bit_estimator = GeneralSmileMIEstimator(
            x_dim=config['feature_size'],
            y_dim=1,
            critic_output_size=config['critic_output_size'],
            x_critics_hidden_sizes=[512, 512, 128],
            y_critics_hidden_sizes=[512, 512, 128],
            clip=config['clip'],
            include_bias=config['bias']
        ).to(device)
        extra_bit_optimizer = torch.optim.Adam(
            extra_bit_estimator.parameters(),
            lr=1e-4,
            weight_decay=config["weight_decay"]
        )

        bonus_bit_optimizer = torch.optim.Adam(
            bonus_bit_estimator.parameters(),
            lr=1e-4,
            weight_decay=config["weight_decay"]
        )
        xor_optimizer = torch.optim.Adam(
            xor_estimator.parameters(),
            lr=1e-4,
            weight_decay=config["weight_decay"]
        )


    # init feature network optimizer
    if config['train_mode']:
        feature_optimizer = torch.optim.Adam(
            feature_network_training.parameters(),
            lr=config["feature_lr"],
            weight_decay=config["weight_decay"]
        )
        # init train_model_B specific estimators
        if config['train_model_B']:
            MI_AB_estimator = GeneralSmileMIEstimator(
                x_dim=config['feature_size'],
                y_dim=config['feature_size'],
                critic_output_size=config['critic_output_size'],
                x_critics_hidden_sizes=[512, 512, 128],
                y_critics_hidden_sizes=[512, 512, 128],
                clip=config['clip'],
                include_bias=config['bias']
            ).to(device)
            MI_AB_optimizer = torch.optim.Adam(
                MI_AB_estimator.parameters(),
                lr=1e-4,
                weight_decay=config["weight_decay"]
            )

    # TODO: figure out why only f network is being watched, I would like to keep a closer eye on the grad n params.
    # TODO: Look at how GANs are trained with pytorch and make sure I'm not doing anything unreasonable.
    # Eg, https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py 
    # ^ this does not require retain_graph=True, so maybe this can be optomized somehow
    wandb.watch(feature_network_training, log='all')
    wandb.watch(decoupled_MI_estimator, log="all")
    for dc in downward_MI_estimators:
        wandb.watch(dc, log='all')

    ##
    ## TRAINING LOOP
    ##

    epochs = config['epochs']

    step = 0
    for _ in tqdm.tqdm(range(epochs), desc='Training'):
        for batch_num, batch in enumerate(trainloader):
            x0 = batch[:, 0].to(device).float()
            x1 = batch[:, 1].to(device).float()

            v0 = feature_network_training(x0).detach()
            v1 = feature_network_training(x1).detach()


            # update decoupled critic
            decoupled_optimizer.zero_grad()
            decoupled_MI = decoupled_MI_estimator(v0, v1)
            decoupled_loss = -decoupled_MI
            decoupled_loss.backward(retain_graph=True)
            decoupled_optimizer.step()


            # update each downward critic 
            for i in range(config['num_atoms']):
                downward_optims[i].zero_grad()
                channel_i = x0[:, i].unsqueeze(1).detach()
                downward_MI_i = downward_MI_estimators[i](v1, channel_i)
                downward_loss = - downward_MI_i
                downward_loss.backward(retain_graph=True)
                downward_optims[i].step()
                wandb.log({
                    f"downward_MI_{i}": downward_MI_i   
                }, step=step)


            # Find Psi (and maybe adjust it)
            downward_MIs_post_update = [] # used for calculating the adjusted Psi and Psi adjustment
            v0 = feature_network_training(x0)
            v1 = feature_network_training(x1)

            for i in range(config['num_atoms']): # finds sum of downward terms
                channel_i = x0[:, i].unsqueeze(1)
                channel_i_MI = downward_MI_estimators[i](v1, channel_i)
                downward_MIs_post_update.append(channel_i_MI)

            sum_downward_MI_post_update = sum(downward_MIs_post_update)
            decoupled_MI_post_update = decoupled_MI_estimator(v0, v1)

            if config["adjust_Psi"]:
                clipped_min_MIs = max(0, min(downward_MIs_post_update))
                Psi_adjustment = (config['num_atoms'] - 1) * clipped_min_MIs
                Psi = decoupled_MI_post_update - sum_downward_MI_post_update + Psi_adjustment
            else:
                Psi = decoupled_MI_post_update - sum_downward_MI_post_update

            if config['train_mode']: # if in training mode, update the network
                feature_optimizer.zero_grad()
                if config['train_model_B']: # if training a model_B, update MI_AB critic and feature_network loss
                    v0_A = feature_network_A(x0).detach()
                    # update MI_AB critic
                    MI_AB_optimizer.zero_grad()
                    MI_AB = MI_AB_estimator(v0, v0_A)
                    MI_AB_loss = - MI_AB
                    MI_AB_loss.backward(retain_graph=True)
                    MI_AB_optimizer.step()
                    MI_AB_post_update = MI_AB_estimator(v0, v0_A)
                    wandb.log({"MI_AB": MI_AB_post_update},step=step)
                    feature_loss = - Psi + MI_AB_post_update
                else:
                    feature_loss = - Psi

                if config['start_updating_f_after'] < step:
                    if batch_num % config['update_f_every_N_steps'] == 0:
                        feature_loss.backward(retain_graph=True)
                        feature_optimizer.step()

            wandb.log({
                "decoupled_MI": decoupled_MI_post_update,
                "sum_downward_MI": sum_downward_MI_post_update,
                "Psi": Psi,
            }, step=step)


            if config["dataset_type"] == 'bits':
                v0 = feature_network_training(x0).detach()
                v1 = feature_network_training(x1).detach()
                xor_bits = (reduce(x0[: , :5], 'b n -> b', 'sum') % 2).unsqueeze(1)
                extra_bit = x0[:, -1].unsqueeze(1)
                bonus_bit = ( xor_bits + extra_bit ) % 2


                xor_optimizer.zero_grad()
                xor_MI = xor_estimator(v0, xor_bits)
                xor_loss = -xor_MI
                xor_loss.backward(retain_graph=True)
                xor_optimizer.step()

                extra_bit_optimizer.zero_grad()
                extra_bit_MI = extra_bit_estimator(v0, extra_bit)
                extra_bit_loss = -extra_bit_MI
                extra_bit_loss.backward(retain_graph=True)
                extra_bit_optimizer.step()

                bonus_bit_optimizer.zero_grad()
                bonus_bit_MI = bonus_bit_estimator(v0, bonus_bit)
                bonus_bit_loss = -bonus_bit_MI
                bonus_bit_loss.backward(retain_graph=True)
                bonus_bit_optimizer.step()

                wandb.log({
                    "xor_MI": xor_MI,
                    "extra_bit_MI": extra_bit_MI,
                    "bonus_bit_MI": bonus_bit_MI
                }, step=step)

            step += 1
        
    torch.save(feature_network_training.state_dict(), f"models/feature_network_{wandb.run.name}.pth")
    
    return feature_network_training

