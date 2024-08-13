
import torch
import torch.nn as nn
import tqdm
import wandb
from info_theory_experiments.models import DecoupledSmileMIEstimator, DownwardSmileMIEstimator, GeneralSmileMIEstimator
from einops import reduce

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# NOTE: To be honest this was a waste of time. I did not need any of the utility methods, and
# it complicated things for no reason. The code to use it isn't even that much nicer.
# From now on, if I want to do something with a class I'm drawing a diagram and planning things
# out ahead of time. Iterating on class structures without a plan is absolutely ASS.

class EmergentFeatureTrainer:
    def __init__(
        self,
        train_config: dict,
        test_config: dict,
        train_loader: torch.utils.data.DataLoader,
        feature_network: nn.Module,
        project_name: str,
        path_where_to_save_model: str
    ):
        """
        We will start off only using this class to train
        model A's. 
        
        In order to get the extra tracking for bits
        dataset, create a subclass and add these bits or something.

        When this class is finished a run, it should be possible to 
        continue training as usual.

        For now, if you swtich training mode the estimators will be 
        written over.
        """
        self.train_config = train_config
        self.test_config = test_config
        self.train_loader = train_loader
        self.feature_network = feature_network
        self.project_name = project_name
        self.path_where_to_save_model = path_where_to_save_model
        self.train_step = 0
        self.test_step = 0
        self.train_epochs = 0
        self.test_epochs = 0

        self.train_run_id = None
        self.test_run_id = None

    def _init_wandb_train(self):
        wandb.init(project=self.project_name, config=self.train_config)

    def _init_wandb_test(self):
        wandb.init(project=self.project_name+"-validation", config=self.test_config)
    
    def init_estimators(
        self,
        train_mode: bool
    ):
        if train_mode:
            config = self.train_config
        else:
            config = self.test_config
            
        self.decoupled_MI_estimator = DecoupledSmileMIEstimator(
            feature_size=config['feature_size'],
            critic_output_size=config['decoupled_critic_config']['critic_output_size'],
            hidden_sizes_1=config['decoupled_critic_config']['hidden_sizes_encoder_1'],
            hidden_sizes_2=config['decoupled_critic_config']['hidden_sizes_encoder_2'],
            clip=config['clip'],
            include_bias=config['decoupled_critic_config']['bias']
        ).to(device)

        self.downward_MI_estimators = [
            DownwardSmileMIEstimator(
                feature_size=config['feature_size'],
                critic_output_size=config['downward_critics_config']['critic_output_size'],
                hidden_sizes_v_critic=config['downward_critics_config']['hidden_sizes_v_critic'],
                hidden_sizes_xi_critic=config['downward_critics_config']['hidden_sizes_xi_critic'],
                clip=config['clip'],
                include_bias=config['downward_critics_config']['bias']
            ).to(device) 
            for _ in range(config['num_atoms'])
        ]

        self.decoupled_optimizer = torch.optim.Adam(
            self.decoupled_MI_estimator.parameters(),
            lr=config['decoupled_critic_config']["lr"],
            weight_decay=config['decoupled_critic_config']["weight_decay"]
        )
        self.downward_optims = [
            torch.optim.Adam(
                dc.parameters(),
                lr=config['downward_critics_config']["lr"],
                weight_decay=config['downward_critics_config']["weight_decay"]
            ) 
            for dc in self.downward_MI_estimators
        ]
        if train_mode:
            self.feature_optimizer = torch.optim.Adam(
                self.feature_network.parameters(),
                lr=config['feature_network_config']["lr"],
                weight_decay=config['feature_network_config']["weight_decay"]
            )
        else:
            self.feature_network.eval()

    def train(
        self,
        epochs: int,
    ):
        self.train_epochs += epochs

        if self.train_run_id is None:
            self._init_wandb_train()
            self.init_estimators(train_mode=True)
            self.train_run_id = wandb.run.id
        else:
            wandb.init(project=self.project_name, id=self.train_run_id, resume="must")

        wandb.config.update({'epochs': self.train_epochs}, allow_val_change=True)

        for _ in tqdm.tqdm(range(epochs), desc='Training'):
            for batch_num, batch in enumerate(self.train_loader):
                x0 = batch[:, 0].to(device).float()
                x1 = batch[:, 1].to(device).float()
                v0_B = self.feature_network(x0).detach()
                v1_B = self.feature_network(x1).detach()

                # update decoupled critic
                self.decoupled_optimizer.zero_grad()
                decoupled_MI = self.decoupled_MI_estimator(v0_B, v1_B)
                decoupled_loss = -decoupled_MI
                decoupled_loss.backward(retain_graph=True)
                self.decoupled_optimizer.step()


                # update each downward critic 
                for i in range(self.train_config['num_atoms']):
                    self.downward_optims[i].zero_grad()
                    channel_i = x0[:, i].unsqueeze(1).detach()
                    downward_MI_i = self.downward_MI_estimators[i](v1_B, channel_i)
                    downward_loss = - downward_MI_i
                    downward_loss.backward(retain_graph=True)
                    self.downward_optims[i].step()
                    wandb.log({
                        f"downward_MI_{i}": downward_MI_i   
                    }, step=self.train_step)
    
                self.feature_optimizer.zero_grad()

                downward_MIs_post_update = [] # used for calculating the adjusted Psi and Psi adjustment
                v0_B = self.feature_network(x0)
                v1_B = self.feature_network(x1)

                for i in range(self.train_config['num_atoms']): # finds sum of downward terms
                    channel_i = x0[:, i].unsqueeze(1)
                    channel_i_MI = self.downward_MI_estimators[i](v1_B, channel_i)
                    downward_MIs_post_update.append(channel_i_MI)

                sum_downward_MI_post_update = sum(downward_MIs_post_update)
                decoupled_MI_post_update = self.decoupled_MI_estimator(v0_B, v1_B)

                if self.train_config["adjust_Psi"]:
                    clipped_min_MIs = max(0, min(downward_MIs_post_update))
                    Psi_adjustment = (self.train_config['num_atoms'] - 1) * clipped_min_MIs
                    Psi = decoupled_MI_post_update - sum_downward_MI_post_update + Psi_adjustment
                else:
                    Psi = decoupled_MI_post_update - sum_downward_MI_post_update

                # train mode specific
                if self.train_config['minimize_neg_terms_until'] < self.train_step:
                    feature_loss = - Psi
                else:
                    feature_loss = - (-sum_downward_MI_post_update)

                if self.train_config['start_updating_f_after'] < self.train_step:
                    if batch_num % self.train_config['update_f_every_N_steps'] == 0:
                        feature_loss.backward(retain_graph=True)
                        self.feature_optimizer.step()

                wandb.log({
                    "decoupled_MI": decoupled_MI_post_update,
                    "sum_downward_MI": sum_downward_MI_post_update,
                    "Psi": Psi,
                }, step=self.train_step)

                self.train_step += 1

        wandb.finish()

        return None

    def test(
        self,
        epochs: int,
    ):
        
        if self.test_run_id is None:
            self._init_wandb_test()
            self.init_estimators(train_mode=False)
            self.test_run_id = wandb.run.id
        else:
            wandb.init(project=self.project_name+"-validation", id=self.test_run_id, resume="must")
        
        if self.train_run_id is not None:
            wandb.config.update({'train_run_id': self.train_run_id}, allow_val_change=True)
            
        wandb.config.update({'epochs': self.test_epochs}, allow_val_change=True)

        for _ in tqdm.tqdm(range(epochs), desc='Training'):
            for _, batch in enumerate(self.train_loader):
                x0 = batch[:, 0].to(device).float()
                x1 = batch[:, 1].to(device).float()
                v0_B = self.feature_network(x0).detach()
                v1_B = self.feature_network(x1).detach()


                # update decoupled critic
                self.decoupled_optimizer.zero_grad()
                decoupled_MI = self.decoupled_MI_estimator(v0_B, v1_B)
                decoupled_loss = -decoupled_MI
                decoupled_loss.backward(retain_graph=True)
                self.decoupled_optimizer.step()


                # update each downward critic 
                for i in range(self.test_config['num_atoms']):
                    self.downward_optims[i].zero_grad()
                    channel_i = x0[:, i].unsqueeze(1).detach()
                    downward_MI_i = self.downward_MI_estimators[i](v1_B, channel_i)
                    downward_loss = - downward_MI_i
                    downward_loss.backward(retain_graph=True)
                    self.downward_optims[i].step()
                    wandb.log({
                        f"downward_MI_{i}": downward_MI_i   
                    }, step=self.test_step)


                downward_MIs_post_update = [] # used for calculating the adjusted Psi and Psi adjustment
                v0_B = self.feature_network(x0)
                v1_B = self.feature_network(x1)

                for i in range(self.test_config['num_atoms']): # finds sum of downward terms
                    channel_i = x0[:, i].unsqueeze(1)
                    channel_i_MI = self.downward_MI_estimators[i](v1_B, channel_i)
                    downward_MIs_post_update.append(channel_i_MI)

                sum_downward_MI_post_update = sum(downward_MIs_post_update)
                decoupled_MI_post_update = self.decoupled_MI_estimator(v0_B, v1_B)

                if self.test_config["adjust_Psi"]:
                    clipped_min_MIs = max(0, min(downward_MIs_post_update))
                    Psi_adjustment = (self.test_config['num_atoms'] - 1) * clipped_min_MIs
                    Psi = decoupled_MI_post_update - sum_downward_MI_post_update + Psi_adjustment
                else:
                    Psi = decoupled_MI_post_update - sum_downward_MI_post_update

                wandb.log({
                    "decoupled_MI": decoupled_MI_post_update,
                    "sum_downward_MI": sum_downward_MI_post_update,
                    "Psi": Psi,
                }, step=self.test_step)

                self.test_step += 1
            
        wandb.finish()
        return None


    def finish_run(self):
        if self.path_where_to_save_model is not None:
            torch.save(self.feature_network.state_dict(), self.path_where_to_save_model)
        wandb.finish()





            
        