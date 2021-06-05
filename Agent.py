import numpy as np
import torch
from torch import optim
from Network import *

GAMMA =0.99
TAU = 0.05
LR_CRITIC = 0.0025
LR_ACTOR = 0.0025

class DDPGAgent():

    def __init__(self, critic_state_dim, critic_action_dim, actor_state_dim, action_dim, device):
        self.critic_state_dim = critic_state_dim
        self.critic_action_dim = critic_action_dim
        self.actor_state_dim = actor_state_dim
        self.action_dim = action_dim
        self.device = device

        self.critic_target = Network_Critic(critic_state_dim, critic_action_dim).to(self.device)
        for param in self.critic_target.parameters():
            param.requires_grad = False

        self.actor_target = Network_Actor(actor_state_dim, action_dim).to(self.device)
        for param in self.actor_target.parameters():
            param.requires_grad = False

        self.critic = Network_Critic(critic_state_dim, critic_action_dim).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC, weight_decay=1.e-5)
        
        self.actor = Network_Actor(actor_state_dim, action_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        self.sync_target(soft_update=False)

    def act(self, state, policy_noise):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        action = self.actor(state).data.numpy()
        # random noise and clip
        action += np.random.randn(self.action_dim) * policy_noise

        return action.clip(-1., 1.)

    def learn(self, states_full, actions_full, rewards, states_next_full, actions_next_full, dones, actor_preds_full):
        # learn Value function (critic)
        Qs = self.critic(states_full, actions_full)
        Qs_target = self.critic_target(states_next_full, actions_next_full) * GAMMA * (1 - dones) + rewards
        loss_critic = torch.nn.functional.mse_loss(Qs, Qs_target)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # learn (optimized continuous) policy function (actor)
        loss_actor = -self.critic(states_full, actor_preds_full).mean()
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

    def sync_target(self, soft_update=True):
        if soft_update:
            for target_p, p in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_p.data.copy_(TAU * p.data + (1 - TAU) * target_p.data)
            for target_p, p in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_p.data.copy_(TAU * p.data + (1 - TAU) * target_p.data)
        else:
            for target_p, p in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_p.data.copy_(p.data)
            for target_p, p in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_p.data.copy_(p.data)