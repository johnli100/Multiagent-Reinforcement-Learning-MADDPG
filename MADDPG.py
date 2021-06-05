import torch
from Agent import DDPGAgent

class MADDPG():
    def __init__(self, num_agents=2, state_all_dim=48, action_all_dim=4, state_dim=24, action_dim=2, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.num_agents = num_agents
        self.state_all_dim = state_all_dim
        self.action_all_dim = action_all_dim
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.agents = [DDPGAgent(state_all_dim, action_all_dim, state_dim, action_dim, self.device) for _ in range(num_agents)]

    def act(self, state, policy_noise):
        act_list = []
        for i, agent in enumerate(self.agents):
            act_list.append(agent.act(state[i], policy_noise))
        return act_list

    def learn(self, transitions, agent_id):
        actions_next_list, actor_preds_list = [], []
        for i, (agent, transition) in enumerate(zip(self.agents, transitions)):
            states_full, states, actions_full, actions, rewards, states_next_full, states_next, dones = transition
            actions_next_list.append(agent.actor_target(states_next).data)
            actor_preds_list.append(agent.actor(states) if i == agent_id else agent.actor(states).data)

        actions_next_full = torch.cat(actions_next_list, dim=1)
        actor_preds_full = torch.cat(actor_preds_list, dim=1)

        agent = self.agents[agent_id]
        states_full, states, actions_full, actions, rewards, states_next_full, states_next, dones = transitions[agent_id]
        agent.learn(states_full, actions_full, rewards, states_next_full, actions_next_full, dones, actor_preds_full)

    def soft_update(self, soft_update=True):
        for agent in self.agents:
            agent.sync_target(soft_update)

    def save_checkpoint(self):
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor.state_dict(), 'agent'+ str(i) + '_actor.pth')
            torch.save(agent.critic.state_dict(), 'agent' + str(i) + '_critic.pth')

    def load_checkpoint(self):
        for i, agent in enumerate(self.agents):
            agent.actor.load_state_dict(torch.load('agent'+ str(i) + '_actor.pth'))