from collections import deque, namedtuple
import random
import torch

MEMORY_SIZE = 100000
Transition = namedtuple('Transition', ('state_full', 'state', 'action_full', 'action', 'reward', 'state_next_full', 'state_next', 'done'))

class buffer():
    def __init__(self, num_agents, device=torch.device('cpu')):
        super().__init__()
        self.num_agents = num_agents
        self.device = device
        self.memory = deque(maxlen=MEMORY_SIZE)

    def cache(self, transitions):
        assert len(transitions) == self.num_agents
        transitions_list = []
        for transition in transitions:
            state_full = torch.tensor(transition.state_full, dtype=torch.float, device=self.device).squeeze(0)
            state = torch.tensor(transition.state, dtype=torch.float, device=self.device).squeeze(0)
            action_full = torch.tensor(transition.action_full, dtype=torch.float, device=self.device).squeeze(0)
            action = torch.tensor(transition.action, dtype=torch.float, device=self.device).squeeze(0)
            reward = torch.tensor(transition.reward, dtype=torch.float, device=self.device).unsqueeze(0)
            state_next_full = torch.tensor(transition.state_next_full, dtype=torch.float, device=self.device).squeeze(0)
            state_next = torch.tensor(transition.state_next, dtype=torch.float, device=self.device).squeeze(0)
            done = torch.tensor(transition.done, dtype=torch.float, device=self.device).unsqueeze(0)
            transitions_list.append(Transition(state_full, state, action_full, action, reward, state_next_full, state_next, done))
        self.memory.append(transitions_list)

    def sample(self, batch_size):
        transitions_list = []
        transitions = random.sample(self.memory, batch_size)
        for transition in zip(*transitions):
            states_full, states, actions_full, actions, rewards, states_next_full, states_next, dones = map(torch.stack, zip(*transition))
            transitions_list.append([states_full, states, actions_full, actions, rewards, states_next_full, states_next, dones])
        return transitions_list
