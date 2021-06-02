import numpy as np
import torch
import matplotlib.pyplot as plt
import itertools
from collections import namedtuple, deque
from MADDPG import MADDPG
from Buffer import buffer
from unityagents import UnityEnvironment

Transition = namedtuple('Transition', ('state_full','state', 'action_full', 'action', 'reward', 'state_next_full', 'state_next', 'done'))

env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86_64")
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]
# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)
# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)
# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

maddpg_agents = MADDPG(num_agents, state_size*num_agents, action_size*num_agents, state_size, action_size, device)
buffer = buffer(num_agents, device)

def train(maddpg_agents, n_episodes=2200, n_episodes_wait=200, update_every=2, \
          batch_size=1024, noise_start=2., noise_decay=0.9999, noise_end=0.2, share_experience=False):

    noise = noise_start
    scores = []
    scores_window = deque(maxlen=100)

    for i_episode in range(1, n_episodes+1):
        score = np.zeros(num_agents)
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        dones = env_info.local_done

        while not np.any(dones):
            transitions = []

            if i_episode <= n_episodes_wait:
                noise = noise_start
                actions = np.random.randn(num_agents, action_size)  # select an action (for each agent)
                actions = np.clip(actions, -1., 1.)
                #actions = maddpg_agents.act(states, noise_start)
            else:
                actions = maddpg_agents.act(states, noise)

            env_info = env.step(actions)[brain_name]
            states_next= env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            states_full = states.reshape(1, -1)
            actions_full = np.array(actions).reshape(1, -1)
            states_next_full = states_next.reshape(1, -1)
            for j in range(num_agents):
                transitions.append(Transition(states_full, states[j], actions_full, actions[j], rewards[j], states_next_full, states_next[j], dones[j]))
            buffer.cache(transitions)
            if share_experience:
                # transitions = list(itertools.permutations(transitions))
                # for t in transitions:
                #     buffer.cache(t)
                transitions.reverse()
                buffer.cache(transitions)

            score += rewards
            states = states_next
            noise *= noise_decay
            noise = max(noise_end, noise)

            if (len(buffer.memory) >= batch_size) and ((i_episode % update_every) == 0):
                for agent_id in range(num_agents):
                    experiences = buffer.sample(batch_size)
                    maddpg_agents.learn(experiences, agent_id)
                maddpg_agents.soft_update()

        scores_window.append(max(score))
        scores.append(max(score))
        print('\rEpisode {}\tAverage Score: {:.2f}; noise: {:.2f}'.format(i_episode, np.mean(scores_window),noise), end="")
        if (i_episode) % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

        if scores_window[-1]>0.5:
            maddpg_agents.save_checkpoint()

    return scores

scores = train(maddpg_agents, share_experience=False)
# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()




env.close()