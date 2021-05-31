import numpy as np
import torch
from collections import namedtuple, deque
import matplotlib.pyplot as plt
from MADDPG import MADDPG
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

maddpg_agents = MADDPG(num_agents, state_size*num_agents, action_size*num_agents, state_size, action_size)

def maddpg(maddpg_agents, n_episodes=2500, n_episodes_burn=200, update_every=2, batch_size=1024, policy_noise=0.5, policy_noise_decay=0.9999,policy_noise_lowest=0.05):
    step = 0
    scores = []
    scores_window = deque(maxlen=100)

    for i_episode in range(1, n_episodes+1):
        score = np.zeros(num_agents)
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations

        while True:
            transitions = []

            if i_episode <= n_episodes_burn:
                actions = np.random.randn(num_agents, action_size)  # select an action (for each agent)
                actions = np.clip(actions, -1, 1)  # all actions between -1 and 1
            else:
                actions = maddpg_agents.act(states, policy_noise)
                policy_noise *= policy_noise_decay
                policy_noise = max(policy_noise_lowest, policy_noise)

            env_info = env.step(actions)[brain_name]
            states_next= env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            states_full = states.reshape(1, -1)
            actions_full = np.array(actions).reshape(1, -1)
            states_next_full = states_next.reshape(1, -1)
            for j in range(num_agents):
                transitions.append(Transition(states_full, states[j], actions_full, actions[j], rewards[j], states_next_full, states_next[j], dones[j]))
            maddpg_agents.cache(transitions)

            step += 1
            score += rewards
            states = states_next

            if (len(maddpg_agents.memory) >= batch_size) and ((i_episode % update_every) == 0):
                for agent_id in range(num_agents):
                    experiences = maddpg_agents.recall(batch_size)
                    maddpg_agents.learn(experiences, agent_id)
                maddpg_agents.soft_update()

            if np.any(dones):
                break

        scores_window.append(max(score))
        scores.append(max(score))
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if (i_episode) % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

    maddpg_agents.save_checkpoint()

    return scores

scores = maddpg(maddpg_agents)
# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()




env.close()