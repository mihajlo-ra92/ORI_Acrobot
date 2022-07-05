from neural_network import Agent
import numpy as np
import gym
import tensorflow as tf
import matplotlib.pyplot as plt

if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    env = gym.make('Acrobot-v1')
    input_shape = env.observation_space.shape
    actions_num = env.action_space.n
    lr = 0.001
    gamma = 0.99
    epsilon_min = 0.01
    epsilon = 1.0
    N = 100
    rewards = []
    fname = 'test_save.h5'
    load_val = input('Do you want to load model instead of creating a new one? y/n')
        
    if load_val.upper() == 'Y':
        epsilon = epsilon_min

    agent = Agent(gamma=gamma, epsilon=epsilon, lr=lr, 
                input_dims=input_shape,
                n_actions=actions_num, mem_size=1000000, batch_size=64,
                epsilon_min=epsilon_min, fname=fname)

    if load_val.upper() == 'Y':
        agent.load_model()

    for i in range(N):
        done = False
        reward = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, r, done, info = env.step(action)
            if i == N-1:
                env.render()
            reward += r
            agent.store_transition(observation, action, r, observation_, done)
            observation = observation_
            agent.learn()
        rewards.append(reward)

        avg_score = np.mean(rewards[-100:])
        print('episode: ', i, 'reward %.2f' % reward,
                'average_score %.2f' % avg_score,
                'epsilon %.2f' % agent.epsilon)

    plt.plot(rewards)
    plt.title("rewards")
    plt.show()

    save_val = input('Do you want to save model? y/n')
        
    if save_val.upper() == 'Y':
        agent.save_model()
        print('Model saved to: ', fname)

    
