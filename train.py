import gymnasium as gym
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import os

from src.dqn_agent import Agent

def dqn_train(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.997):
    """Deep Q-Learning Training Loop."""
    
    # Initialize Environment and Agent
    env = gym.make('LunarLander-v3')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = Agent(state_size=state_size, action_size=action_size, seed=42)

    scores = []                        
    scores_window = deque(maxlen=100)  
    eps = eps_start    
    max_score = 150               

    print("Training Starts : ")
    
    for i_episode in range(1, n_episodes+1):
        state, info = env.reset()
        score = 0
        
        for t in range(max_t):
            # Choose an action
            action = agent.act(state, eps)
            
            # Execute the action in the environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Learns
            agent.step(state, action, reward, next_state, done)
            
            # Advance states and reqrds
            state = next_state
            score += reward
            
            if done:
                break 

        # Update tracking metrics
        scores_window.append(score)      
        scores.append(score)             
        eps = max(eps_end, eps_decay*eps) 
        
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tCurrent Epsilon: {eps:.3f}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')
            
        #Save best model continuosly    
        if len(scores_window) >= 100 and np.mean(scores_window) > max_score:
            max_score = np.mean(scores_window)
            os.makedirs('models', exist_ok=True)
            torch.save(agent.qnetwork_local.state_dict(), 'models/dqn_weights_best.pth')
            
        # Win Condition
        if len(scores_window) >= 100 and np.mean(scores_window) >= 240.0:
            print(f'\n\n Environment solved in {i_episode - 100:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
            os.makedirs('models', exist_ok=True)
            torch.save(agent.qnetwork_local.state_dict(), 'models/dqn_weights.pth')
            print("Model weights saved to 'models/dqn_weights.pth'")
            break
            
    env.close()
    return scores

if __name__ == "__main__":
    scores = dqn_train()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title('DQN Training Curve')
    plt.savefig('training_curve_1.png')
    print(" Training curve saved as 'training_curve_1.png'")