# Autonomous Lunar Lander: Dueling Double DQN

An implementation of a Deep Reinforcement Learning agent trained to autonomously navigate the Gymnasium `LunarLander-v3` continuous control environment. 

The project features a custom-built **Dueling Double Deep Q-Network (D3QN)**, optimized with Huber loss and gradient clipping, achieving a highly stable, risk-adjusted convergence score of **242.25** in under 800 episodes.

##  Model Architecture
Instead of a standard Multi-Layer Perceptron, this agent utilizes a **Dueling Network Architecture**. 
* **State Space:** 8-dimensional continuous vector (X/Y position, X/Y velocity, angle, angular velocity, leg contact).
* **Value Stream $V(s)$:** Evaluates the baseline safety of the current environment state.
* **Advantage Stream $A(s,a)$:** Evaluates the marginal benefit of specific thruster actions.
* **Double DQN Logic:** Decouples action selection from action evaluation to prevent the overestimation of future Q-values, stabilizing the learning curve.

## 📈 Training Performance
The model was subjected to a rigorous training loop, deliberately slowing epsilon decay to force deep exploration and avoid local minimums. 

* **Target Win Condition:** Rolling average of $\ge 240$ over 100 consecutive episodes.
* **Convergence:** Successfully solved in **796 episodes**.
* **Optimization:** Implemented Smooth L1 (Huber) Loss and Gradient Norm Clipping to prevent catastrophic forgetting during highly volatile early-episode crashes.


## 🛠️ Tech Stack
* **Backend:** PyTorch, NumPy, Gymnasium (Box2D)
* **Frontend:** Streamlit, Pandas, Plotly
* **Dashboard:** A dynamic "Mission Control" interface that runs live inference, captures agent gameplay, and plots real-time telemetry (velocity and pitch correction vectors).

## 🚀 How to Run Locally

**1. Install Dependencies**
```bash
pip install -r requirements.txt