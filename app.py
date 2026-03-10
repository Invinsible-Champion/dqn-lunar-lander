import streamlit as st
import gymnasium as gym
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import imageio
import os

# Import your trained brain
from src.network import DuelingQNetwork

# --- Page Config ---
st.set_page_config(page_title="DQN Lunar Lander", layout="wide")
st.title("🚀 Autonomous Lunar Lander: Live Telemetry")

st.markdown("""
Press **Run Live Simulation** to spawn a fresh environment seed. 
The trained Dueling Double DQN will navigate the descent in real-time.
""")

# --- Load the Model ---
@st.cache_resource
def load_model():
    # Initialize the network structure
    model = DuelingQNetwork(state_size=8, action_size=4)
    # Load the saved weights
    model.load_state_dict(torch.load("models/dqn_weights_best.pth"))
    model.eval() # Set to evaluation mode
    return model

model = load_model()

# --- Simulation Function ---
def run_simulation():
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    state, info = env.reset()
    
    frames = []
    telemetry = []
    total_reward = 0
    done = False
    
    while not done:
        frames.append(env.render())
        
        # Convert state to tensor and get the best action from our model
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_values = model(state_tensor)
            action = np.argmax(action_values.numpy())
            
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # Save telemetry data for the dashboard
        telemetry.append({
            "Frame": len(frames),
            "X_Pos": state[0],
            "Y_Pos": state[1],
            "X_Vel": state[2],
            "Y_Vel": state[3],
            "Angle": state[4],
            "Angular_Vel": state[5]
        })
        
        state = next_state
        
    env.close()
    
    # Save the video
    video_path = "latest_landing.mp4"
    imageio.mimsave(video_path, frames, fps=30)
    
    # Create the dataframe
    df = pd.DataFrame(telemetry)
    return video_path, df, total_reward

# --- Dashboard UI ---
if st.button("Run Live Simulation", type="primary"):
    
    with st.spinner("Model is calculating descent vectors..."):
        video_path, df, final_score = run_simulation()
        
    st.success(f"Landing Complete! Final Score: {final_score:.2f} (Target: > 200)")
    
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("Mission Playback")
        st.video(video_path, format="video/mp4", start_time=0)

    with col2:
        st.subheader("Flight Telemetry")
        
        # Velocity Graph
        fig_vel = px.line(df, x="Frame", y=["X_Vel", "Y_Vel"], 
                          title="Descent Velocity Profile",
                          labels={"value": "Velocity", "variable": "Axis"})
        st.plotly_chart(fig_vel, use_container_width=True)

        # Angle Graph
        fig_angle = px.line(df, x="Frame", y="Angle", 
                            title="Pitch Correction (Angle)",
                            color_discrete_sequence=["#FF4B4B"])
        st.plotly_chart(fig_angle, use_container_width=True)