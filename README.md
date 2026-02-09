# OVERFIT: One Video Episode Reward Function from Imitation Tracking

OVERFIT is an end-to-end pipeline for teaching complex robotic tasks to an Adroit Shadow Hand using a single real-world video demonstration. It leverages Gemini for task understanding and Stable Baselines3 (TD3) for Residual Reinforcement Learning.

---

## 🚀 Quick Start

### 1. Installation
Ensure you have a Python 3.10+ environment.

```powershell
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
Rename `.env.template` to `.env` and add your API keys:
*   `GEMINI_API_KEY`: For video analysis and critiques.
*   `BLOB_ACCESS_KEY` & `BLOB_SECRET_KEY`: For cloud synchronization (optional).

---

## 🛠️ Key Commands

### A. RL Training
Start a new training session with default rewards (v3):
```powershell
python scripts/train_grasp_residual.py --video data/pick-3.mp4 --timesteps 500000
```

#### Dynamic Configuration
OVERFIT automatically generates reward hyperparameters based on the video analysis (e.g., `grasp_frame`, `task_type`). You can override these defaults by editing `data/video_analysis.json` or by using a specific reward version:

```powershell
# v1: Original basic reward
# v2: High-reward/Unstable
# v3: Stabilized expert heuristics (Default)
python scripts/train_grasp_residual.py --video data/pick-3.mp4 --reward-version v3
```

### B. Automated Iteration (Gemini RL Tuner)
OVERFIT can automatically improve its own reward function by analyzing training plots and detecting failure modes (Stalling, Dropping, etc.).
```powershell
# Analyze a run and rewrite the reward for the next iteration
python scripts/gemini_rl_tuner.py --run-dir runs/residual_pick3/pick-3_...
```

#### Custom Reward Logic
Override the entire reward logic with an external `.py` file:
```powershell
python scripts/train_grasp_residual.py --video data/pick-3.mp4 --reward-file experiments/v4_custom.py
```

### C. Launching the Dashboard
The dashboard requires two processes running simultaneously.

**Terminal 1: FastAPI Backend**
```powershell
python scripts/server.py
```

**Terminal 2: React Frontend**
```powershell
cd dashboard
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) to view your experiments.

---

## 📂 Project Structure

*   `scripts/`: Core execution scripts (Training, Server, Evaluation).
    *   `reward_registry.py`: Central store for named reward versions (v1, v2, v3).
*   `dashboard/`: React + Vite frontend source code.
*   `vidreward/`: Internal library for extraction, retargeting, and utilities.
    *   `phases/`: Phase-aware reward logic.
    *   `utils/storage.py`: Cloud-hybrid blob storage bridge.
*   `runs/`: Local storage for training artifacts (models, plots, and history).
*   `data/`: Source videos and generated trajectory data.

---

## ☁️ Cloud-Hybrid Mode
VidReward is designed for a hybrid workflow:
1.  **Local Worker**: Run heavy RL training on your local GPU.
2.  **Cloud Dashboard**: Host the Dashboard in the cloud.
3.  **Auto-Sync**: The training script automatically pushes progress to your configured Blob Storage, making results visible in the cloud dashboard instantly.
