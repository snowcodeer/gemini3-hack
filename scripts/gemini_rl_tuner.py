
import os
import argparse
import json
import re
import sys
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

def setup_gemini():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not found.")
    genai.configure(api_key=api_key)

def upload_to_gemini(path, mime_type=None):
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded {path} as {file.uri}")
    return file

def get_reward_code(script_path):
    with open(script_path, "r") as f:
        content = f.read()
    
    match = re.search(r"# <REWARD_START>(.*?)# <REWARD_END>", content, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find reward tags in {script_path}")
    
    return match.group(1).strip()

def apply_reward_code(script_path, new_code):
    with open(script_path, "r") as f:
        content = f.read()
    
    new_content = re.sub(
        r"# <REWARD_START>.*?# <REWARD_END>",
        f"# <REWARD_START>\n        {new_code}\n        # <REWARD_END>",
        content,
        flags=re.DOTALL
    )
    
    with open(script_path, "w") as f:
        f.write(new_content)
    print(f"Applied new reward logic to {script_path}")

def main():
    setup_gemini()
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Path to training run directory")
    parser.add_argument("--script", default="scripts/train_grasp_residual.py")
    parser.add_argument("--model", default="models/gemini-1.5-pro")
    args = parser.parse_args()

    run_path = Path(args.run_dir)
    plot_path = run_path / "plots" / "training.png"
    
    if not plot_path.exists():
        print(f"Warning: Plot not found at {plot_path}. Gemini will tune without visual history.")
        plot_file = None
    else:
        plot_file = upload_to_gemini(str(plot_path), mime_type="image/png")

    current_reward_code = get_reward_code(args.script)

    system_prompt = """
    You are a Robotics Reinforcement Learning Expert specializing in Reward Shaping.
    
    Your task is to analyze the training progress of a MuJoCo robotics agent and rewrite its reward function to improve performance.
    
    FAILURE MODES & HEURISTICS:
    - Stalling: High reward but low success rate. Solution: Increase success bonus (e.g. 50 -> 500), add time penalty (-0.1).
    - Dropping: Lift rate spikes down. Solution: Use a 'Delayed Drop Penalty' (penalize only if `ever_lifted` and not `is_lifted`).
    - Lazy Motion: Lifted but not moving to target. Solution: Increase transport scale or add a smoother 'Gradient Reward' (e.g., `(threshold - dist) * weight` when close).
    - Sudden Crashes: Success rate hits 40% then dives to 0. Solution: Check for 'Starting Punishment' (is the agent penalized for just being on the table?).
    
    OUTPUT FORMAT:
    You must output a JSON object with:
    1. "analysis": Your expert observation of the plot and current code.
    2. "new_reward_code": The updated Python logic to go INSIDE the _compute_reward method.
    
    Example JSON:
    {
      "analysis": "The agent is stalling because the lift reward is too high. I will boost the success reward to 1000 and add a time penalty.",
      "new_reward_code": "reward = -0.1\\n        if is_lifted:\\n            reward += 5.0\\n        if success:\\n            reward += 1000.0"
    }
    """

    user_prompt = f"""
    Current Reward Logic:
    ```python
    {current_reward_code}
    ```
    
    Analyze the training results and provided plot. Output the JSON with the improved 'new_reward_code'.
    Ensure the code is valid Python and uses the available variables: `dist_to_target`, `success`, `is_lifted`, `n_contacts`.
    """

    model = genai.GenerativeModel(
        model_name=args.model,
        system_instruction=system_prompt,
        generation_config={"response_mime_type": "application/json"}
    )

    inputs = [user_prompt]
    if plot_file:
        inputs.append(plot_file)
    
    print("Requesting iteration from Gemini...")
    response = model.generate_content(inputs)
    
    result = json.loads(response.text)
    print(f"\nGemini Analysis: {result['analysis']}")
    
    apply_reward_code(args.script, result['new_reward_code'])
    
    print(f"\nNext Step: Restart training with branching:")
    print(f"python {args.script} --video data/pick-3.mp4 --resume {args.run_dir}")

if __name__ == "__main__":
    main()
