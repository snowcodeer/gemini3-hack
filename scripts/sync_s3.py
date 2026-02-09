import os
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vidreward.utils.storage import storage

def sync():
    local_runs_dir = "runs"
    if not os.path.exists(local_runs_dir):
        print(f"Error: {local_runs_dir} directory not found.")
        return

    print(f"Starting S3 sync for {local_runs_dir}...")
    try:
        storage.upload_dir(local_runs_dir, "runs")
        print("S3 sync completed successfully.")
    except Exception as e:
        print(f"S3 sync failed: {e}")

if __name__ == "__main__":
    sync()
