"""
Master Entry Point for Autonomous Vision & Behaviour Understanding System
(HNX26PSI07)
Usage:
    python run.py             # Launches interactive Streamlit web dashboard
    python run.py --evaluate  # Runs automated benchmark evaluation against 6 judging criteria
    python run.py --cli       # Runs CLI batch processor on warehouse benchmark video
"""

import sys
import subprocess
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

def launch_dashboard(port: int = 8501):
    print("==================================================================")
    print(" Launching Autonomous Vision & Behaviour Understanding Dashboard")
    print(" Project Code: HNX26PSI07")
    print(f" Web UI: http://localhost:{port}")
    print("==================================================================")
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(ROOT_DIR / "app.py"),
        "--server.port", str(port),
        "--server.headless", "false"
    ]
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="Autonomous Vision & Behaviour Understanding Runner")
    parser.add_argument("--evaluate", action="store_true", help="Run automated 6-criteria benchmark evaluation")
    parser.add_argument("--cli", action="store_true", help="Run CLI video processing pipeline")
    parser.add_argument("--webcam", action="store_true", help="Launch real-time desktop webcam monitor")
    parser.add_argument("--camera-idx", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--port", type=int, default=8501, help="Port for Streamlit dashboard")
    args, unknown = parser.parse_known_args()

    if args.evaluate:
        from evaluate import run_evaluation
        run_evaluation()
    elif args.cli:
        from cli import main as cli_main
        cli_main()
    elif args.webcam:
        from live_monitor import run_live_webcam_monitor
        run_live_webcam_monitor(camera_index=args.camera_idx)
    else:
        launch_dashboard(port=args.port)

if __name__ == "__main__":
    main()
