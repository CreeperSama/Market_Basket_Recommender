import subprocess
import sys
import os

def run_script(script_path):
    """Executes a Python script and stops the pipeline if it crashes."""
    print(f"\n{'='*50}")
    print(f"🚀 EXECUTING: {script_path}")
    print(f"{'='*50}\n")
    
    # sys.executable ensures it uses your current Python virtual environment
    result = subprocess.run([sys.executable, script_path])
    
    if result.returncode != 0:
        print(f"\n❌ FATAL ERROR: {script_path} failed. Halting pipeline.")
        sys.exit(1)

def main():
    # Ensure we are running this from the project root directory
    if not os.path.exists('src'):
        print("Error: Please run this script from the root of your project folder.")
        sys.exit(1)

    print("Starting E-Commerce AI Pipeline Build...\n")

    # PHASE 1: Data Preprocessing
    run_script('src/chatbot/preprocess_nlp.py')
    run_script('src/recommendation/preprocess_instacart.py')
    
    # PHASE 2: Model Training
    run_script('src/chatbot/train_nb.py')
    run_script('src/recommendation/train_rf.py')
    
    # PHASE 3: Launch the Application
    print("\n✅ Build Complete! Launching Chatbot Interface...\n")
    run_script('src/app.py')

if __name__ == "__main__":
    main()