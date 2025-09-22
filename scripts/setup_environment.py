import os
import sys
import subprocess
from pathlib import Path

def run_command(command, check=True):               # Run shell command and handle errors
    try:
        result = subprocess.run(command, shell=True, check=check, 
                              capture_output=True, text=True)
        return result
    except subprocess.CalledProcessError as e:
        print(f" Command failed: {command}")
        print(f"Error: {e.stderr}")
        return None
    
def check_python_version():                         # Check if Python version is compatible
    python_version = sys.version_info
    if python_version.major != 3 or python_version.minor < 8:
        print(" Python 3.8 or higher is required")
        sys.exit(1)
    print(f" Python {python_version.major}.{python_version.minor}.{python_version.micro} detected")


def create_virtual_environment():                   # Create virtual environment

    venv_path = Path("venv")
    
    if venv_path.exists():
        print(" Virtual environment already exists")
        return
    
    print(" Creating virtual environment...")
    result = run_command("python -m venv venv")
    if result:
        print(" Virtual environment created")
    else:
        print(" Failed to create virtual environment")
        sys.exit(1)

def install_requirements():                         # Install requirements in virtual environment
    print(" Installing requirements...")
    
    if os.name == 'nt':                 # Windows
        pip_path = "venv\\Scripts\\pip"
    else:                               # Unix/Linux/MacOS
        pip_path = "venv/bin/pip"
    
    install_cmd = f"{pip_path} install -r requirements.txt"
    result = run_command(install_cmd)
    
    if result:
        print("Requirements installed successfully")
    else:
        print("Failed to install requirements")
        sys.exit(1)

def create_directory_structure():           # Create project directory structure

    print(" Creating directory structure...")
    
    directories = [
        "data/raw", "data/processed", "data/features", "data/external",
        "notebooks", "src/data", "src/models", "src/evaluation", 
        "src/optimization", "src/deployment", "src/utils",
        "tests/test_data", "tests/test_models", "tests/test_evaluation",
        "configs", "scripts", "docs", "reports/figures",
        "models/trained_models", "models/checkpoints", "models/experiment_logs",
        "deployments/docker", "deployments/kubernetes", "deployments/monitoring",
        "environment", "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        if directory.startswith("src/"):
            init_file = Path(directory) / "__init__.py"
            init_file.touch()
    
    print(" Directory structure created")

def create_config_files():                      # Create default configuration files

    print(" Creating configuration files...")
    
    # Create empty __init__.py files for src subdirectories
    src_dirs = ["src/data", "src/models", "src/evaluation", "src/optimization", "src/deployment"]
    for src_dir in src_dirs:
        init_file = Path(src_dir) / "__init__.py"
        if not init_file.exists():
            init_file.touch()
    
    print(" Configuration files created")


def main():             # Main setup function

    print(" Setting up Churn Prediction Development Environment")
    print("-" * 50)
    
    check_python_version()
    create_directory_structure()
    create_virtual_environment()
    install_requirements()
    create_config_files()
    
    print("\n Setup completed successfully!")
    print("\nNext steps:")
    print("1. Activate virtual environment:")
    
    if os.name == 'nt':  # Windows
        print("   venv\\Scripts\\activate")
    else:  # Unix/Linux/MacOS
        print("   source venv/bin/activate")
    
    print("2. Download sample data:")
    print("   python scripts/download_data.py")
    print("3. Run Phase 1 analysis:")
    print("   python src/phase1_data_exploration.py")

if __name__ == "__main__":
    main()
