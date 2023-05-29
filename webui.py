import argparse
import glob
import os
import shutil
import site
import subprocess
import sys

script_dir = os.getcwd()


def run_cmd(cmd, capture_output=False, env=None):
    # Run shell commands
    return subprocess.run(cmd, shell=True, capture_output=capture_output, env=env)


def check_env():
    # If we have access to conda, we are probably in an environment
    conda_not_exist = run_cmd("conda", capture_output=True).returncode
    if conda_not_exist:
        print("Conda is not installed. Exiting...")
        sys.exit()
    
    # Ensure this is a new environment and not the base environment
    if os.environ["CONDA_DEFAULT_ENV"] == "base":
        print("Create an environment for this project and activate it. Exiting...")
        sys.exit()


def install_dependencies():
    # Select your GPU or, choose to run in CPU mode
    print("What is your GPU")
    print()
    print("A) NVIDIA")
    print("B) AMD")
    print("C) Apple M Series")
    print("D) None (I want to run in CPU mode)")
    print()
    gpuchoice = input("Input> ").lower()

    # Install the version of PyTorch needed
    if gpuchoice == "a":
        run_cmd("conda install -y -k pytorch[version=2,build=py3.10_cuda11.7*] torchvision torchaudio pytorch-cuda=11.7 cuda-toolkit ninja git -c pytorch -c nvidia/label/cuda-11.7.0 -c nvidia")
    elif gpuchoice == "b":
        print("AMD GPUs are not supported. Exiting...")
        sys.exit()
    elif gpuchoice == "c" or gpuchoice == "d":
        run_cmd("conda install -y -k pytorch torchvision torchaudio cpuonly git -c pytorch")
    else:
        print("Invalid choice. Exiting...")
        sys.exit()

    # Clone webui to our computer
    run_cmd("git clone https://github.com/rsxdalv/tts-generation-webui.git")
    run_cmd("git clone https://github.com/rsxdalv/bark.git tts-generation-webui/models/bark")
    run_cmd("git clone https://github.com/rsxdalv/tortoise-tts.git tts-generation-webui/models/tortoise")
    run_cmd("pip install python-dotenv")
    run_cmd("pip install gradio")
    run_cmd("pip install soundfile") # torchaudio platform windows
    # run_cmd("pip install sox") # torchaudio platform linux

    # Install the webui dependencies
    update_dependencies()


def update_dependencies():
    # Update the webui dependencies
    os.chdir("tts-generation-webui")
    run_cmd("git pull")

    run_cmd("pip install -r requirements.txt")

    os.chdir(script_dir)

    os.chdir("tts-generation-webui/models/bark")
    run_cmd("git pull")

    # Installs/Updates dependencies from all requirements.txt
    run_cmd("python -m pip install .")
    os.chdir(script_dir)

    os.chdir("tts-generation-webui/models/tortoise")
    run_cmd("git pull")

    # Installs/Updates dependencies from all requirements.txt
    run_cmd("pip install transformers==4.19.0")
    run_cmd("pip install -r requirements.txt")
    run_cmd("python setup.py install")
    run_cmd("conda install -y --channel=numba llvmlite")

    os.chdir(script_dir)
    # clone if doesn't exist
    if not os.path.exists("tts-generation-webui/models/bark_voice_cloning_hubert_quantizer"):
        run_cmd("git clone --branch hubert_only https://github.com/rsxdalv/bark-voice-cloning-HuBERT-quantizer bark_voice_cloning_hubert_quantizer")

    os.chdir("tts-generation-webui/models/bark_voice_cloning_hubert_quantizer")
    # checkout to the correct branch
    run_cmd("git checkout hubert_only")
    run_cmd("git pull")

    # Installs/Updates dependencies from all requirements.txt
    run_cmd("pip install -r requirements.txt")


def run_model():
    os.chdir("tts-generation-webui")
    run_cmd("python server.py")  # put your flags here!


if __name__ == "__main__":
    # Verifies we are in a conda environment
    check_env()

    parser = argparse.ArgumentParser()
    parser.add_argument('--update', action='store_true', help='Update the web UI.')
    args = parser.parse_args()

    if args.update:
        update_dependencies()
    else:
        # If webui has already been installed, skip and run
        if not os.path.exists("tts-generation-webui/"):
            install_dependencies()
            os.chdir(script_dir)

        # Run the model with webui
        run_model()