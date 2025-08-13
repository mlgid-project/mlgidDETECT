import os
import platform
import subprocess
import sys

def run(cmd):
    print(f"\n Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def detect_os():
    os_name = platform.system()
    if os_name == "Windows":
        return "win"
    elif os_name == "Linux":
        return "linux"
    elif os_name == "Darwin":
        return "macos"
    else:
        raise RuntimeError(f"Unsupported OS: {os_name}")

def conda_env_exists(env_name):
    try:
        result = subprocess.run(
            ["conda", "env", "list"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return any(
            line.strip().startswith(env_name)
            or line.strip().endswith(f"/envs/{env_name}")
            for line in result.stdout.splitlines()
        )

    except subprocess.CalledProcessError as e:
        print("Failed to list conda environments.")
        print(e.stderr)
        sys.exit(1)

def create_env():
    os_type = detect_os()

    run("conda env create -f conda_gpu.yaml")
    run("conda run -n mlgiddetect-gpu pip uninstall -y opencv-python")

    # Install GPU-accelerated OpenCV wheel
    wheel_url = {
        "win": "https://github.com/cudawarped/opencv-python-cuda-wheels/releases/download/4.12.0.88/opencv_contrib_python-4.12.0.88-cp37-abi3-win_amd64.whl",
        "linux": "https://github.com/cudawarped/opencv-python-cuda-wheels/releases/download/4.12.0.88/opencv_contrib_python-4.12.0.88-cp37-abi3-linux_x86_64.whl",
        "macos": None
    }

    if wheel_url[os_type]:
        run(f"conda run -n mlgiddetect-gpu pip install {wheel_url[os_type]}")

    else:
        print("No GPU-accelerated OpenCV wheel available for macOS.")
        print("Aborting setup.")
        sys.exit(1)

    print("\nInstall script for CUDA support completed sucessfully. Use the environment with the command 'conda activate mlgiddetect-gpu'")

def main():
    response = input("Install script for the GPU support. Do you want to continue? Type y!")
    if response == "y":
        if conda_env_exists('mlgiddetect-gpu'):
            print("Error: Environment mlgiddetect-gpu already exists.")
            print("Please remove it first using:\n  conda env remove -n mlgiddetect-gpu")
            sys.exit(1)
        else:
            create_env()
    else:
        print("Aborting setup.")
        sys.exit(1)

if __name__ == "__main__":
    main()