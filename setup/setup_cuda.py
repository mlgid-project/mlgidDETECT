import os
import platform
import subprocess
import sys
import re

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

def get_glibc_version():
    try:
        output = subprocess.check_output(["ldd", "--version"], text=True)
        match = re.search(r"GLIBC\s+(\d+\.\d+)", output)
        if match:
            return match.group(1)
        else:
            # Fallback for typical ldd output
            match = re.search(r"ldd\s+\(.*\)\s+(\d+\.\d+)", output)
            return match.group(1) if match else None
    except Exception as e:
        print("Failed to detect glibc version:", e)
        return None

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
    glibc_version = get_glibc_version()

    if os_type == "win":
        yaml_file = 'conda_gpu.yaml'
        opencv_wheel = 'https://github.com/cudawarped/opencv-python-cuda-wheels/releases/download/4.12.0.88/opencv_contrib_python-4.12.0.88-cp37-abi3-win_amd64.whl'
    elif os_type == "linux":
        if glibc_version and float(glibc_version) < 2.35:
            yaml_file = "conda_gpu_old_glib.yaml"
            opencv_wheel = 'https://github.com/cudawarped/opencv-python-cuda-wheels/releases/download/4.7.0.20221229/opencv_contrib_python_rolling-4.7.0.20221229-cp36-abi3-linux_x86_64.whl'
        else:
            yaml_file = 'conda_gpu.yaml'
            opencv_wheel = 'https://github.com/cudawarped/opencv-python-cuda-wheels/releases/download/4.12.0.88/opencv_contrib_python-4.12.0.88-cp37-abi3-linux_x86_64.whl'
    else:
        print("No GPU-accelerated OpenCV wheel available for macOS.")
        print("Aborting setup.")
        sys.exit(1)

    run(f"conda env create -f {yaml_file}")
    run("conda run -n mlgiddetect-gpu pip uninstall -y opencv-python")
    run(f"conda run -n mlgiddetect-gpu pip install {opencv_wheel}")

    if os_type == "linux" and float(glibc_version) < 2.35:
        run("conda run -n mlgiddetect-gpu pip install 'numpy<2'")

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