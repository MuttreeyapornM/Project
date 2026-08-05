import torch
import torchvision  # Add this import
import subprocess
import re

def check_pytorch_and_cuda():
    print("=== PyTorch & CUDA Info ===")
    print("PyTorch version:", torch.__version__)
    print("torchvision version:", torchvision.__version__)  # Added line
    
    if torch.cuda.is_available():
        print("CUDA is available")
        print("CUDA version:", torch.version.cuda)
        print("GPU device name:", torch.cuda.get_device_name(0))
    else:
        print("CUDA is not available")

def map_l4t_to_jetpack(l4t_version):
    l4t_to_jetpack = {
        "35.6.1": "5.1.2",
        "35.5.0": "5.1.1",
        "35.4.1": "5.1.0",
        "35.3.1": "5.0.2",
        "34.1.1": "5.0.1",
        "34.1.0": "5.0.0"
    }
    return l4t_to_jetpack.get(l4t_version, "Unknown JetPack version")

def check_jetpack_version():
    print("\n=== JetPack Info ===")
    try:
        result = subprocess.run(['head', '-n', '1', '/etc/nv_tegra_release'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            output = result.stdout.decode().strip()
            print("Raw L4T info:", output)

            match = re.search(r'R(\d+)\s.*REVISION:\s(\d+)\.(\d+)', output)
            if match:
                major = match.group(1)
                rev_major = match.group(2)
                rev_minor = match.group(3)
                l4t_version = f"{major}.{rev_major}.{rev_minor}"
                jetpack_version = map_l4t_to_jetpack(l4t_version)
                print(f"JetPack version: {jetpack_version}")
            else:
                print("Could not parse L4T version from output.")
        else:
            print("Not running on a Jetson device or file not found.")
    except Exception as e:
        print("Error fetching JetPack version:", e)

# Run checks
check_pytorch_and_cuda()
check_jetpack_version()



#----------------------------------output--------------------------
# === PyTorch & CUDA Info ===
# PyTorch version: 2.1.0a0+41361538.nv23.06
# CUDA is available
# CUDA version: 11.4
# GPU device name: Xavier

# === JetPack Info ===
# Raw L4T info: # R35 (release), REVISION: 6.1, GCID: 39721438, BOARD: t186ref, EABI: aarch64, DATE: Tue Mar  4 10:13:09 UTC 2025
# JetPack version: 5.1.2