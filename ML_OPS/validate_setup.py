#!/usr/bin/env python3
"""
Validation Script for DenseNet Optimization and Benchmarking Suite

This script validates that all components are properly set up and working
before running the full benchmark suite.
"""

import os
import sys
import subprocess
import importlib
import torch
import torchvision
from pathlib import Path

def print_status(message, status="INFO"):
    """Print status message with color coding."""
    colors = {
        "INFO": "\033[94m",
        "SUCCESS": "\033[92m",
        "WARNING": "\033[93m",
        "ERROR": "\033[91m",
        "RESET": "\033[0m"
    }
    
    status_symbol = {
        "INFO": "ℹ️",
        "SUCCESS": "✅",
        "WARNING": "⚠️",
        "ERROR": "❌"
    }
    
    print(f"{colors[status]}{status_symbol[status]} {message}{colors['RESET']}")

def check_python_version():
    """Check Python version compatibility."""
    print_status("Checking Python version...")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print_status(f"Python {version.major}.{version.minor}.{version.micro} - Compatible", "SUCCESS")
        return True
    else:
        print_status(f"Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.10+", "ERROR")
        return False

def check_dependencies():
    """Check if all required dependencies are installed."""
    print_status("Checking dependencies...")
    
    required_packages = [
        'torch',
        'torchvision',
        'tensorboard',
        'numpy',
        'pandas',
        'PIL',
        'tqdm',
        'psutil',
        'matplotlib',
        'seaborn',
        'scikit-learn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                importlib.import_module('PIL')
            else:
                importlib.import_module(package)
            print_status(f"  {package} - Installed", "SUCCESS")
        except ImportError:
            print_status(f"  {package} - Missing", "ERROR")
            missing_packages.append(package)
    
    if missing_packages:
        print_status(f"Missing packages: {', '.join(missing_packages)}", "ERROR")
        print_status("Install with: pip install -r requirements.txt", "INFO")
        return False
    
    return True

def check_pytorch_setup():
    """Check PyTorch installation and CUDA availability."""
    print_status("Checking PyTorch setup...")
    
    # Check PyTorch version
    torch_version = torch.__version__
    print_status(f"PyTorch version: {torch_version}", "SUCCESS")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        cuda_version = torch.version.cuda
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
        print_status(f"CUDA available: {cuda_version} ({gpu_count} GPU(s))", "SUCCESS")
        print_status(f"GPU: {gpu_name}", "SUCCESS")
    else:
        print_status("CUDA not available - will use CPU", "WARNING")
    
    return True

def check_docker():
    """Check Docker installation and availability."""
    print_status("Checking Docker...")
    
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, check=True)
        print_status(f"Docker: {result.stdout.strip()}", "SUCCESS")
        
        # Check if Docker daemon is running
        result = subprocess.run(['docker', 'info'], 
                              capture_output=True, text=True, check=True)
        print_status("Docker daemon is running", "SUCCESS")
        return True
        
    except subprocess.CalledProcessError:
        print_status("Docker not available or not running", "ERROR")
        return False
    except FileNotFoundError:
        print_status("Docker not installed", "ERROR")
        return False

def check_docker_compose():
    """Check Docker Compose availability."""
    print_status("Checking Docker Compose...")
    
    try:
        # Try docker compose (newer version)
        result = subprocess.run(['docker', 'compose', 'version'], 
                              capture_output=True, text=True, check=True)
        print_status(f"Docker Compose: {result.stdout.strip()}", "SUCCESS")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # Try docker-compose (older version)
            result = subprocess.run(['docker-compose', '--version'], 
                                  capture_output=True, text=True, check=True)
            print_status(f"Docker Compose: {result.stdout.strip()}", "SUCCESS")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print_status("Docker Compose not available", "ERROR")
            return False

def check_nvidia_docker():
    """Check NVIDIA Docker runtime availability."""
    print_status("Checking NVIDIA Docker runtime...")
    
    try:
        result = subprocess.run(['docker', 'run', '--rm', '--gpus', 'all', 
                               'nvidia/cuda:11.8-base-ubuntu22.04', 'nvidia-smi'], 
                              capture_output=True, text=True, check=True)
        print_status("NVIDIA Docker runtime available", "SUCCESS")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_status("NVIDIA Docker runtime not available", "WARNING")
        return False

def check_file_structure():
    """Check if all required files are present."""
    print_status("Checking file structure...")
    
    required_files = [
        'src/__init__.py',
        'src/benchmark.py',
        'src/optimizations.py',
        'src/main.py',
        'src/serverless_app.py',
        'Dockerfile',
        'Dockerfile.serverless',
        'docker-compose.yml',
        'build_and_run.sh',
        'setup-cluster.sh',
        'deploy-knative.sh',
        'cleanup-knative.sh',
        'kind-config.yaml',
        'knative-service.yaml',
        'pyproject.toml',
        'requirements.txt',
        'README.md'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print_status(f"  {file_path} - Present", "SUCCESS")
        else:
            print_status(f"  {file_path} - Missing", "ERROR")
            missing_files.append(file_path)
    
    if missing_files:
        print_status(f"Missing files: {', '.join(missing_files)}", "ERROR")
        return False
    
    return True

def check_script_permissions():
    """Check if shell scripts have execute permissions."""
    print_status("Checking script permissions...")
    
    scripts = [
        'build_and_run.sh',
        'setup-cluster.sh',
        'deploy-knative.sh',
        'cleanup-knative.sh'
    ]
    
    for script in scripts:
        if os.path.exists(script):
            if os.access(script, os.X_OK):
                print_status(f"  {script} - Executable", "SUCCESS")
            else:
                print_status(f"  {script} - Not executable", "WARNING")
                # Try to make it executable
                try:
                    os.chmod(script, 0o755)
                    print_status(f"  {script} - Made executable", "SUCCESS")
                except Exception as e:
                    print_status(f"  {script} - Failed to make executable: {e}", "ERROR")
        else:
            print_status(f"  {script} - Not found", "ERROR")

def check_model_loading():
    """Test if DenseNet model can be loaded."""
    print_status("Testing model loading...")
    
    try:
        model = torchvision.models.densenet121(pretrained=False)  # Use untrained for faster loading
        print_status("DenseNet-121 model loaded successfully", "SUCCESS")
        
        # Test model properties
        total_params = sum(p.numel() for p in model.parameters())
        print_status(f"Model parameters: {total_params:,}", "SUCCESS")
        
        return True
    except Exception as e:
        print_status(f"Failed to load model: {e}", "ERROR")
        return False

def check_output_directories():
    """Check if output directories can be created."""
    print_status("Checking output directories...")
    
    test_dir = "test_output"
    
    try:
        os.makedirs(test_dir, exist_ok=True)
        os.makedirs(os.path.join(test_dir, "tensorboard"), exist_ok=True)
        os.makedirs(os.path.join(test_dir, "profiles"), exist_ok=True)
        os.makedirs(os.path.join(test_dir, "models"), exist_ok=True)
        
        print_status("Output directories can be created", "SUCCESS")
        
        # Clean up
        import shutil
        shutil.rmtree(test_dir)
        
        return True
    except Exception as e:
        print_status(f"Failed to create output directories: {e}", "ERROR")
        return False

def run_quick_test():
    """Run a quick test of the benchmark system."""
    print_status("Running quick test...")
    
    try:
        # Import the modules
        sys.path.insert(0, 'src')
        from benchmark import BenchmarkConfig, DenseNetBenchmarker
        
        # Create a minimal config for testing
        config = BenchmarkConfig(
            device="cpu",
            batch_sizes=[1],
            num_iterations=1,
            num_warmup=1
        )
        
        # Test benchmarker initialization
        benchmarker = DenseNetBenchmarker(config, "test_output")
        print_status("Benchmarker initialized successfully", "SUCCESS")
        
        # Test model size calculation
        model_size = benchmarker._get_model_size(benchmarker.model)
        print_status(f"Model size: {model_size:.2f} MB", "SUCCESS")
        
        # Clean up
        import shutil
        if os.path.exists("test_output"):
            shutil.rmtree("test_output")
        
        return True
    except Exception as e:
        print_status(f"Quick test failed: {e}", "ERROR")
        return False

def main():
    """Main validation function."""
    print("DenseNet Optimization and Benchmarking Suite - Validation")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("PyTorch Setup", check_pytorch_setup),
        ("Docker", check_docker),
        ("Docker Compose", check_docker_compose),
        ("NVIDIA Docker", check_nvidia_docker),
        ("File Structure", check_file_structure),
        ("Script Permissions", check_script_permissions),
        ("Model Loading", check_model_loading),
        ("Output Directories", check_output_directories),
        ("Quick Test", run_quick_test)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        print(f"\n--- {check_name} ---")
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print_status(f"Check failed with exception: {e}", "ERROR")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for check_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✅" if result else "❌"
        print(f"{symbol} {check_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print_status("All checks passed! System is ready for benchmarking.", "SUCCESS")
        return True
    else:
        print_status(f"{total - passed} checks failed. Please fix the issues before running benchmarks.", "ERROR")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
