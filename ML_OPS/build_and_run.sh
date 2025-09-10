#!/bin/bash

# DenseNet Optimization and Benchmarking - Build and Run Script
# This script automates the complete workflow from building to running the benchmark suite

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
OUTPUT_DIR="./results"
GPU_ENABLED="true"
DEVICE="cuda"
CLEAN_BUILD=false
SKIP_BUILD=false
RUN_TENSORBOARD=true
TENSORBOARD_PORT=6006

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

DenseNet Optimization and Benchmarking Suite

OPTIONS:
    --output-dir DIR          Output directory for results (default: ./results)
    --gpu-enabled BOOL        Enable GPU benchmarking (default: true)
    --device DEVICE           Device to use: cuda, cpu, auto (default: cuda)
    --clean-build             Clean build (remove existing images)
    --skip-build              Skip Docker build step
    --no-tensorboard          Skip starting TensorBoard
    --tensorboard-port PORT   TensorBoard port (default: 6006)
    --help                    Show this help message

EXAMPLES:
    $0 --output-dir ./my_results --gpu-enabled true
    $0 --device cpu --clean-build
    $0 --skip-build --no-tensorboard

EOF
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if Docker Compose is available
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not available. Please install Docker Compose."
        exit 1
    fi
    
    # Check if NVIDIA Docker is available (if GPU is enabled)
    if [ "$GPU_ENABLED" = "true" ] && [ "$DEVICE" = "cuda" ]; then
        if ! docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi &> /dev/null; then
            print_warning "NVIDIA Docker runtime not available. Falling back to CPU mode."
            DEVICE="cpu"
        fi
    fi
    
    print_success "Prerequisites check passed"
}

# Function to clean up
cleanup() {
    print_status "Cleaning up..."
    
    # Stop containers
    if [ "$RUN_TENSORBOARD" = "true" ]; then
        docker-compose down 2>/dev/null || true
    fi
    
    # Remove containers if clean build
    if [ "$CLEAN_BUILD" = "true" ]; then
        print_status "Removing existing images..."
        docker rmi densenet-optimization-benchmark 2>/dev/null || true
        docker rmi densenet-optimization-benchmark-cpu 2>/dev/null || true
    fi
}

# Function to build Docker image
build_image() {
    if [ "$SKIP_BUILD" = "true" ]; then
        print_status "Skipping Docker build..."
        return
    fi
    
    print_status "Building Docker image..."
    
    # Build the image
    docker build -t densenet-optimization-benchmark .
    
    if [ $? -eq 0 ]; then
        print_success "Docker image built successfully"
    else
        print_error "Docker build failed"
        exit 1
    fi
}

# Function to create output directories
create_directories() {
    print_status "Creating output directories..."
    
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR/tensorboard"
    mkdir -p "$OUTPUT_DIR/profiles"
    mkdir -p "$OUTPUT_DIR/models"
    mkdir -p "./logs"
    
    print_success "Output directories created"
}

# Function to run benchmark
run_benchmark() {
    print_status "Starting DenseNet optimization and benchmarking..."
    
    # Set device based on GPU availability
    if [ "$DEVICE" = "cuda" ] && [ "$GPU_ENABLED" = "true" ]; then
        print_status "Running with GPU acceleration..."
        docker run --rm \
            --gpus all \
            -v "$(pwd)/$OUTPUT_DIR:/app/results" \
            -v "$(pwd)/logs:/app/logs" \
            -e CUDA_VISIBLE_DEVICES=0 \
            densenet-optimization-benchmark \
            python src/main.py --output-dir /app/results --device cuda
    else
        print_status "Running with CPU only..."
        docker run --rm \
            -v "$(pwd)/$OUTPUT_DIR:/app/results" \
            -v "$(pwd)/logs:/app/logs" \
            densenet-optimization-benchmark \
            python src/main.py --output-dir /app/results --device cpu
    fi
    
    if [ $? -eq 0 ]; then
        print_success "Benchmark completed successfully"
    else
        print_error "Benchmark failed"
        exit 1
    fi
}

# Function to start TensorBoard
start_tensorboard() {
    if [ "$RUN_TENSORBOARD" = "false" ]; then
        print_status "Skipping TensorBoard startup"
        return
    fi
    
    print_status "Starting TensorBoard..."
    
    # Start TensorBoard in background
    docker run -d \
        --name densenet-tensorboard \
        -p "$TENSORBOARD_PORT:6006" \
        -v "$(pwd)/$OUTPUT_DIR/tensorboard:/logs" \
        tensorflow/tensorflow:latest-jupyter \
        tensorboard --logdir=/logs --host=0.0.0.0 --port=6006
    
    if [ $? -eq 0 ]; then
        print_success "TensorBoard started on port $TENSORBOARD_PORT"
        print_status "Access TensorBoard at: http://localhost:$TENSORBOARD_PORT"
    else
        print_warning "Failed to start TensorBoard"
    fi
}

# Function to display results
display_results() {
    print_status "Displaying results summary..."
    
    if [ -f "$OUTPUT_DIR/benchmark_results.csv" ]; then
        print_success "Results saved to: $OUTPUT_DIR/benchmark_results.csv"
        
        # Show basic statistics
        if command -v python3 &> /dev/null; then
            python3 -c "
import pandas as pd
import sys
try:
    df = pd.read_csv('$OUTPUT_DIR/benchmark_results.csv')
    print('\n' + '='*60)
    print('BENCHMARK RESULTS SUMMARY')
    print('='*60)
    print(f'Total experiments: {len(df)}')
    print(f'Optimization techniques: {df[\"optimization_technique\"].nunique()}')
    print(f'Batch sizes tested: {sorted(df[\"batch_size\"].unique())}')
    print(f'Device: {df[\"device\"].iloc[0]}')
    
    # Best performance
    best_throughput = df.loc[df['throughput_samples_sec'].idxmax()]
    best_latency = df.loc[df['latency_ms'].idxmin()]
    
    print(f'\nBest Throughput: {best_throughput[\"optimization_technique\"]} '
          f'(batch_size={best_throughput[\"batch_size\"]}) - '
          f'{best_throughput[\"throughput_samples_sec\"]:.2f} samples/sec')
    
    print(f'Best Latency: {best_latency[\"optimization_technique\"]} '
          f'(batch_size={best_latency[\"batch_size\"]}) - '
          f'{best_latency[\"latency_ms\"]:.2f} ms')
    
    print('='*60)
except Exception as e:
    print(f'Error reading results: {e}')
"
        fi
    else
        print_warning "Results file not found: $OUTPUT_DIR/benchmark_results.csv"
    fi
}

# Function to wait for completion
wait_for_completion() {
    print_status "Waiting for benchmark completion..."
    
    # Wait for the main container to finish
    while docker ps --format "table {{.Names}}" | grep -q "densenet-optimization-benchmark"; do
        sleep 5
        print_status "Benchmark in progress..."
    done
    
    print_success "Benchmark completed"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --gpu-enabled)
            GPU_ENABLED="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --clean-build)
            CLEAN_BUILD=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --no-tensorboard)
            RUN_TENSORBOARD=false
            shift
            ;;
        --tensorboard-port)
            TENSORBOARD_PORT="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_status "Starting DenseNet Optimization and Benchmarking Suite"
    print_status "Output directory: $OUTPUT_DIR"
    print_status "Device: $DEVICE"
    print_status "GPU enabled: $GPU_ENABLED"
    
    # Set up trap for cleanup on exit
    trap cleanup EXIT
    
    # Execute workflow
    check_prerequisites
    cleanup
    build_image
    create_directories
    run_benchmark
    start_tensorboard
    display_results
    
    print_success "DenseNet optimization and benchmarking suite completed successfully!"
    print_status "Results are available in: $OUTPUT_DIR"
    if [ "$RUN_TENSORBOARD" = "true" ]; then
        print_status "TensorBoard is running at: http://localhost:$TENSORBOARD_PORT"
    fi
}

# Run main function
main "$@"
