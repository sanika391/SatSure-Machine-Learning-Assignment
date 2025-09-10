"""
Main Application for DenseNet Optimization and Benchmarking

This module orchestrates the complete benchmarking and optimization workflow,
integrating all optimization techniques with comprehensive performance analysis.
"""

import os
import sys
import argparse
import logging
import torch
import torchvision.models as models
from typing import Dict, List
import pandas as pd
from datetime import datetime

from benchmark import DenseNetBenchmarker, BenchmarkConfig, BenchmarkResult
from optimizations import create_optimization_suite, OptimizationManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('benchmark.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DenseNetOptimizationSuite:
    """Main class for DenseNet optimization and benchmarking suite."""
    
    def __init__(self, output_dir: str = "./results", device: str = "auto"):
        self.output_dir = output_dir
        self.device = self._setup_device(device)
        self.results: List[BenchmarkResult] = []
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "tensorboard"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "profiles"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
        
        # Load base model
        self.base_model = self._load_base_model()
        
        # Create optimization suite
        self.optimization_suite = create_optimization_suite(self.base_model, self.device)
        
    def _setup_device(self, device: str) -> str:
        """Setup and validate device configuration."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = "cpu"
                logger.info("Using CPU (GPU not available)")
        else:
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                device = "cpu"
        
        return device
    
    def _load_base_model(self) -> torch.nn.Module:
        """Load the base DenseNet model."""
        logger.info("Loading DenseNet-121 model...")
        try:
            model = models.densenet121(weights=None)
        except TypeError:
            model = models.densenet121(pretrained=False)
        model.to(self.device)
        model.eval()
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model loaded: {total_params:,} total parameters, {trainable_params:,} trainable")
        
        return model
    
    def run_complete_benchmark(self) -> Dict[str, List[BenchmarkResult]]:
        """Run complete benchmarking suite with all optimization techniques."""
        logger.info("Starting complete DenseNet optimization and benchmarking suite")
        
        all_results = {}
        
        # 1. Baseline benchmark
        logger.info("=" * 60)
        logger.info("RUNNING BASELINE BENCHMARK")
        logger.info("=" * 60)
        
        baseline_config = BenchmarkConfig(device=self.device)
        baseline_benchmarker = DenseNetBenchmarker(baseline_config, self.output_dir)
        baseline_results = baseline_benchmarker.run_benchmark("baseline")
        all_results["baseline"] = baseline_results
        self.results.extend(baseline_results)
        baseline_benchmarker.cleanup()
        
        # 2. Optimization benchmarks
        logger.info("=" * 60)
        logger.info("RUNNING OPTIMIZATION BENCHMARKS")
        logger.info("=" * 60)
        
        # Get all optimized models
        optimized_models = self.optimization_suite.optimize_all()
        
        for opt_name, optimized_model in optimized_models.items():
            try:
                logger.info(f"Benchmarking {opt_name}...")
                
                # Create benchmarker for this optimization
                config = BenchmarkConfig(device=self.device)
                benchmarker = DenseNetBenchmarker(config, self.output_dir)
                
                # Replace the model with optimized version
                benchmarker.model = optimized_model
                benchmarker.model.to(self.device)
                benchmarker.model.eval()
                
                # Run benchmark
                opt_results = benchmarker.run_benchmark(opt_name)
                all_results[opt_name] = opt_results
                self.results.extend(opt_results)
                
                # Save optimized model
                model_path = os.path.join(self.output_dir, "models", f"{opt_name}_model.pth")
                if hasattr(optimized_model, 'state_dict'):
                    torch.save(optimized_model.state_dict(), model_path)
                else:
                    # For ONNX models, save the original path
                    logger.info(f"ONNX model saved separately")
                
                benchmarker.cleanup()
                
            except Exception as e:
                logger.error(f"Failed to benchmark {opt_name}: {e}")
                continue
        
        return all_results
    
    def save_comprehensive_results(self, filename: str = "benchmark_results.csv"):
        """Save all results to CSV with comprehensive analysis."""
        if not self.results:
            logger.warning("No results to save")
            return
        
        # Create DataFrame
        df = pd.DataFrame(self.results)
        
        # Add additional analysis columns
        df['efficiency_score'] = df['throughput_samples_sec'] / df['model_size_mb']
        df['memory_efficiency'] = df['throughput_samples_sec'] / (df['ram_usage_mb'] + df['vram_usage_mb'])
        df['timestamp'] = datetime.now().isoformat()
        
        # Save to CSV
        output_path = os.path.join(self.output_dir, filename)
        df.to_csv(output_path, index=False)
        logger.info(f"Comprehensive results saved to {output_path}")
        
        # Save summary statistics
        self._save_summary_statistics(df)
        
        return df
    
    def _save_summary_statistics(self, df: pd.DataFrame):
        """Save detailed summary statistics."""
        summary_path = os.path.join(self.output_dir, "summary_statistics.txt")
        
        with open(summary_path, 'w') as f:
            f.write("DenseNet Optimization Benchmark Summary\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall statistics
            f.write("Overall Statistics:\n")
            f.write(f"Total experiments: {len(df)}\n")
            f.write(f"Optimization techniques: {df['optimization_technique'].nunique()}\n")
            f.write(f"Batch sizes tested: {sorted(df['batch_size'].unique())}\n")
            f.write(f"Device: {df['device'].iloc[0]}\n\n")
            
            # Performance by optimization technique
            f.write("Performance by Optimization Technique:\n")
            f.write("-" * 40 + "\n")
            
            for technique in df['optimization_technique'].unique():
                tech_df = df[df['optimization_technique'] == technique]
                f.write(f"\n{technique.upper()}:\n")
                f.write(f"  Average Latency: {tech_df['latency_ms'].mean():.2f} ms\n")
                f.write(f"  Average Throughput: {tech_df['throughput_samples_sec'].mean():.2f} samples/sec\n")
                f.write(f"  Average RAM Usage: {tech_df['ram_usage_mb'].mean():.2f} MB\n")
                f.write(f"  Average VRAM Usage: {tech_df['vram_usage_mb'].mean():.2f} MB\n")
                f.write(f"  Model Size: {tech_df['model_size_mb'].iloc[0]:.2f} MB\n")
                f.write(f"  Efficiency Score: {tech_df['efficiency_score'].mean():.2f}\n")
            
            # Best performing configurations
            f.write("\n\nBest Performing Configurations:\n")
            f.write("-" * 40 + "\n")
            
            # Best throughput
            best_throughput = df.loc[df['throughput_samples_sec'].idxmax()]
            f.write(f"Best Throughput: {best_throughput['optimization_technique']} "
                   f"(batch_size={best_throughput['batch_size']}) - "
                   f"{best_throughput['throughput_samples_sec']:.2f} samples/sec\n")
            
            # Best latency
            best_latency = df.loc[df['latency_ms'].idxmin()]
            f.write(f"Best Latency: {best_latency['optimization_technique']} "
                   f"(batch_size={best_latency['batch_size']}) - "
                   f"{best_latency['latency_ms']:.2f} ms\n")
            
            # Most memory efficient
            best_memory = df.loc[df['memory_efficiency'].idxmax()]
            f.write(f"Most Memory Efficient: {best_memory['optimization_technique']} "
                   f"(batch_size={best_memory['batch_size']}) - "
                   f"{best_memory['memory_efficiency']:.2f} efficiency score\n")
    
    def print_final_summary(self):
        """Print final summary to console."""
        if not self.results:
            logger.warning("No results to summarize")
            return
        
        df = pd.DataFrame(self.results)
        
        print("\n" + "=" * 80)
        print("DENSENET OPTIMIZATION BENCHMARK - FINAL SUMMARY")
        print("=" * 80)
        
        print(f"\nTotal Experiments: {len(df)}")
        print(f"Optimization Techniques: {df['optimization_technique'].nunique()}")
        print(f"Batch Sizes: {sorted(df['batch_size'].unique())}")
        print(f"Device: {df['device'].iloc[0]}")
        
        # Performance comparison
        print("\n" + "=" * 50)
        print("PERFORMANCE COMPARISON")
        print("=" * 50)
        
        comparison_df = df.groupby('optimization_technique').agg({
            'latency_ms': 'mean',
            'throughput_samples_sec': 'mean',
            'ram_usage_mb': 'mean',
            'vram_usage_mb': 'mean',
            'model_size_mb': 'first',
            'efficiency_score': 'mean'
        }).round(2)
        
        print(comparison_df)
        
        # Recommendations
        print("\n" + "=" * 50)
        print("RECOMMENDATIONS")
        print("=" * 50)
        
        # Best for throughput
        best_throughput = df.loc[df['throughput_samples_sec'].idxmax()]
        print(f"Best for High Throughput: {best_throughput['optimization_technique']} "
              f"(batch_size={best_throughput['batch_size']})")
        
        # Best for latency
        best_latency = df.loc[df['latency_ms'].idxmin()]
        print(f"Best for Low Latency: {best_latency['optimization_technique']} "
              f"(batch_size={best_latency['batch_size']})")
        
        # Best for memory efficiency
        best_memory = df.loc[df['memory_efficiency'].idxmax()]
        print(f"Best for Memory Efficiency: {best_memory['optimization_technique']} "
              f"(batch_size={best_memory['batch_size']})")
        
        print("\n" + "=" * 80)
        print("Benchmark completed successfully!")
        print(f"Results saved to: {self.output_dir}")
        print(f"TensorBoard logs: {os.path.join(self.output_dir, 'tensorboard')}")
        print("=" * 80)


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="DenseNet Optimization and Benchmarking Suite")
    parser.add_argument("--output-dir", default="./results", 
                       help="Output directory for results")
    parser.add_argument("--device", default="auto", 
                       choices=["auto", "cpu", "cuda"],
                       help="Device to use for benchmarking")
    parser.add_argument("--gpu-enabled", action="store_true",
                       help="Enable GPU benchmarking (deprecated, use --device)")
    
    args = parser.parse_args()
    
    # Handle deprecated gpu-enabled flag
    if args.gpu_enabled and args.device == "auto":
        args.device = "cuda"
    
    logger.info("Starting DenseNet Optimization and Benchmarking Suite")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Device: {args.device}")
    
    try:
        # Create optimization suite
        suite = DenseNetOptimizationSuite(args.output_dir, args.device)
        
        # Run complete benchmark
        all_results = suite.run_complete_benchmark()
        
        # Save results
        df = suite.save_comprehensive_results()
        
        # Print final summary
        suite.print_final_summary()
        
        logger.info("Benchmark suite completed successfully!")
        
    except Exception as e:
        logger.error(f"Benchmark suite failed: {e}")
        raise


if __name__ == "__main__":
    main()
