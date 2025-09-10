"""
Test Suite for DenseNet Optimization and Benchmarking

This module provides comprehensive testing for all components of the
DenseNet optimization and benchmarking suite.
"""

import os
import sys
import unittest
import tempfile
import shutil
import torch
import torchvision.models as models
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from benchmark import DenseNetBenchmarker, BenchmarkConfig, BenchmarkResult, SystemMonitor
from optimizations import (
    QuantizationOptimizer, 
    PruningOptimizer, 
    ONNXOptimizer, 
    MixedPrecisionOptimizer,
    create_optimization_suite
)
from main import DenseNetOptimizationSuite


class TestSystemMonitor(unittest.TestCase):
    """Test system monitoring utilities."""
    
    def test_get_memory_usage(self):
        """Test memory usage monitoring."""
        ram_usage, vram_usage = SystemMonitor.get_memory_usage()
        
        self.assertIsInstance(ram_usage, float)
        self.assertIsInstance(vram_usage, float)
        self.assertGreaterEqual(ram_usage, 0)
        self.assertGreaterEqual(vram_usage, 0)
    
    def test_get_cpu_utilization(self):
        """Test CPU utilization monitoring."""
        cpu_util = SystemMonitor.get_cpu_utilization()
        
        self.assertIsInstance(cpu_util, float)
        self.assertGreaterEqual(cpu_util, 0)
        self.assertLessEqual(cpu_util, 100)
    
    def test_get_gpu_utilization(self):
        """Test GPU utilization monitoring."""
        gpu_util = SystemMonitor.get_gpu_utilization()
        
        self.assertIsInstance(gpu_util, float)
        self.assertGreaterEqual(gpu_util, 0)
        self.assertLessEqual(gpu_util, 100)


class TestBenchmarkConfig(unittest.TestCase):
    """Test benchmark configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = BenchmarkConfig()
        
        self.assertEqual(config.model_name, "densenet121")
        self.assertEqual(config.input_size, (3, 224, 224))
        self.assertEqual(config.batch_sizes, [1, 4, 8, 16, 32])
        self.assertEqual(config.num_warmup, 10)
        self.assertEqual(config.num_iterations, 100)
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = BenchmarkConfig(
            model_name="densenet121",
            batch_sizes=[1, 2],
            device="cpu",
            num_iterations=5
        )
        
        self.assertEqual(config.batch_sizes, [1, 2])
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.num_iterations, 5)


class TestBenchmarkResult(unittest.TestCase):
    """Test benchmark result data structure."""
    
    def test_benchmark_result_creation(self):
        """Test creating benchmark result."""
        result = BenchmarkResult(
            model_variant="test",
            batch_size=1,
            device="cpu",
            ram_usage_mb=100.0,
            vram_usage_mb=0.0,
            cpu_utilization_pct=50.0,
            gpu_utilization_pct=0.0,
            latency_ms=10.0,
            throughput_samples_sec=100.0,
            accuracy_top1=0.8,
            accuracy_top5=0.9,
            model_size_mb=32.0,
            optimization_technique="baseline"
        )
        
        self.assertEqual(result.model_variant, "test")
        self.assertEqual(result.batch_size, 1)
        self.assertEqual(result.device, "cpu")
        self.assertEqual(result.ram_usage_mb, 100.0)


class TestDenseNetBenchmarker(unittest.TestCase):
    """Test DenseNet benchmarker."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = BenchmarkConfig(device="cpu", num_iterations=2)
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_benchmarker_initialization(self):
        """Test benchmarker initialization."""
        benchmarker = DenseNetBenchmarker(self.config, self.temp_dir)
        
        self.assertIsNotNone(benchmarker.model)
        self.assertEqual(benchmarker.config.device, "cpu")
        self.assertEqual(benchmarker.output_dir, self.temp_dir)
    
    def test_get_model_size(self):
        """Test model size calculation."""
        benchmarker = DenseNetBenchmarker(self.config, self.temp_dir)
        model_size = benchmarker._get_model_size(benchmarker.model)
        
        self.assertIsInstance(model_size, float)
        self.assertGreater(model_size, 0)
    
    def test_create_dummy_data(self):
        """Test dummy data creation."""
        benchmarker = DenseNetBenchmarker(self.config, self.temp_dir)
        dummy_data = benchmarker._create_dummy_data()
        
        self.assertEqual(dummy_data.shape, (1, 3, 224, 224))
        self.assertEqual(dummy_data.device.type, "cpu")


class TestOptimizationTechniques(unittest.TestCase):
    """Test optimization techniques."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = models.densenet121(pretrained=False)  # Use untrained for faster testing
        self.device = "cpu"
    
    def test_quantization_optimizer(self):
        """Test quantization optimization."""
        optimizer = QuantizationOptimizer(self.model, self.device, "dynamic")
        optimized_model = optimizer.optimize()
        
        self.assertIsNotNone(optimized_model)
        self.assertNotEqual(optimized_model, self.model)
    
    def test_pruning_optimizer(self):
        """Test pruning optimization."""
        optimizer = PruningOptimizer(self.model, self.device, 0.1, "unstructured")
        optimized_model = optimizer.optimize()
        
        self.assertIsNotNone(optimized_model)
        self.assertNotEqual(optimized_model, self.model)
    
    def test_mixed_precision_optimizer(self):
        """Test mixed precision optimization."""
        optimizer = MixedPrecisionOptimizer(self.model, self.device)
        optimized_model = optimizer.optimize()
        
        self.assertIsNotNone(optimized_model)
        self.assertNotEqual(optimized_model, self.model)
    
    def test_optimization_suite_creation(self):
        """Test optimization suite creation."""
        suite = create_optimization_suite(self.model, self.device)
        
        self.assertIsNotNone(suite)
        self.assertGreater(len(suite.optimizers), 0)


class TestDenseNetOptimizationSuite(unittest.TestCase):
    """Test main optimization suite."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_suite_initialization(self):
        """Test suite initialization."""
        suite = DenseNetOptimizationSuite(self.temp_dir, "cpu")
        
        self.assertIsNotNone(suite.base_model)
        self.assertEqual(suite.device, "cpu")
        self.assertEqual(suite.output_dir, self.temp_dir)
    
    def test_device_setup(self):
        """Test device setup."""
        suite = DenseNetOptimizationSuite(self.temp_dir, "auto")
        
        # Should fallback to CPU if CUDA not available
        self.assertIn(suite.device, ["cpu", "cuda"])


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    @patch('src.benchmark.DenseNetBenchmarker.run_benchmark')
    def test_benchmark_integration(self, mock_run_benchmark):
        """Test benchmark integration."""
        # Mock the benchmark results
        mock_result = BenchmarkResult(
            model_variant="test",
            batch_size=1,
            device="cpu",
            ram_usage_mb=100.0,
            vram_usage_mb=0.0,
            cpu_utilization_pct=50.0,
            gpu_utilization_pct=0.0,
            latency_ms=10.0,
            throughput_samples_sec=100.0,
            accuracy_top1=0.8,
            accuracy_top5=0.9,
            model_size_mb=32.0,
            optimization_technique="baseline"
        )
        mock_run_benchmark.return_value = [mock_result]
        
        # Test the suite
        suite = DenseNetOptimizationSuite(self.temp_dir, "cpu")
        results = suite.run_complete_benchmark()
        
        self.assertIsInstance(results, dict)
        self.assertIn("baseline", results)
    
    def test_results_saving(self):
        """Test results saving functionality."""
        suite = DenseNetOptimizationSuite(self.temp_dir, "cpu")
        
        # Create mock results
        mock_result = BenchmarkResult(
            model_variant="test",
            batch_size=1,
            device="cpu",
            ram_usage_mb=100.0,
            vram_usage_mb=0.0,
            cpu_utilization_pct=50.0,
            gpu_utilization_pct=0.0,
            latency_ms=10.0,
            throughput_samples_sec=100.0,
            accuracy_top1=0.8,
            accuracy_top5=0.9,
            model_size_mb=32.0,
            optimization_technique="baseline"
        )
        suite.results = [mock_result]
        
        # Save results
        df = suite.save_comprehensive_results("test_results.csv")
        
        # Check if file was created
        results_file = os.path.join(self.temp_dir, "test_results.csv")
        self.assertTrue(os.path.exists(results_file))
        
        # Check if DataFrame is valid
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)


class TestDataValidation(unittest.TestCase):
    """Test data validation and edge cases."""
    
    def test_invalid_batch_size(self):
        """Test handling of invalid batch sizes."""
        config = BenchmarkConfig(batch_sizes=[0, -1, 1000])
        
        # Should handle invalid batch sizes gracefully
        self.assertIn(0, config.batch_sizes)
        self.assertIn(-1, config.batch_sizes)
        self.assertIn(1000, config.batch_sizes)
    
    def test_memory_usage_validation(self):
        """Test memory usage validation."""
        ram_usage, vram_usage = SystemMonitor.get_memory_usage()
        
        # Should return valid values
        self.assertIsInstance(ram_usage, (int, float))
        self.assertIsInstance(vram_usage, (int, float))
        self.assertGreaterEqual(ram_usage, 0)
        self.assertGreaterEqual(vram_usage, 0)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestSystemMonitor,
        TestBenchmarkConfig,
        TestBenchmarkResult,
        TestDenseNetBenchmarker,
        TestOptimizationTechniques,
        TestDenseNetOptimizationSuite,
        TestIntegration,
        TestDataValidation
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running DenseNet Optimization Test Suite...")
    print("=" * 50)
    
    success = run_tests()
    
    if success:
        print("\n" + "=" * 50)
        print("All tests passed! ✅")
        sys.exit(0)
    else:
        print("\n" + "=" * 50)
        print("Some tests failed! ❌")
        sys.exit(1)
