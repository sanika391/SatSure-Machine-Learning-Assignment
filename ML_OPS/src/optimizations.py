"""
Model Optimization Techniques

This module implements various optimization techniques for DenseNet models including
quantization, pruning, and ONNX conversion for production deployment.
"""

import os
import torch
import torch.nn as nn
try:
    import torch.quantization as quantization
except ImportError:
    import torch.ao.quantization as quantization
import torchvision.models as models
try:
    from torch.quantization import quantize_dynamic, quantize_fx
except ImportError:
    from torch.ao.quantization import quantize_dynamic, quantize_fx
import onnx
import onnxruntime as ort
from typing import Dict, Any, Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


class ModelOptimizer:
    """Base class for model optimization techniques."""
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device
        self.optimized_model = None
        
    def optimize(self) -> nn.Module:
        """Apply optimization technique."""
        raise NotImplementedError
        
    def get_model_size(self) -> float:
        """Get optimized model size in MB."""
        if self.optimized_model is None:
            return 0.0
        
        param_size = sum(p.numel() * p.element_size() for p in self.optimized_model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.optimized_model.buffers())
        return (param_size + buffer_size) / (1024 * 1024)
    
    def save_model(self, path: str):
        """Save optimized model."""
        if self.optimized_model is None:
            raise ValueError("Model not optimized yet")
        
        torch.save(self.optimized_model.state_dict(), path)
        logger.info(f"Optimized model saved to {path}")


class QuantizationOptimizer(ModelOptimizer):
    """Dynamic and Static Quantization for DenseNet."""
    
    def __init__(self, model: nn.Module, device: str = "cuda", quantization_type: str = "dynamic"):
        super().__init__(model, device)
        self.quantization_type = quantization_type
        
    def optimize(self) -> nn.Module:
        """Apply quantization optimization."""
        logger.info(f"Applying {self.quantization_type} quantization...")
        
        if self.quantization_type == "dynamic":
            self.optimized_model = self._apply_dynamic_quantization()
        elif self.quantization_type == "static":
            self.optimized_model = self._apply_static_quantization()
        else:
            raise ValueError(f"Unsupported quantization type: {self.quantization_type}")
        
        return self.optimized_model
    
    def _apply_dynamic_quantization(self) -> nn.Module:
        """Apply dynamic quantization."""
        # Dynamic quantization is applied at inference time
        quantized_model = quantize_dynamic(
            self.model, 
            {nn.Linear, nn.Conv2d}, 
            dtype=torch.qint8
        )
        return quantized_model
    
    def _apply_static_quantization(self) -> nn.Module:
        """Apply static quantization with calibration."""
        # Set model to evaluation mode
        self.model.eval()
        
        # Create a dummy calibration dataset
        def calibrate_model(model, data_loader):
            model.eval()
            with torch.no_grad():
                for i, (data, _) in enumerate(data_loader):
                    if i >= 10:  # Use only first 10 batches for calibration
                        break
                    model(data)
        
        # Prepare model for quantization
        try:
            self.model.qconfig = quantization.get_default_qconfig('fbgemm')
            quantization.prepare(self.model, inplace=True)
        except Exception as e:
            logger.warning(f"Quantization preparation failed: {e}")
            return self.model
        
        # Create dummy data loader for calibration
        dummy_data = torch.randn(1, 3, 224, 224)
        dummy_loader = [(dummy_data, torch.tensor(0)) for _ in range(10)]
        
        # Calibrate model
        calibrate_model(self.model, dummy_loader)
        
        # Convert to quantized model
        try:
            quantized_model = quantization.convert(self.model, inplace=False)
        except Exception as e:
            logger.warning(f"Quantization conversion failed: {e}")
            return self.model
        
        return quantized_model


class PruningOptimizer(ModelOptimizer):
    """Structured and Unstructured Pruning for DenseNet."""
    
    def __init__(self, model: nn.Module, device: str = "cuda", 
                 pruning_ratio: float = 0.2, pruning_type: str = "unstructured"):
        super().__init__(model, device)
        self.pruning_ratio = pruning_ratio
        self.pruning_type = pruning_type
        
    def optimize(self) -> nn.Module:
        """Apply pruning optimization."""
        logger.info(f"Applying {self.pruning_type} pruning with ratio {self.pruning_ratio}...")
        
        if self.pruning_type == "unstructured":
            self.optimized_model = self._apply_unstructured_pruning()
        elif self.pruning_type == "structured":
            self.optimized_model = self._apply_structured_pruning()
        else:
            raise ValueError(f"Unsupported pruning type: {self.pruning_type}")
        
        return self.optimized_model
    
    def _apply_unstructured_pruning(self) -> nn.Module:
        """Apply unstructured pruning."""
        import torch.nn.utils.prune as prune
        
        # Create a copy of the model
        pruned_model = self.model
        
        # Prune all Conv2d and Linear layers
        for name, module in pruned_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prune.l1_unstructured(module, name='weight', amount=self.pruning_ratio)
        
        # Remove pruning reparameterization
        for name, module in pruned_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prune.remove(module, 'weight')
        
        return pruned_model
    
    def _apply_structured_pruning(self) -> nn.Module:
        """Apply structured pruning."""
        import torch.nn.utils.prune as prune
        
        # Create a copy of the model
        pruned_model = self.model
        
        # Prune Conv2d layers with structured pruning
        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.ln_structured(module, name='weight', amount=self.pruning_ratio, n=2, dim=0)
        
        # Remove pruning reparameterization
        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.remove(module, 'weight')
        
        return pruned_model


class ONNXOptimizer(ModelOptimizer):
    """ONNX conversion and optimization for DenseNet."""
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        super().__init__(model, device)
        self.onnx_model = None
        self.ort_session = None
        
    def optimize(self) -> nn.Module:
        """Convert model to ONNX format."""
        logger.info("Converting model to ONNX format...")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        
        # Export to ONNX
        onnx_path = "densenet_optimized.onnx"
        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Load and optimize ONNX model
        self.onnx_model = onnx.load(onnx_path)
        
        # Optimize ONNX model
        from onnxruntime.tools import optimizer
        self.onnx_model = optimizer.optimize_model(self.onnx_model)
        
        # Create ONNX Runtime session
        self.ort_session = ort.InferenceSession(onnx_path)
        
        # Create a wrapper to make it compatible with PyTorch interface
        self.optimized_model = ONNXWrapper(self.ort_session)
        
        return self.optimized_model
    
    def get_model_size(self) -> float:
        """Get ONNX model size in MB."""
        if self.onnx_model is None:
            return 0.0
        
        return os.path.getsize("densenet_optimized.onnx") / (1024 * 1024)


class ONNXWrapper(nn.Module):
    """Wrapper to make ONNX model compatible with PyTorch interface."""
    
    def __init__(self, ort_session):
        super().__init__()
        self.ort_session = ort_session
        
    def forward(self, x):
        """Forward pass using ONNX Runtime."""
        # Convert to numpy
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
        else:
            x_np = x
            
        # Run inference
        outputs = self.ort_session.run(None, {'input': x_np})
        
        # Convert back to torch tensor
        return torch.from_numpy(outputs[0])


class MixedPrecisionOptimizer(ModelOptimizer):
    """Mixed precision optimization using automatic mixed precision (AMP)."""
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        super().__init__(model, device)
        self.scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
        
    def optimize(self) -> nn.Module:
        """Apply mixed precision optimization."""
        logger.info("Applying mixed precision optimization...")
        
        # Mixed precision is typically applied during training
        # For inference, we can use autocast
        self.optimized_model = MixedPrecisionWrapper(self.model)
        
        return self.optimized_model


class MixedPrecisionWrapper(nn.Module):
    """Wrapper for mixed precision inference."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        """Forward pass with mixed precision."""
        with torch.cuda.amp.autocast():
            return self.model(x)


class OptimizationManager:
    """Manager class for applying multiple optimization techniques."""
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device
        self.optimizers = {}
        
    def add_optimizer(self, name: str, optimizer: ModelOptimizer):
        """Add an optimizer to the manager."""
        self.optimizers[name] = optimizer
        
    def optimize_all(self) -> Dict[str, nn.Module]:
        """Apply all optimizations."""
        optimized_models = {}
        
        for name, optimizer in self.optimizers.items():
            try:
                logger.info(f"Applying optimization: {name}")
                optimized_model = optimizer.optimize()
                optimized_models[name] = optimized_model
                logger.info(f"Successfully applied {name}")
            except Exception as e:
                logger.error(f"Failed to apply {name}: {e}")
                continue
                
        return optimized_models
    
    def get_optimization_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all optimizations."""
        summary = {}
        
        for name, optimizer in self.optimizers.items():
            if hasattr(optimizer, 'optimized_model') and optimizer.optimized_model is not None:
                summary[name] = {
                    'model_size_mb': optimizer.get_model_size(),
                    'optimization_type': type(optimizer).__name__
                }
        
        return summary


def create_optimization_suite(model: nn.Module, device: str = "cuda") -> OptimizationManager:
    """Create a complete optimization suite for DenseNet."""
    manager = OptimizationManager(model, device)
    
    # Add different optimization techniques
    manager.add_optimizer("dynamic_quantization", 
                         QuantizationOptimizer(model, device, "dynamic"))
    manager.add_optimizer("static_quantization", 
                         QuantizationOptimizer(model, device, "static"))
    manager.add_optimizer("unstructured_pruning", 
                         PruningOptimizer(model, device, 0.2, "unstructured"))
    manager.add_optimizer("structured_pruning", 
                         PruningOptimizer(model, device, 0.2, "structured"))
    manager.add_optimizer("onnx_conversion", 
                         ONNXOptimizer(model, device))
    manager.add_optimizer("mixed_precision", 
                         MixedPrecisionOptimizer(model, device))
    
    return manager


def main():
    """Main function for testing optimizations."""
    # Load DenseNet model
    try:
        model = models.densenet121(weights=None)
    except TypeError:
        model = models.densenet121(pretrained=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    # Create optimization suite
    optimization_suite = create_optimization_suite(model, device)
    
    # Apply all optimizations
    optimized_models = optimization_suite.optimize_all()
    
    # Print summary
    summary = optimization_suite.get_optimization_summary()
    print("\nOptimization Summary:")
    print("=" * 50)
    for name, info in summary.items():
        print(f"{name}: {info['model_size_mb']:.2f} MB ({info['optimization_type']})")


if __name__ == "__main__":
    main()
