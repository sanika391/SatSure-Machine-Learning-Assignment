"""
Serverless DenseNet Application for KNative Deployment

This module provides a Flask-based web service for DenseNet inference
optimized for serverless deployment with KNative.
"""

import os
import base64
import io
import logging
import time
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify
import numpy as np
from typing import Dict, Any, Tuple
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global variables for model and preprocessing
model = None
preprocess = None
device = None


def load_model() -> torch.nn.Module:
    """Load and optimize DenseNet model for inference."""
    global model, preprocess, device
    
    logger.info("Loading DenseNet model...")
    
    # Set device
    device = os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    try:
        model = models.densenet121(weights=None)
    except TypeError:
        model = models.densenet121(pretrained=False)
    model.to(device)
    model.eval()
    
    # Set up preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply optimizations for serverless deployment
    model = optimize_model_for_inference(model)
    
    logger.info("Model loaded and optimized successfully")
    return model


def optimize_model_for_inference(model: torch.nn.Module) -> torch.nn.Module:
    """Apply optimizations for serverless inference."""
    logger.info("Applying optimizations for serverless deployment...")
    
    # Enable inference mode
    model.eval()
    
    # Apply torch.jit.script for better performance
    try:
        model = torch.jit.script(model)
        logger.info("Applied TorchScript optimization")
    except Exception as e:
        logger.warning(f"TorchScript optimization failed: {e}")
    
    # Apply mixed precision if on GPU
    if device == 'cuda':
        try:
            model = model.half()  # Use FP16
            logger.info("Applied mixed precision optimization")
        except Exception as e:
            logger.warning(f"Mixed precision optimization failed: {e}")
    
    return model


def preprocess_image(image_data: str) -> torch.Tensor:
    """Preprocess base64 encoded image for inference."""
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply preprocessing
        input_tensor = preprocess(image).unsqueeze(0)
        input_tensor = input_tensor.to(device)
        
        # Apply mixed precision if using FP16
        if device == 'cuda' and input_tensor.dtype == torch.float32:
            input_tensor = input_tensor.half()
        
        return input_tensor
        
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise ValueError(f"Invalid image data: {e}")


def run_inference(input_tensor: torch.Tensor, batch_size: int = 1) -> Tuple[list, float]:
    """Run inference on preprocessed input."""
    start_time = time.time()
    
    try:
        with torch.no_grad():
            # Run inference
            outputs = model(input_tensor)
            
            # Get predictions
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top5_prob, top5_indices = torch.topk(probabilities, 5)
            
            # Convert to lists
            predictions = []
            for i in range(5):
                predictions.append({
                    'class_id': int(top5_indices[i]),
                    'confidence': float(top5_prob[i])
                })
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            return predictions, latency_ms
            
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise RuntimeError(f"Inference error: {e}")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': device,
        'timestamp': time.time()
    })


@app.route('/ready', methods=['GET'])
def readiness_check():
    """Readiness check endpoint."""
    if model is None:
        return jsonify({'status': 'not_ready', 'reason': 'model_not_loaded'}), 503
    
    return jsonify({
        'status': 'ready',
        'model_loaded': True,
        'device': device,
        'timestamp': time.time()
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint."""
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Extract parameters
        image_data = data.get('image')
        batch_size = data.get('batch_size', 1)
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Validate batch size
        if not isinstance(batch_size, int) or batch_size < 1 or batch_size > 32:
            return jsonify({'error': 'Invalid batch_size. Must be integer between 1 and 32'}), 400
        
        # Preprocess image
        input_tensor = preprocess_image(image_data)
        
        # Run inference
        predictions, latency_ms = run_inference(input_tensor, batch_size)
        
        # Prepare response
        response = {
            'predictions': predictions,
            'latency_ms': latency_ms,
            'model_variant': os.getenv('MODEL_VARIANT', 'optimized'),
            'device': device,
            'batch_size': batch_size,
            'timestamp': time.time()
        }
        
        return jsonify(response)
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({'error': str(e)}), 400
        
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        return jsonify({'error': str(e)}), 500
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/metrics', methods=['GET'])
def metrics():
    """Metrics endpoint for monitoring."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    # Get model info
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    
    # Get system info
    memory_info = {}
    if torch.cuda.is_available():
        memory_info = {
            'gpu_memory_allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024),
            'gpu_memory_reserved_mb': torch.cuda.memory_reserved() / (1024 * 1024)
        }
    
    return jsonify({
        'model_info': {
            'total_parameters': total_params,
            'model_size_mb': model_size_mb,
            'device': device
        },
        'system_info': memory_info,
        'timestamp': time.time()
    })


@app.route('/info', methods=['GET'])
def info():
    """Service information endpoint."""
    return jsonify({
        'service': 'densenet-optimization',
        'version': '1.0.0',
        'model': 'densenet121',
        'device': device,
        'optimizations': ['torchscript', 'mixed_precision'],
        'endpoints': {
            'health': '/health',
            'ready': '/ready',
            'predict': '/predict',
            'metrics': '/metrics',
            'info': '/info'
        }
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({'error': 'Method not allowed'}), 405


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


def main():
    """Main function to run the serverless app."""
    # Load model on startup
    load_model()
    
    # Get configuration
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 8080))
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    
    logger.info(f"Starting DenseNet serverless app on {host}:{port}")
    logger.info(f"Debug mode: {debug}")
    
    # Run the app
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == '__main__':
    main()
