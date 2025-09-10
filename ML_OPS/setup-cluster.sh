#!/bin/bash

# Setup Kind cluster with KNative for DenseNet serverless deployment
# This script creates and configures a local Kind cluster with KNative Serving

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if kind is installed
    if ! command -v kind &> /dev/null; then
        print_error "Kind is not installed. Please install Kind first:"
        echo "curl -Lo kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64"
        echo "chmod +x kind"
        echo "sudo mv kind /usr/local/bin/"
        exit 1
    fi
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed. Please install kubectl first:"
        echo "curl -LO \"https://dl.k8s.io/release/\$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl\""
        echo "chmod +x kubectl"
        echo "sudo mv kubectl /usr/local/bin/"
        exit 1
    fi
    
    # Check if Docker is running
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Create Kind cluster
create_cluster() {
    print_status "Creating Kind cluster..."
    
    # Delete existing cluster if it exists
    if kind get clusters | grep -q "densenet-knative"; then
        print_warning "Existing cluster found, deleting..."
        kind delete cluster --name densenet-knative
    fi
    
    # Create new cluster
    kind create cluster --config=kind-config.yaml
    
    if [ $? -eq 0 ]; then
        print_success "Kind cluster created successfully"
    else
        print_error "Failed to create Kind cluster"
        exit 1
    fi
}

# Install KNative Serving
install_knative() {
    print_status "Installing KNative Serving..."
    
    # Install KNative CRDs
    print_status "Installing KNative CRDs..."
    kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.11.0/serving-crds.yaml
    
    # Install KNative Core
    print_status "Installing KNative Core..."
    kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.11.0/serving-core.yaml
    
    # Wait for KNative to be ready
    print_status "Waiting for KNative to be ready..."
    kubectl wait --for=condition=Ready --timeout=300s pod -l app=controller -n knative-serving
    
    print_success "KNative Serving installed successfully"
}

# Install networking layer
install_networking() {
    print_status "Installing networking layer..."
    
    # Install Kourier (lightweight networking layer)
    kubectl apply -f https://github.com/knative/net-kourier/releases/download/knative-v1.11.0/kourier.yaml
    
    # Configure Kourier as default ingress
    kubectl patch configmap/config-network \
      --namespace knative-serving \
      --type merge \
      --patch '{"data":{"ingress-class":"kourier.ingress.networking.knative.dev"}}'
    
    # Wait for Kourier to be ready
    print_status "Waiting for Kourier to be ready..."
    kubectl wait --for=condition=Ready --timeout=300s pod -l app=3scale-kourier-gateway -n kourier-system
    
    print_success "Networking layer installed successfully"
}

# Build and load Docker image
build_and_load_image() {
    print_status "Building and loading Docker image..."
    
    # Build the Docker image
    docker build -t densenet-optimization:latest .
    
    # Load image into Kind cluster
    kind load docker-image densenet-optimization:latest --name densenet-knative
    
    print_success "Docker image built and loaded successfully"
}

# Deploy DenseNet service
deploy_service() {
    print_status "Deploying DenseNet service..."
    
    # Apply the KNative service
    kubectl apply -f knative-service.yaml
    
    # Wait for service to be ready
    print_status "Waiting for service to be ready..."
    kubectl wait --for=condition=Ready --timeout=300s ksvc densenet-service
    
    # Get service URL
    SERVICE_URL=$(kubectl get ksvc densenet-service -o jsonpath='{.status.url}')
    print_success "Service deployed successfully"
    print_status "Service URL: $SERVICE_URL"
}

# Test the service
test_service() {
    print_status "Testing the service..."
    
    # Get service URL
    SERVICE_URL=$(kubectl get ksvc densenet-service -o jsonpath='{.status.url}')
    
    if [ -z "$SERVICE_URL" ]; then
        print_error "Service URL not found"
        return 1
    fi
    
    # Test health endpoint
    print_status "Testing health endpoint..."
    curl -f "$SERVICE_URL/health" || print_warning "Health check failed"
    
    # Test predict endpoint with dummy data
    print_status "Testing predict endpoint..."
    curl -X POST "$SERVICE_URL/predict" \
      -H "Content-Type: application/json" \
      -d '{"image": "dummy_base64_data", "batch_size": 1}' \
      || print_warning "Predict endpoint test failed"
    
    print_success "Service testing completed"
}

# Display cluster information
display_info() {
    print_status "Cluster Information:"
    echo "========================"
    
    # Cluster status
    kubectl cluster-info
    
    # Service information
    echo ""
    print_status "DenseNet Service:"
    kubectl get ksvc densenet-service
    
    # Service URL
    SERVICE_URL=$(kubectl get ksvc densenet-service -o jsonpath='{.status.url}')
    if [ ! -z "$SERVICE_URL" ]; then
        echo ""
        print_status "Service Endpoints:"
        echo "  Health: $SERVICE_URL/health"
        echo "  Predict: $SERVICE_URL/predict"
    fi
    
    echo ""
    print_status "Useful Commands:"
    echo "  View logs: kubectl logs -l serving.knative.dev/service=densenet-service -c user-container"
    echo "  Scale service: kubectl scale ksvc densenet-service --replicas=3"
    echo "  Delete service: kubectl delete ksvc densenet-service"
    echo "  Delete cluster: kind delete cluster --name densenet-knative"
}

# Main execution
main() {
    print_status "Setting up Kind cluster with KNative for DenseNet serverless deployment"
    
    check_prerequisites
    create_cluster
    install_knative
    install_networking
    build_and_load_image
    deploy_service
    test_service
    display_info
    
    print_success "Kind cluster with KNative setup completed successfully!"
    print_status "Your DenseNet service is now running in a serverless environment"
}

# Run main function
main "$@"
