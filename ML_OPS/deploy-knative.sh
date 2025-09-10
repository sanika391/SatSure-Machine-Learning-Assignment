#!/bin/bash

# Deploy DenseNet service to KNative
# This script handles the complete deployment workflow

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

# Default values
SERVICE_NAME="densenet-service"
NAMESPACE="default"
IMAGE_TAG="densenet-optimization:latest"
REPLICAS=1
MEMORY_LIMIT="4Gi"
CPU_LIMIT="2000m"

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy DenseNet service to KNative

OPTIONS:
    --service-name NAME       Service name (default: densenet-service)
    --namespace NAMESPACE     Kubernetes namespace (default: default)
    --image-tag TAG           Docker image tag (default: densenet-optimization:latest)
    --replicas NUM            Number of replicas (default: 1)
    --memory-limit SIZE       Memory limit (default: 4Gi)
    --cpu-limit SIZE          CPU limit (default: 2000m)
    --help                    Show this help message

EXAMPLES:
    $0 --service-name my-densenet --replicas 3
    $0 --memory-limit 8Gi --cpu-limit 4000m
    $0 --namespace production

EOF
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check if cluster is accessible
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check if KNative is installed
    if ! kubectl get crd services.serving.knative.dev &> /dev/null; then
        print_error "KNative Serving is not installed"
        print_status "Please run ./setup-cluster.sh first"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Function to build and push image
build_image() {
    print_status "Building Docker image..."
    
    # Build the image
    docker build -t "$IMAGE_TAG" .
    
    if [ $? -eq 0 ]; then
        print_success "Docker image built successfully"
    else
        print_error "Docker build failed"
        exit 1
    fi
    
    # Load image into Kind cluster if running locally
    if kubectl config current-context | grep -q "kind"; then
        print_status "Loading image into Kind cluster..."
        kind load docker-image "$IMAGE_TAG" --name densenet-knative
        print_success "Image loaded into Kind cluster"
    fi
}

# Function to create namespace if it doesn't exist
create_namespace() {
    if [ "$NAMESPACE" != "default" ]; then
        print_status "Creating namespace: $NAMESPACE"
        kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
        print_success "Namespace created/verified"
    fi
}

# Function to deploy the service
deploy_service() {
    print_status "Deploying DenseNet service..."
    
    # Update the service YAML with provided parameters
    cat > knative-service-deploy.yaml << EOF
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: $SERVICE_NAME
  namespace: $NAMESPACE
  annotations:
    autoscaling.knative.dev/minScale: "0"
    autoscaling.knative.dev/maxScale: "10"
    autoscaling.knative.dev/target: "100"
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "0"
        autoscaling.knative.dev/maxScale: "10"
        serving.knative.dev/execution-class: "HPA"
    spec:
      containers:
      - name: user-container
        image: $IMAGE_TAG
        ports:
        - containerPort: 8080
          protocol: TCP
        env:
        - name: MODEL_VARIANT
          value: "optimized"
        - name: BATCH_SIZE
          value: "1"
        - name: DEVICE
          value: "cuda"
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "$MEMORY_LIMIT"
            cpu: "$CPU_LIMIT"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
  traffic:
  - percent: 100
    latestRevision: true
EOF
    
    # Apply the service
    kubectl apply -f knative-service-deploy.yaml
    
    if [ $? -eq 0 ]; then
        print_success "Service deployed successfully"
    else
        print_error "Service deployment failed"
        exit 1
    fi
    
    # Clean up temporary file
    rm -f knative-service-deploy.yaml
}

# Function to wait for service to be ready
wait_for_service() {
    print_status "Waiting for service to be ready..."
    
    # Wait for service to be ready
    kubectl wait --for=condition=Ready --timeout=300s ksvc "$SERVICE_NAME" -n "$NAMESPACE"
    
    if [ $? -eq 0 ]; then
        print_success "Service is ready"
    else
        print_error "Service failed to become ready"
        exit 1
    fi
}

# Function to get service information
get_service_info() {
    print_status "Service Information:"
    echo "======================"
    
    # Get service details
    kubectl get ksvc "$SERVICE_NAME" -n "$NAMESPACE" -o wide
    
    # Get service URL
    SERVICE_URL=$(kubectl get ksvc "$SERVICE_NAME" -n "$NAMESPACE" -o jsonpath='{.status.url}')
    
    if [ ! -z "$SERVICE_URL" ]; then
        echo ""
        print_status "Service Endpoints:"
        echo "  Health: $SERVICE_URL/health"
        echo "  Ready: $SERVICE_URL/ready"
        echo "  Predict: $SERVICE_URL/predict"
        echo "  Metrics: $SERVICE_URL/metrics"
        echo "  Info: $SERVICE_URL/info"
    fi
    
    # Get pods
    echo ""
    print_status "Service Pods:"
    kubectl get pods -l serving.knative.dev/service="$SERVICE_NAME" -n "$NAMESPACE"
}

# Function to test the service
test_service() {
    print_status "Testing the service..."
    
    # Get service URL
    SERVICE_URL=$(kubectl get ksvc "$SERVICE_NAME" -n "$NAMESPACE" -o jsonpath='{.status.url}')
    
    if [ -z "$SERVICE_URL" ]; then
        print_error "Service URL not found"
        return 1
    fi
    
    # Test health endpoint
    print_status "Testing health endpoint..."
    if curl -f "$SERVICE_URL/health" &> /dev/null; then
        print_success "Health check passed"
    else
        print_warning "Health check failed"
    fi
    
    # Test info endpoint
    print_status "Testing info endpoint..."
    if curl -f "$SERVICE_URL/info" &> /dev/null; then
        print_success "Info endpoint working"
    else
        print_warning "Info endpoint failed"
    fi
    
    # Test predict endpoint with dummy data
    print_status "Testing predict endpoint..."
    DUMMY_IMAGE=$(echo -n "dummy_image_data" | base64)
    
    if curl -X POST "$SERVICE_URL/predict" \
      -H "Content-Type: application/json" \
      -d "{\"image\": \"$DUMMY_IMAGE\", \"batch_size\": 1}" \
      &> /dev/null; then
        print_success "Predict endpoint working"
    else
        print_warning "Predict endpoint test failed (expected with dummy data)"
    fi
}

# Function to show useful commands
show_commands() {
    echo ""
    print_status "Useful Commands:"
    echo "=================="
    echo "  View service: kubectl get ksvc $SERVICE_NAME -n $NAMESPACE"
    echo "  View pods: kubectl get pods -l serving.knative.dev/service=$SERVICE_NAME -n $NAMESPACE"
    echo "  View logs: kubectl logs -l serving.knative.dev/service=$SERVICE_NAME -n $NAMESPACE -c user-container"
    echo "  Scale service: kubectl scale ksvc $SERVICE_NAME --replicas=$REPLICAS -n $NAMESPACE"
    echo "  Delete service: kubectl delete ksvc $SERVICE_NAME -n $NAMESPACE"
    echo "  Port forward: kubectl port-forward svc/$SERVICE_NAME 8080:80 -n $NAMESPACE"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --service-name)
            SERVICE_NAME="$2"
            shift 2
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --image-tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --replicas)
            REPLICAS="$2"
            shift 2
            ;;
        --memory-limit)
            MEMORY_LIMIT="$2"
            shift 2
            ;;
        --cpu-limit)
            CPU_LIMIT="$2"
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
    print_status "Deploying DenseNet service to KNative"
    print_status "Service: $SERVICE_NAME"
    print_status "Namespace: $NAMESPACE"
    print_status "Image: $IMAGE_TAG"
    
    check_prerequisites
    build_image
    create_namespace
    deploy_service
    wait_for_service
    get_service_info
    test_service
    show_commands
    
    print_success "DenseNet service deployed successfully!"
}

# Run main function
main "$@"
