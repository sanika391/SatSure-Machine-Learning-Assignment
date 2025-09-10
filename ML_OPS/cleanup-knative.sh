#!/bin/bash

# Cleanup script for KNative DenseNet deployment
# This script removes all resources created during deployment

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
DELETE_CLUSTER=false
DELETE_NAMESPACE=false

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Cleanup KNative DenseNet deployment

OPTIONS:
    --service-name NAME       Service name to delete (default: densenet-service)
    --namespace NAMESPACE     Kubernetes namespace (default: default)
    --delete-namespace        Delete the entire namespace
    --delete-cluster          Delete the entire Kind cluster
    --help                    Show this help message

EXAMPLES:
    $0 --service-name my-densenet
    $0 --delete-namespace
    $0 --delete-cluster

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
    
    print_success "Prerequisites check passed"
}

# Function to delete service
delete_service() {
    print_status "Deleting DenseNet service..."
    
    # Check if service exists
    if kubectl get ksvc "$SERVICE_NAME" -n "$NAMESPACE" &> /dev/null; then
        # Delete the service
        kubectl delete ksvc "$SERVICE_NAME" -n "$NAMESPACE"
        
        # Wait for service to be deleted
        print_status "Waiting for service to be deleted..."
        kubectl wait --for=delete ksvc "$SERVICE_NAME" -n "$NAMESPACE" --timeout=60s
        
        print_success "Service deleted successfully"
    else
        print_warning "Service $SERVICE_NAME not found in namespace $NAMESPACE"
    fi
}

# Function to delete namespace
delete_namespace() {
    if [ "$DELETE_NAMESPACE" = "true" ] && [ "$NAMESPACE" != "default" ]; then
        print_status "Deleting namespace: $NAMESPACE"
        
        # Check if namespace exists
        if kubectl get namespace "$NAMESPACE" &> /dev/null; then
            # Delete the namespace
            kubectl delete namespace "$NAMESPACE"
            
            # Wait for namespace to be deleted
            print_status "Waiting for namespace to be deleted..."
            kubectl wait --for=delete namespace "$NAMESPACE" --timeout=60s
            
            print_success "Namespace deleted successfully"
        else
            print_warning "Namespace $NAMESPACE not found"
        fi
    fi
}

# Function to delete cluster
delete_cluster() {
    if [ "$DELETE_CLUSTER" = "true" ]; then
        print_status "Deleting Kind cluster..."
        
        # Check if kind is available
        if ! command -v kind &> /dev/null; then
            print_error "Kind is not installed"
            exit 1
        fi
        
        # Check if cluster exists
        if kind get clusters | grep -q "densenet-knative"; then
            # Delete the cluster
            kind delete cluster --name densenet-knative
            
            print_success "Kind cluster deleted successfully"
        else
            print_warning "Kind cluster densenet-knative not found"
        fi
    fi
}

# Function to clean up Docker images
cleanup_docker() {
    print_status "Cleaning up Docker images..."
    
    # Remove DenseNet optimization images
    if docker images | grep -q "densenet-optimization"; then
        docker rmi $(docker images | grep "densenet-optimization" | awk '{print $3}') 2>/dev/null || true
        print_success "Docker images cleaned up"
    else
        print_warning "No DenseNet optimization images found"
    fi
}

# Function to show remaining resources
show_remaining_resources() {
    print_status "Remaining resources:"
    echo "====================="
    
    # Show services
    echo "KNative Services:"
    kubectl get ksvc --all-namespaces | grep densenet || echo "  No DenseNet services found"
    
    # Show pods
    echo ""
    echo "Pods:"
    kubectl get pods --all-namespaces | grep densenet || echo "  No DenseNet pods found"
    
    # Show namespaces
    echo ""
    echo "Namespaces:"
    kubectl get namespaces | grep -E "(densenet|default)" || echo "  No relevant namespaces found"
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
        --delete-namespace)
            DELETE_NAMESPACE=true
            shift
            ;;
        --delete-cluster)
            DELETE_CLUSTER=true
            shift
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
    print_status "Starting cleanup of KNative DenseNet deployment"
    print_status "Service: $SERVICE_NAME"
    print_status "Namespace: $NAMESPACE"
    print_status "Delete namespace: $DELETE_NAMESPACE"
    print_status "Delete cluster: $DELETE_CLUSTER"
    
    check_prerequisites
    delete_service
    delete_namespace
    delete_cluster
    cleanup_docker
    show_remaining_resources
    
    print_success "Cleanup completed successfully!"
}

# Run main function
main "$@"
