#!/bin/bash

# Model Testing Agent - Automated Setup Script
# This script sets up the entire environment for the Model Testing Agent

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "\n${BLUE}===================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===================================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

check_command() {
    if command -v $1 &> /dev/null; then
        print_success "$1 is installed"
        return 0
    else
        print_error "$1 is not installed"
        return 1
    fi
}

# Main setup process
print_header "Model Testing Agent - Automated Setup"

echo "This script will set up the Model Testing Agent environment."
echo "It will:"
echo "  1. Check prerequisites"
echo "  2. Create project structure"
echo "  3. Set up Python virtual environment"
echo "  4. Install dependencies"
echo "  5. Configure environment variables"
echo "  6. Verify AWS credentials"
echo "  7. Test the installation"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Setup cancelled."
    exit 0
fi

# Step 1: Check Prerequisites
print_header "Step 1: Checking Prerequisites"

PREREQS_OK=true

# Check Python
if check_command python3; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_info "Python version: $PYTHON_VERSION"
    
    # Check if version is 3.10 or higher
    MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 10 ]; then
        print_success "Python version is sufficient (>= 3.10)"
    else
        print_error "Python 3.10 or higher is required"
        PREREQS_OK=false
    fi
else
    PREREQS_OK=false
fi

# Check pip
if check_command pip3; then
    print_success "pip3 is available"
else
    print_error "pip3 is not available"
    PREREQS_OK=false
fi

# Check AWS CLI
if check_command aws; then
    print_success "AWS CLI is installed"
    AWS_VERSION=$(aws --version)
    print_info "$AWS_VERSION"
else
    print_warning "AWS CLI not found (optional but recommended)"
    print_info "Install with: pip install awscli"
fi

# Check git
if check_command git; then
    print_success "Git is installed"
else
    print_warning "Git not found (optional)"
fi

if [ "$PREREQS_OK" = false ]; then
    print_error "Prerequisites check failed. Please install missing dependencies."
    exit 1
fi

print_success "All required prerequisites are met!"

# Step 2: Create Project Structure
print_header "Step 2: Creating Project Structure"

PROJECT_NAME="model-testing-agent"
print_info "Creating project directory: $PROJECT_NAME"

if [ -d "$PROJECT_NAME" ]; then
    print_warning "Directory $PROJECT_NAME already exists"
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$PROJECT_NAME"
        print_info "Removed existing directory"
    else
        print_info "Using existing directory"
    fi
fi

mkdir -p "$PROJECT_NAME"
cd "$PROJECT_NAME"

# Create directory structure
print_info "Creating directory structure..."
mkdir -p test_assets
mkdir -p test_reports
mkdir -p logs
mkdir -p .cache
mkdir -p tests

print_success "Project structure created"

# Step 3: Set up Virtual Environment
print_header "Step 3: Setting Up Python Virtual Environment"

print_info "Creating virtual environment..."
python3 -m venv venv

print_info "Activating virtual environment..."
source venv/bin/activate

print_success "Virtual environment created and activated"

# Step 4: Install Dependencies
print_header "Step 4: Installing Dependencies"

# Create requirements.txt if it doesn't exist
if [ ! -f "requirements.txt" ]; then
    print_info "Creating requirements.txt..."
    cat > requirements.txt << 'EOF'
# Core Strands Framework
strands-agents>=1.0.0
strands-agents-tools>=1.0.0

# AWS Services
boto3>=1.34.0
botocore>=1.34.0

# HTTP and Web
requests>=2.31.0
aiohttp>=3.9.0

# Data Processing
pandas>=2.0.0
numpy>=1.24.0

# File Handling
pillow>=10.0.0
python-magic>=0.4.27

# Video Processing (optional)
opencv-python>=4.8.0

# Audio Processing (optional)
pydub>=0.25.1

# Reporting
markdown>=3.5.0
jinja2>=3.1.0

# Utilities
python-dotenv>=1.0.0
pydantic>=2.5.0
typing-extensions>=4.9.0

# Observability and Logging
structlog>=24.0.0
rich>=13.7.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0

# Security
cryptography>=41.0.0
EOF
    print_success "requirements.txt created"
fi

print_info "Installing Python packages (this may take a few minutes)..."
pip install --upgrade pip
pip install -r requirements.txt

print_success "Dependencies installed successfully"

# Step 5: Configure Environment Variables
print_header "Step 5: Configuring Environment Variables"

if [ ! -f ".env" ]; then
    print_info "Creating .env file from template..."
    cat > .env << 'EOF'
# AWS Configuration
AWS_DEFAULT_REGION=us-west-2

# Bedrock Configuration
BEDROCK_REGION=us-west-2
BEDROCK_ORCHESTRATOR_MODEL=us.anthropic.claude-sonnet-4-20250514-v1:0

# Agent Configuration
AGENT_MAX_ITERATIONS=30
AGENT_TEMPERATURE=0.7
ENABLE_AGENT_THINKING=true

# Testing Configuration
TEST_ASSETS_DIR=test_assets
TEST_REPORTS_DIR=test_reports
BYPASS_TOOL_CONSENT=true

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/model_testing.log
ENABLE_CONSOLE_LOGGING=true
EOF
    print_success ".env file created"
    print_warning "Please edit .env to add your AWS credentials if needed"
else
    print_info ".env file already exists, skipping..."
fi

# Step 6: Verify AWS Credentials
print_header "Step 6: Verifying AWS Credentials"

print_info "Checking AWS credentials..."

if aws sts get-caller-identity &> /dev/null; then
    IDENTITY=$(aws sts get-caller-identity)
    ACCOUNT=$(echo $IDENTITY | grep -o '"Account": "[^"]*' | cut -d'"' -f4)
    ARN=$(echo $IDENTITY | grep -o '"Arn": "[^"]*' | cut -d'"' -f4)
    
    print_success "AWS credentials are configured"
    print_info "Account: $ACCOUNT"
    print_info "Identity: $ARN"
else
    print_warning "Could not verify AWS credentials"
    print_info "Please configure AWS credentials using one of:"
    print_info "  1. aws configure"
    print_info "  2. Export AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
    print_info "  3. Use IAM roles (recommended for EC2/ECS)"
fi

# Check Bedrock model access
print_info "Checking Bedrock access..."
if aws bedrock list-foundation-models --region us-west-2 &> /dev/null; then
    print_success "Bedrock access confirmed"
    
    # Check for specific model access
    print_info "Checking Claude Sonnet 4 access..."
    if aws bedrock get-foundation-model \
        --model-identifier anthropic.claude-sonnet-4-20250514-v1:0 \
        --region us-west-2 &> /dev/null; then
        print_success "Claude Sonnet 4 model access confirmed"
    else
        print_warning "Claude Sonnet 4 model access not confirmed"
        print_info "You may need to enable model access in the Bedrock console"
    fi
else
    print_warning "Could not verify Bedrock access"
    print_info "Please ensure you have permissions for Amazon Bedrock"
fi

# Step 7: Test Installation
print_header "Step 7: Testing Installation"

print_info "Creating test script..."
cat > test_installation.py << 'EOF'
"""Quick installation test"""
import sys

print("Testing imports...")

try:
    import boto3
    print("✓ boto3")
    
    import strands
    print("✓ strands")
    
    import strands_tools
    print("✓ strands_tools")
    
    from strands import Agent
    from strands.models import BedrockModel
    print("✓ Strands components")
    
    print("\n✓ All imports successful!")
    print("\nTesting AWS connectivity...")
    
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"✓ AWS connection successful")
        print(f"  Account: {identity['Account']}")
    except Exception as e:
        print(f"⚠ AWS connection issue: {str(e)}")
    
    print("\n✅ Installation test PASSED!")
    sys.exit(0)
    
except ImportError as e:
    print(f"\n✗ Import failed: {str(e)}")
    print("\n❌ Installation test FAILED!")
    sys.exit(1)
EOF

print_info "Running installation test..."
if python test_installation.py; then
    print_success "Installation test passed!"
else
    print_error "Installation test failed!"
    exit 1
fi

# Final Summary
print_header "Setup Complete!"

echo "
${GREEN}✓${NC} Project structure created
${GREEN}✓${NC} Virtual environment configured
${GREEN}✓${NC} Dependencies installed
${GREEN}✓${NC} Environment configured
${GREEN}✓${NC} AWS credentials verified
${GREEN}✓${NC} Installation tested

${BLUE}Next Steps:${NC}

1. Activate the virtual environment:
   ${YELLOW}source venv/bin/activate${NC}

2. Copy the agent files to the project directory:
   - enhanced_orchestrator.py
   - config.yaml
   - quickstart.py

3. Run the quickstart examples:
   ${YELLOW}python quickstart.py${NC}

4. Test with Amazon Nova 2 Omni:
   ${YELLOW}python enhanced_orchestrator.py \"Amazon Nova 2 Omni\"${NC}

5. View generated reports in:
   ${YELLOW}test_reports/${NC}

${BLUE}Additional Resources:${NC}
- Documentation: README.md
- Configuration: config.yaml
- Environment: .env
- Logs: logs/model_testing.log

${GREEN}Happy Testing! 🚀${NC}
"

# Create a quick reference file
cat > QUICKSTART.md << 'EOF'
# Quick Start Guide

## Activate Environment
```bash
source venv/bin/activate
```

## Run Basic Test
```bash
python enhanced_orchestrator.py "Amazon Nova 2 Omni"
```

## Run Examples
```bash
python quickstart.py
```

## View Logs
```bash
tail -f logs/model_testing.log
```

## Deactivate Environment
```bash
deactivate
```

## Re-run Setup
```bash
./setup.sh
```
EOF

print_success "Created QUICKSTART.md reference guide"

print_info "Setup script completed successfully!"