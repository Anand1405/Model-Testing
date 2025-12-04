# AI Model Testing Agent - Production Deployment Guide

A comprehensive, production-ready multi-agent system built with Strands framework for automated AI model testing on AWS Bedrock.

## 🚀 Features

- **Multi-Agent Architecture**: Specialized agents for research, planning, execution, analysis, and reporting
- **Comprehensive Testing**: Tests all modalities (text, image, video, audio, documents)
- **AWS Bedrock Integration**: Native integration with AWS Bedrock for model invocation
- **Extended Reasoning**: Leverages Claude Sonnet 4's thinking capabilities for better results
- **Automated Asset Management**: Downloads and manages test assets automatically
- **Professional Reporting**: Generates detailed JSON and Markdown reports
- **Production-Ready**: Full error handling, logging, and observability
- **Configurable**: YAML-based configuration for easy customization

## 📋 Prerequisites

- Python 3.10 or higher
- AWS Account with Bedrock access
- AWS CLI configured with appropriate credentials
- Enabled model access in Amazon Bedrock console

### Required AWS IAM Permissions

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream",
        "bedrock:GetFoundationModel",
        "bedrock:ListFoundationModels"
      ],
      "Resource": "*"
    }
  ]
}
```

## 📦 Installation

### 1. Clone and Setup Environment

```bash
# Create project directory
mkdir model-testing-agent
cd model-testing-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure AWS Credentials

**Option A: Environment Variables**
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-west-2
```

**Option B: AWS CLI Configuration**
```bash
aws configure
```

**Option C: IAM Roles (Recommended for EC2/ECS)**
- Attach appropriate IAM role to your compute instance

### 3. Enable Model Access in Bedrock

1. Open AWS Console → Amazon Bedrock
2. Navigate to "Model access"
3. Request access for:
   - Claude 4 Sonnet (orchestrator brain)
   - Amazon Nova 2 Omni (or your target model)
4. Wait for approval (usually immediate)

### 4. Configure the Agent

Edit `config.yaml` to customize:
- AWS region
- Model IDs
- Test parameters
- Asset sources
- Reporting options

## 🎯 Usage

### Basic Usage

```bash
# Test Amazon Nova 2 Omni
python enhanced_orchestrator.py "Amazon Nova 2 Omni"

# Test with specific region
python enhanced_orchestrator.py "Amazon Nova 2 Omni" --region us-east-1
```

### Python API Usage

```python
from enhanced_orchestrator import ModelTestingCoordinator

# Initialize coordinator
coordinator = ModelTestingCoordinator(region="us-west-2")

# Run comprehensive testing
results = coordinator.test_model("Amazon Nova 2 Omni")

# Access results
print(f"Status: {results['status']}")
print(f"Reports: {results['reports']}")
```

### Async Usage

```python
import asyncio
from enhanced_orchestrator import ModelTestingCoordinator

async def main():
    coordinator = ModelTestingCoordinator()
    results = await coordinator.test_model_async("Amazon Nova 2 Omni")
    return results

results = asyncio.run(main())
```

## 📊 Output Structure

### Generated Files

```
project/
├── test_assets/          # Downloaded test files
│   ├── test_image.jpg
│   ├── test_video.mp4
│   ├── test_audio.wav
│   └── test_document.pdf
├── test_reports/         # Generated reports
│   ├── model_test_report_20241203_143022.json
│   └── model_test_report_20241203_143022.md
└── logs/                 # Execution logs
    └── model_testing.log
```

### JSON Report Structure

```json
{
  "model_name": "Amazon Nova 2 Omni",
  "model_id": "us.amazon.nova-omni-v1:0",
  "test_date": "2024-12-03T14:30:22",
  "test_summary": {
    "total_tests": 15,
    "passed": 13,
    "failed": 2,
    "success_rate": 86.67
  },
  "capabilities": {
    "input_modalities": ["text", "image", "video", "audio"],
    "output_modalities": ["text", "image"]
  },
  "test_results": [...],
  "analysis": {...},
  "recommendations": [...]
}
```

## 🏗️ Architecture

### Multi-Agent Design

```
┌─────────────────────────────────────────────┐
│         Orchestrator Agent                   │
│      (Claude Sonnet 4 with Thinking)        │
└─────────────┬───────────────────────────────┘
              │
    ┌─────────┼─────────┬──────────┬──────────┐
    │         │         │          │          │
┌───▼───┐ ┌──▼──┐ ┌────▼────┐ ┌──▼───┐ ┌────▼────┐
│Research│ │Plan │ │ Execute │ │Analyze│ │ Report  │
│ Agent  │ │Agent│ │  Agent  │ │ Agent │ │  Agent  │
└────────┘ └─────┘ └─────────┘ └───────┘ └─────────┘
     │         │         │          │          │
     └─────────┴─────────┴──────────┴──────────┘
                         │
                    ┌────▼─────┐
                    │  Bedrock │
                    │  Runtime │
                    └──────────┘
```

### Agent Responsibilities

1. **Research Agent**: Discovers model capabilities and documentation
2. **Test Planner Agent**: Creates comprehensive test plans
3. **Execution Agent**: Runs tests across all modalities
4. **Analysis Agent**: Analyzes results and generates insights
5. **Report Agent**: Creates professional reports

### Execution Flow

```
1. Model Research
   ├── Search documentation
   ├── Identify capabilities
   └── Map modalities

2. Test Planning
   ├── Generate test cases
   ├── Prioritize tests
   └── Define success criteria

3. Asset Preparation
   ├── Download test images
   ├── Download test videos
   ├── Download test audio
   └── Download test documents

4. Test Execution
   ├── Text generation tests
   ├── Image understanding tests
   ├── Video analysis tests
   ├── Audio processing tests
   └── Document parsing tests

5. Result Analysis
   ├── Calculate metrics
   ├── Identify patterns
   ├── Generate insights
   └── Create recommendations

6. Report Generation
   ├── JSON report (machine-readable)
   └── Markdown report (human-readable)
```

## 🔧 Configuration

### Model Configuration

Add new models to `config.yaml`:

```yaml
model_configs:
  your_custom_model:
    model_id: your.model.id
    region: us-west-2
    input_modalities:
      - text
      - image
    output_modalities:
      - text
    max_tokens: 4096
    supports_streaming: true
    supports_tool_use: false
    supports_extended_thinking: false
```

### Test Customization

Modify test cases in `config.yaml`:

```yaml
testing:
  test_cases:
    custom_modality:
      - name: custom_test
        description: Your custom test
        priority: high
```

## 🚀 Deployment Options

### Option 1: Local Development

```bash
python enhanced_orchestrator.py "Amazon Nova 2 Omni"
```

### Option 2: AWS EC2

```bash
# 1. Launch EC2 instance with appropriate IAM role
# 2. Install dependencies
# 3. Run agent

ssh ec2-user@your-instance
sudo yum install python3.11
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python enhanced_orchestrator.py "Amazon Nova 2 Omni"
```

### Option 3: AWS Lambda

```python
# lambda_handler.py
import json
from enhanced_orchestrator import ModelTestingCoordinator

def lambda_handler(event, context):
    model_name = event.get('model_name', 'Amazon Nova 2 Omni')
    
    coordinator = ModelTestingCoordinator()
    results = coordinator.test_model(model_name)
    
    return {
        'statusCode': 200,
        'body': json.dumps(results)
    }
```

Package and deploy:
```bash
# Create deployment package
pip install -r requirements.txt -t package/
cp enhanced_orchestrator.py package/
cd package && zip -r ../lambda.zip . && cd ..

# Deploy with AWS CLI
aws lambda create-function \
  --function-name model-testing-agent \
  --runtime python3.11 \
  --handler lambda_handler.lambda_handler \
  --zip-file fileb://lambda.zip \
  --role arn:aws:iam::YOUR_ACCOUNT:role/lambda-bedrock-role \
  --timeout 900 \
  --memory-size 2048
```

### Option 4: Amazon ECS/Fargate

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "enhanced_orchestrator.py", "Amazon Nova 2 Omni"]
```

Build and deploy:
```bash
# Build image
docker build -t model-testing-agent .

# Push to ECR
aws ecr create-repository --repository-name model-testing-agent
docker tag model-testing-agent:latest ${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/model-testing-agent:latest
docker push ${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/model-testing-agent:latest

# Deploy to ECS (use AWS Console or CLI)
```

### Option 5: Bedrock AgentCore Runtime

```python
# agentcore_app.py
from strands import Agent
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from enhanced_orchestrator import ModelTestingCoordinator

app = BedrockAgentCoreApp()

@app.entrypoint
def invoke(payload):
    model_name = payload.get("model_name", "Amazon Nova 2 Omni")
    
    coordinator = ModelTestingCoordinator()
    results = coordinator.test_model(model_name)
    
    return results

if __name__ == "__main__":
    app.run()
```

Deploy:
```bash
# Configure
agentcore configure --entrypoint agentcore_app.py

# Deploy to AWS
agentcore launch

# Test
agentcore invoke '{"model_name": "Amazon Nova 2 Omni"}'
```

## 📈 Monitoring and Observability

### Logging

The agent provides comprehensive logging:

```python
# View logs
tail -f logs/model_testing.log

# Structured logging output
{
  "timestamp": "2024-12-03T14:30:22",
  "level": "INFO",
  "agent": "ResearchAgent",
  "message": "Starting model research",
  "trace_id": "abc-123-def"
}
```

### Metrics

Key metrics tracked:
- Test execution time
- Success/failure rates
- Model response times
- Token usage
- Cost estimation

### CloudWatch Integration

```python
# Add CloudWatch logging
import boto3

cloudwatch = boto3.client('cloudwatch')

def publish_metrics(test_results):
    cloudwatch.put_metric_data(
        Namespace='ModelTesting',
        MetricData=[
            {
                'MetricName': 'TestsPassed',
                'Value': test_results['passed'],
                'Unit': 'Count'
            },
            {
                'MetricName': 'TestDuration',
                'Value': test_results['duration'],
                'Unit': 'Seconds'
            }
        ]
    )
```

## 🔒 Security Best Practices

1. **IAM Roles**: Use IAM roles instead of access keys
2. **Least Privilege**: Grant only required permissions
3. **Encryption**: Enable encryption at rest and in transit
4. **Secrets Management**: Use AWS Secrets Manager for credentials
5. **VPC**: Deploy in private subnets when possible
6. **Logging**: Enable CloudTrail for audit logs

## 🧪 Testing the Agent Itself

```bash
# Run unit tests
pytest tests/

# Run integration tests
pytest tests/integration/

# Run with coverage
pytest --cov=enhanced_orchestrator tests/
```

## 🐛 Troubleshooting

### Common Issues

**Issue: Model Access Denied**
```
Solution: Enable model access in Bedrock console
```

**Issue: Timeout Errors**
```
Solution: Increase timeout in config.yaml or code
```

**Issue: Asset Download Fails**
```
Solution: Check internet connectivity and firewall rules
```

**Issue: Out of Memory**
```
Solution: Increase Lambda memory or EC2 instance size
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python enhanced_orchestrator.py "Amazon Nova 2 Omni"
```

## 📚 Examples

### Testing Multiple Models

```python
models_to_test = [
    "Amazon Nova 2 Omni",
    "Claude Sonnet 4",
    "Amazon Nova Lite",
    "Amazon Nova Pro"
]

coordinator = ModelTestingCoordinator()

for model in models_to_test:
    print(f"\nTesting {model}...")
    results = coordinator.test_model(model)
    print(f"Completed: {results['status']}")
```

### Custom Test Cases

```python
from enhanced_orchestrator import TestCase, ModalityType, TestStatus

custom_test = TestCase(
    test_id="T999",
    name="Custom Reasoning Test",
    description="Test advanced mathematical reasoning",
    modality=ModalityType.TEXT,
    status=TestStatus.PENDING
)

coordinator.test_cases.append(custom_test)
```

### Batch Testing

```bash
# Create a batch script
cat > batch_test.sh << 'EOF'
#!/bin/bash
for model in "Amazon Nova 2 Omni" "Claude Sonnet 4"
do
    python enhanced_orchestrator.py "$model" --region us-west-2
    sleep 10
done
EOF

chmod +x batch_test.sh
./batch_test.sh
```

## 🤝 Contributing

This is a production-ready template. Customize for your needs:

1. Add new test cases in `config.yaml`
2. Extend agent capabilities in `enhanced_orchestrator.py`
3. Add new modality support
4. Enhance reporting formats
5. Integrate with your CI/CD pipeline

## 📄 License

This is a production template for AWS Bedrock model testing. Customize as needed for your organization.

## 🆘 Support

- AWS Documentation: https://docs.aws.amazon.com/bedrock/
- Strands Documentation: https://strandsagents.com/latest/
- Create issues for bugs or feature requests

## 🎯 Roadmap

- [ ] Add visual test result dashboards
- [ ] Support for more output modalities
- [ ] Integration with MLOps pipelines
- [ ] A/B testing capabilities
- [ ] Cost optimization recommendations
- [ ] Automated model comparison reports
- [ ] Real-time monitoring dashboard

---

**Built with ❤️ using Strands Framework and AWS Bedrock**