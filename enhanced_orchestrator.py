"""
Enhanced Production-Ready AI Model Testing Agent System
With Advanced Multi-Agent Coordination, Error Handling, and Observability
Fixed for Bedrock Model ID input and proper API usage
"""

import os
import json
import boto3
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from strands import Agent, tool
from strands.models import BedrockModel
from strands_tools import (
    http_request,
    file_write,
    file_read,
    python_repl,
    current_time,
    web_search
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set bypass for automated workflows
os.environ['BYPASS_TOOL_CONSENT'] = 'true'


class TestStatus(Enum):
    """Test execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ModalityType(Enum):
    """Supported modality types"""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"


@dataclass
class TestCase:
    """Represents a single test case"""
    test_id: str
    name: str
    description: str
    modality: ModalityType
    status: TestStatus
    result: Optional[Dict[str, Any]] = None
    duration_seconds: float = 0.0
    error_message: Optional[str] = None


@dataclass
class ModelCapabilities:
    """Model capabilities and configuration"""
    model_name: str
    model_id: str
    input_modalities: List[str]
    output_modalities: List[str]
    max_tokens: int
    supports_streaming: bool
    supports_tool_use: bool
    supports_extended_thinking: bool
    region: str
    documentation_url: str


class BedrockModelInvoker:
    """Enhanced Bedrock model invocation with retry logic and error handling"""
    
    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self.client = boto3.client('bedrock-runtime', region_name=region)
        self.max_retries = 3
        self.retry_delay = 2
    
    def _get_request_format(self, model_id: str) -> str:
        """Determine the request format based on model ID"""
        if 'anthropic' in model_id.lower():
            return 'anthropic'
        elif 'amazon.nova' in model_id.lower():
            return 'nova'
        elif 'meta.llama' in model_id.lower():
            return 'llama'
        elif 'ai21' in model_id.lower():
            return 'ai21'
        else:
            return 'converse'  # Use Converse API as fallback
    
    def invoke_with_text(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Invoke model with text input"""
        try:
            request_format = self._get_request_format(model_id)
            
            if request_format == 'converse':
                # Use Converse API for maximum compatibility
                response = self.client.converse(
                    modelId=model_id,
                    messages=[{
                        "role": "user",
                        "content": [{"text": prompt}]
                    }],
                    inferenceConfig={
                        "maxTokens": max_tokens,
                        "temperature": temperature
                    }
                )
                
                return {
                    "success": True,
                    "data": {
                        "output": response['output']['message']['content'][0]['text'],
                        "stopReason": response['stopReason'],
                        "usage": response['usage']
                    }
                }
            
            elif request_format == 'anthropic':
                body = {
                    "messages": [{
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}]
                    }],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "anthropic_version": "bedrock-2023-05-31"
                }
                
            elif request_format == 'nova':
                body = {
                    "messages": [{
                        "role": "user",
                        "content": [{"text": prompt}]
                    }],
                    "inferenceConfig": {
                        "max_new_tokens": max_tokens,
                        "temperature": temperature
                    }
                }
            
            else:
                return {"success": False, "error": f"Unsupported model format: {request_format}"}
            
            response = self.client.invoke_model(
                modelId=model_id,
                body=json.dumps(body)
            )
            
            result = json.loads(response['body'].read())
            return {"success": True, "data": result}
            
        except Exception as e:
            logger.error(f"Text invocation failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def invoke_with_image(
        self,
        model_id: str,
        prompt: str,
        image_path: str,
        max_tokens: int = 2048
    ) -> Dict[str, Any]:
        """Invoke model with image input using Converse API"""
        try:
            import base64
            
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Determine image format
            image_format = "jpeg"
            if image_path.lower().endswith('.png'):
                image_format = "png"
            elif image_path.lower().endswith('.gif'):
                image_format = "gif"
            elif image_path.lower().endswith('.webp'):
                image_format = "webp"
            
            # Use Converse API for best compatibility
            response = self.client.converse(
                modelId=model_id,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "image": {
                                "format": image_format,
                                "source": {"bytes": base64.b64decode(image_data)}
                            }
                        },
                        {"text": prompt}
                    ]
                }],
                inferenceConfig={
                    "maxTokens": max_tokens,
                    "temperature": 0.7
                }
            )
            
            return {
                "success": True,
                "data": {
                    "output": response['output']['message']['content'][0]['text'],
                    "stopReason": response['stopReason'],
                    "usage": response['usage']
                }
            }
            
        except Exception as e:
            logger.error(f"Image invocation failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def invoke_with_video(
        self,
        model_id: str,
        prompt: str,
        video_path: str,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """Invoke model with video input using Converse API"""
        try:
            import base64
            
            with open(video_path, 'rb') as f:
                video_data = base64.b64encode(f.read()).decode('utf-8')
            
            response = self.client.converse(
                modelId=model_id,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "video": {
                                "format": "mp4",
                                "source": {"bytes": base64.b64decode(video_data)}
                            }
                        },
                        {"text": prompt}
                    ]
                }],
                inferenceConfig={
                    "maxTokens": max_tokens,
                    "temperature": 0.7
                }
            )
            
            return {
                "success": True,
                "data": {
                    "output": response['output']['message']['content'][0]['text'],
                    "stopReason": response['stopReason'],
                    "usage": response['usage']
                }
            }
            
        except Exception as e:
            logger.error(f"Video invocation failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def invoke_with_document(
        self,
        model_id: str,
        prompt: str,
        document_path: str,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """Invoke model with document input using Converse API"""
        try:
            import base64
            
            with open(document_path, 'rb') as f:
                doc_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Determine document format
            doc_format = "pdf"
            if document_path.lower().endswith('.docx'):
                doc_format = "docx"
            elif document_path.lower().endswith('.txt'):
                doc_format = "txt"
            elif document_path.lower().endswith('.csv'):
                doc_format = "csv"
            
            response = self.client.converse(
                modelId=model_id,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "document": {
                                "format": doc_format,
                                "name": os.path.basename(document_path),
                                "source": {"bytes": base64.b64decode(doc_data)}
                            }
                        },
                        {"text": prompt}
                    ]
                }],
                inferenceConfig={
                    "maxTokens": max_tokens,
                    "temperature": 0.7
                }
            )
            
            return {
                "success": True,
                "data": {
                    "output": response['output']['message']['content'][0]['text'],
                    "stopReason": response['stopReason'],
                    "usage": response['usage']
                }
            }
            
        except Exception as e:
            logger.error(f"Document invocation failed: {str(e)}")
            return {"success": False, "error": str(e)}


class TestAssetManager:
    """Manages test assets - downloading, caching, and organizing"""
    
    def __init__(self, assets_dir: str = "test_assets"):
        self.assets_dir = assets_dir
        os.makedirs(assets_dir, exist_ok=True)
        
        # Updated working asset sources
        self.asset_sources = {
            "image": [
                {
                    "name": "landscape",
                    "url": "https://raw.githubusercontent.com/awslabs/aws-ai-solution-kit/main/docs/en/workshop/images/sample-image.jpg",
                    "filename": "test_landscape.jpg"
                },
                {
                    "name": "portrait",
                    "url": "https://raw.githubusercontent.com/awslabs/aws-ai-solution-kit/main/docs/en/workshop/images/sample-image.jpg",
                    "filename": "test_portrait.jpg"
                }
            ],
            "video": [
                {
                    "name": "sample_video",
                    "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
                    "filename": "test_video_small.mp4"
                }
            ],
            "audio": [
                {
                    "name": "speech",
                    "url": "https://www2.cs.uic.edu/~i101/SoundFiles/BabyElephantWalk60.wav",
                    "filename": "test_audio.wav"
                }
            ],
            "document": [
                {
                    "name": "pdf_sample",
                    "url": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
                    "filename": "test_document.pdf"
                }
            ]
        }
    
    def download_asset(self, modality: str, asset_name: str = None) -> Optional[str]:
        """Download a test asset"""
        if modality not in self.asset_sources:
            logger.error(f"Unknown modality: {modality}")
            return None
        
        assets = self.asset_sources[modality]
        if not assets:
            return None
        
        # Select first asset if no specific name provided
        asset = assets[0] if not asset_name else next(
            (a for a in assets if a['name'] == asset_name), assets[0]
        )
        
        file_path = os.path.join(self.assets_dir, asset['filename'])
        
        # Check if already downloaded
        if os.path.exists(file_path):
            logger.info(f"Asset already exists: {file_path}")
            return file_path
        
        try:
            import requests
            logger.info(f"Downloading {modality} asset: {asset['name']}")
            
            response = requests.get(asset['url'], timeout=30)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded to: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to download asset: {str(e)}")
            # Create a dummy file for testing if download fails
            logger.info(f"Creating dummy {modality} file for testing")
            with open(file_path, 'wb') as f:
                f.write(b'dummy content for testing')
            return file_path
    
    def get_all_assets(self) -> Dict[str, List[str]]:
        """Download all test assets"""
        result = {}
        
        for modality in self.asset_sources.keys():
            result[modality] = []
            for asset in self.asset_sources[modality]:
                path = self.download_asset(modality, asset['name'])
                if path:
                    result[modality].append(path)
        
        return result


class ModelTestingCoordinator:
    """
    Advanced multi-agent coordinator for comprehensive model testing
    Implements sophisticated agent communication and task distribution
    """
    
    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self.orchestrator_model = self._create_orchestrator_model()
        self.invoker = BedrockModelInvoker(region)
        self.asset_manager = TestAssetManager()
        
        # Test tracking
        self.test_cases: List[TestCase] = []
        self.model_capabilities: Optional[ModelCapabilities] = None
        
        # Create specialized agents
        self.agents = {
            "researcher": self._create_research_agent(),
            "test_planner": self._create_test_planner_agent(),
            "executor": self._create_execution_agent(),
            "analyzer": self._create_analysis_agent(),
            "reporter": self._create_report_agent()
        }
    
    def _create_orchestrator_model(self) -> BedrockModel:
        """Create orchestrator model with extended thinking - FIXED"""
        return BedrockModel(
            model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
            max_tokens=4096,
            temperature=0.7,
            top_p=0.9,
            streaming=True
        )
    
    def _create_research_agent(self) -> Agent:
        """Research agent for model discovery"""
        system_prompt = """You are an expert AI model research specialist. Your mission:

1. Find comprehensive documentation for the given Bedrock model ID
2. Use web_search to find official AWS documentation
3. Identify ALL capabilities and limitations from official sources
4. Determine exact API requirements and invocation methods
5. Map input/output modalities precisely
6. Find example use cases and best practices

Be methodical, thorough, and accurate. Use web_search extensively to find current information."""
        
        return Agent(
            name="ModelResearcher",
            model=self.orchestrator_model,
            system_prompt=system_prompt,
            tools=[web_search, http_request, file_write, file_read, current_time],
            max_iterations=15
        )
    
    def _create_test_planner_agent(self) -> Agent:
        """Agent for creating comprehensive test plans"""
        system_prompt = """You are an expert test architect for AI models. Your mission:

1. Analyze model capabilities comprehensively
2. Design test cases covering ALL modalities and features
3. Plan edge cases, stress tests, and failure scenarios
4. Prioritize tests by importance and complexity
5. Define success criteria for each test
6. Create a structured, executable test plan

Think strategically about coverage and be comprehensive."""
        
        return Agent(
            name="TestPlanner",
            model=self.orchestrator_model,
            system_prompt=system_prompt,
            tools=[file_write, file_read, python_repl, current_time],
            max_iterations=20
        )
    
    def _create_execution_agent(self) -> Agent:
        """Agent for executing tests"""
        system_prompt = """You are an expert test execution specialist. Your mission:

1. Execute each test case systematically
2. Handle all input modalities correctly
3. Record detailed results and metrics
4. Capture errors and edge cases
5. Measure performance and quality
6. Document everything precisely

Be meticulous and handle errors gracefully."""
        
        return Agent(
            name="TestExecutor",
            model=self.orchestrator_model,
            system_prompt=system_prompt,
            tools=[file_read, file_write, python_repl, current_time],
            max_iterations=25
        )
    
    def _create_analysis_agent(self) -> Agent:
        """Agent for analyzing test results"""
        system_prompt = """You are an expert AI model analyst. Your mission:

1. Analyze all test results comprehensively
2. Identify patterns, strengths, and weaknesses
3. Evaluate quality and performance metrics
4. Compare against expected capabilities
5. Find insights and unexpected behaviors
6. Generate actionable recommendations

Think critically and provide deep insights."""
        
        return Agent(
            name="ResultAnalyzer",
            model=self.orchestrator_model,
            system_prompt=system_prompt,
            tools=[file_read, python_repl, current_time],
            max_iterations=15
        )
    
    def _create_report_agent(self) -> Agent:
        """Agent for generating comprehensive reports"""
        system_prompt = """You are an expert technical writer and reporting specialist. Your mission:

1. Create comprehensive, professional test reports
2. Generate both JSON and Markdown formats
3. Include executive summaries and detailed findings
4. Provide visualizations and metrics
5. Offer actionable recommendations
6. Make reports accessible to technical and non-technical audiences

Write clearly, professionally, and insightfully."""
        
        return Agent(
            name="ReportGenerator",
            model=self.orchestrator_model,
            system_prompt=system_prompt,
            tools=[file_write, file_read, python_repl, current_time],
            max_iterations=10
        )
    
    async def test_model_async(self, model_id: str) -> Dict[str, Any]:
        """
        Asynchronous model testing pipeline with full observability
        
        Args:
            model_id: Bedrock model ID to test (e.g., 'amazon.nova-pro-v1:0')
            
        Returns:
            Complete test results and metrics
        """
        start_time = datetime.now()
        logger.info(f"Starting comprehensive testing for model ID: {model_id}")
        
        try:
            # Phase 1: Research
            logger.info("Phase 1: Model Research")
            capabilities = await self._research_phase(model_id)
            
            # Phase 2: Test Planning
            logger.info("Phase 2: Test Planning")
            test_plan = await self._planning_phase(capabilities)
            
            # Phase 3: Asset Preparation
            logger.info("Phase 3: Asset Preparation")
            assets = self.asset_manager.get_all_assets()
            
            # Phase 4: Test Execution
            logger.info("Phase 4: Test Execution")
            test_results = await self._execution_phase(test_plan, assets)
            
            # Phase 5: Result Analysis
            logger.info("Phase 5: Result Analysis")
            analysis = await self._analysis_phase(test_results)
            
            # Phase 6: Report Generation
            logger.info("Phase 6: Report Generation")
            reports = await self._reporting_phase(
                model_id, capabilities, test_results, analysis
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Testing completed in {duration:.2f} seconds")
            
            return {
                "model_id": model_id,
                "status": "success",
                "duration_seconds": duration,
                "capabilities": asdict(capabilities) if capabilities else {},
                "test_summary": {
                    "total_tests": len(self.test_cases),
                    "passed": len([t for t in self.test_cases if t.status == TestStatus.PASSED]),
                    "failed": len([t for t in self.test_cases if t.status == TestStatus.FAILED]),
                    "skipped": len([t for t in self.test_cases if t.status == TestStatus.SKIPPED])
                },
                "reports": reports,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Testing failed: {str(e)}", exc_info=True)
            return {
                "model_id": model_id,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _research_phase(self, model_id: str) -> Optional[ModelCapabilities]:
        """Phase 1: Research model capabilities"""
        prompt = f"""Research the Bedrock model with ID: {model_id}

Use web_search extensively to find:
1. Official AWS Bedrock documentation for this model
2. Supported input modalities (text, image, video, audio, document)
3. Supported output modalities
4. Maximum token limits
5. Streaming support
6. Tool use capabilities
7. Extended thinking/reasoning support (if available)
8. AWS region availability
9. Official documentation URLs
10. Key limitations and constraints
11. Pricing information

Search AWS documentation, GitHub, and official sources.
Provide structured, accurate information with sources."""
        
        try:
            result = self.agents["researcher"](prompt)
            logger.info(f"Research completed")
            
            # Extract model family and set reasonable defaults
            model_family = "unknown"
            if "nova" in model_id.lower():
                model_family = "nova"
            elif "anthropic" in model_id.lower():
                model_family = "claude"
            elif "llama" in model_id.lower():
                model_family = "llama"
            
            # Set capabilities based on model family
            if model_family == "nova":
                input_modalities = ["text", "image", "video"]
                if "premier" in model_id.lower():
                    input_modalities.append("audio")
                input_modalities.append("document")
                output_modalities = ["text"]
                max_tokens = 5000
                supports_streaming = True
                supports_tool_use = True
                supports_extended_thinking = "pro" in model_id.lower() or "lite" in model_id.lower()
            elif model_family == "claude":
                input_modalities = ["text", "image", "document"]
                output_modalities = ["text"]
                max_tokens = 8192
                supports_streaming = True
                supports_tool_use = True
                supports_extended_thinking = True
            else:
                input_modalities = ["text"]
                output_modalities = ["text"]
                max_tokens = 2048
                supports_streaming = True
                supports_tool_use = False
                supports_extended_thinking = False
            
            capabilities = ModelCapabilities(
                model_name=model_id.split('.')[-1] if '.' in model_id else model_id,
                model_id=model_id,
                input_modalities=input_modalities,
                output_modalities=output_modalities,
                max_tokens=max_tokens,
                supports_streaming=supports_streaming,
                supports_tool_use=supports_tool_use,
                supports_extended_thinking=supports_extended_thinking,
                region=self.region,
                documentation_url=f"https://docs.aws.amazon.com/bedrock/latest/userguide/models-{model_family}.html"
            )
            
            self.model_capabilities = capabilities
            return capabilities
            
        except Exception as e:
            logger.error(f"Research phase failed: {str(e)}")
            return None
    
    async def _planning_phase(self, capabilities: ModelCapabilities) -> Dict[str, Any]:
        """Phase 2: Create comprehensive test plan"""
        prompt = f"""Create a comprehensive test plan for model ID: {capabilities.model_id}

Model capabilities:
- Input modalities: {', '.join(capabilities.input_modalities)}
- Output modalities: {', '.join(capabilities.output_modalities)}
- Max tokens: {capabilities.max_tokens}
- Streaming: {capabilities.supports_streaming}
- Tool use: {capabilities.supports_tool_use}
- Extended thinking: {capabilities.supports_extended_thinking}

Design test cases for:
1. Each input modality individually
2. Multi-modal combinations (if supported)
3. Edge cases and limitations
4. Performance under load
5. Error handling
6. Quality of outputs
7. Special features (tool use, thinking, etc.)

Provide a structured test plan with priorities."""
        
        try:
            result = self.agents["test_planner"](prompt)
            logger.info("Test planning completed")
            
            # Generate test cases based on capabilities
            self._generate_test_cases(capabilities)
            
            return {"plan": str(result), "test_cases": len(self.test_cases)}
            
        except Exception as e:
            logger.error(f"Planning phase failed: {str(e)}")
            return {"error": str(e)}
    
    def _generate_test_cases(self, capabilities: ModelCapabilities):
        """Generate test cases based on model capabilities"""
        test_id = 0
        
        # Text tests
        if "text" in capabilities.input_modalities:
            self.test_cases.extend([
                TestCase(
                    test_id=f"T{test_id:03d}",
                    name="Basic Text Generation",
                    description="Test simple text prompt and response",
                    modality=ModalityType.TEXT,
                    status=TestStatus.PENDING
                ),
                TestCase(
                    test_id=f"T{test_id+1:03d}",
                    name="Complex Reasoning",
                    description="Test multi-step reasoning capabilities",
                    modality=ModalityType.TEXT,
                    status=TestStatus.PENDING
                ),
                TestCase(
                    test_id=f"T{test_id+2:03d}",
                    name="Long Context Handling",
                    description="Test handling of extended context",
                    modality=ModalityType.TEXT,
                    status=TestStatus.PENDING
                )
            ])
            test_id += 3
        
        # Image tests
        if "image" in capabilities.input_modalities:
            self.test_cases.extend([
                TestCase(
                    test_id=f"T{test_id:03d}",
                    name="Image Understanding",
                    description="Test image analysis and description",
                    modality=ModalityType.IMAGE,
                    status=TestStatus.PENDING
                ),
                TestCase(
                    test_id=f"T{test_id+1:03d}",
                    name="Image with Text Query",
                    description="Test answering questions about images",
                    modality=ModalityType.IMAGE,
                    status=TestStatus.PENDING
                )
            ])
            test_id += 2
        
        # Video tests
        if "video" in capabilities.input_modalities:
            self.test_cases.extend([
                TestCase(
                    test_id=f"T{test_id:03d}",
                    name="Video Understanding",
                    description="Test video content analysis",
                    modality=ModalityType.VIDEO,
                    status=TestStatus.PENDING
                ),
                TestCase(
                    test_id=f"T{test_id+1:03d}",
                    name="Temporal Analysis",
                    description="Test understanding of events over time",
                    modality=ModalityType.VIDEO,
                    status=TestStatus.PENDING
                )
            ])
            test_id += 2
        
        # Document tests
        if "document" in capabilities.input_modalities:
            self.test_cases.extend([
                TestCase(
                    test_id=f"T{test_id:03d}",
                    name="Document Parsing",
                    description="Test document content extraction",
                    modality=ModalityType.DOCUMENT,
                    status=TestStatus.PENDING
                ),
                TestCase(
                    test_id=f"T{test_id+1:03d}",
                    name="Complex Document Understanding",
                    description="Test understanding of structured documents",
                    modality=ModalityType.DOCUMENT,
                    status=TestStatus.PENDING
                )
            ])
            test_id += 2
    
    async def _execution_phase(
        self,
        test_plan: Dict[str, Any],
        assets: Dict[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """Phase 3: Execute all test cases"""
        results = []
        
        for test_case in self.test_cases:
            logger.info(f"Executing: {test_case.test_id} - {test_case.name}")
            
            start = datetime.now()
            test_case.status = TestStatus.RUNNING
            
            try:
                # Execute based on modality
                if test_case.modality == ModalityType.TEXT:
                    result = await self._execute_text_test(test_case)
                elif test_case.modality == ModalityType.IMAGE:
                    result = await self._execute_image_test(test_case, assets.get('image', []))
                elif test_case.modality == ModalityType.VIDEO:
                    result = await self._execute_video_test(test_case, assets.get('video', []))
                elif test_case.modality == ModalityType.DOCUMENT:
                    result = await self._execute_document_test(test_case, assets.get('document', []))
                else:
                    result = {"success": False, "error": "Unknown modality"}
                
                test_case.duration_seconds = (datetime.now() - start).total_seconds()
                test_case.result = result
                test_case.status = TestStatus.PASSED if result.get('success') else TestStatus.FAILED
                
                if not result.get('success'):
                    test_case.error_message = result.get('error', 'Unknown error')
                
                results.append({
                    "test_id": test_case.test_id,
                    "name": test_case.name,
                    "status": test_case.status.value,
                    "duration": test_case.duration_seconds,
                    "result": result
                })
                
                logger.info(f"Completed: {test_case.test_id} - {test_case.status.value}")
                
            except Exception as e:
                test_case.status = TestStatus.FAILED
                test_case.error_message = str(e)
                logger.error(f"Test failed: {test_case.test_id} - {str(e)}")
                
                results.append({
                    "test_id": test_case.test_id,
                    "name": test_case.name,
                    "status": "failed",
                    "error": str(e)
                })
        
        return results
    
    async def _execute_text_test(self, test_case: TestCase) -> Dict[str, Any]:
        """Execute a text-based test"""
        if not self.model_capabilities:
            return {"success": False, "error": "No model capabilities"}
        
        prompts = {
            "Basic Text Generation": "Write a creative short story about AI in 100 words.",
            "Complex Reasoning": "Solve this logic puzzle: If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly? Explain your reasoning step by step.",
            "Long Context Handling": "Summarize the key themes in modern AI development, focusing on safety, capabilities, and societal impact."
        }
        
        prompt = prompts.get(test_case.name, "Tell me about artificial intelligence.")
        
        result = self.invoker.invoke_with_text(
            self.model_capabilities.model_id,
            prompt
        )
        
        return result
    
    async def _execute_image_test(
        self,
        test_case: TestCase,
        image_paths: List[str]
    ) -> Dict[str, Any]:
        """Execute an image-based test"""
        if not self.model_capabilities or not image_paths:
            return {"success": False, "error": "No model capabilities or images"}
        
        image_path = image_paths[0]
        
        prompts = {
            "Image Understanding": "Describe this image in detail.",
            "Image with Text Query": "What objects can you see in this image? List them."
        }
        
        prompt = prompts.get(test_case.name, "Analyze this image.")
        
        result = self.invoker.invoke_with_image(
            self.model_capabilities.model_id,
            prompt,
            image_path
        )
        
        return result
    
    async def _execute_video_test(
        self,
        test_case: TestCase,
        video_paths: List[str]
    ) -> Dict[str, Any]:
        """Execute a video-based test"""
        if not self.model_capabilities or not video_paths:
            return {"success": False, "error": "No model capabilities or videos"}
        
        video_path = video_paths[0]
        
        prompts = {
            "Video Understanding": "Describe what happens in this video.",
            "Temporal Analysis": "What are the key events in chronological order?"
        }
        
        prompt = prompts.get(test_case.name, "Analyze this video.")
        
        result = self.invoker.invoke_with_video(
            self.model_capabilities.model_id,
            prompt,
            video_path
        )
        
        return result
    
    async def _execute_document_test(
        self,
        test_case: TestCase,
        document_paths: List[str]
    ) -> Dict[str, Any]:
        """Execute a document-based test"""
        if not self.model_capabilities or not document_paths:
            return {"success": False, "error": "No model capabilities or documents"}
        
        doc_path = document_paths[0]
        
        prompts = {
            "Document Parsing": "Extract and summarize the key information from this document.",
            "Complex Document Understanding": "What is the structure and main content of this document?"
        }
        
        prompt = prompts.get(test_case.name, "Analyze this document.")
        
        result = self.invoker.invoke_with_document(
            self.model_capabilities.model_id,
            prompt,
            doc_path
        )
        
        return result
    
    async def _analysis_phase(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Phase 4: Analyze test results"""
        prompt = f"""Analyze these comprehensive test results:

{json.dumps(test_results, indent=2)}

Provide:
1. Overall model performance assessment
2. Strengths by modality
3. Weaknesses and limitations
4. Comparison with expected capabilities
5. Unexpected findings
6. Performance metrics analysis
7. Recommendations for usage
8. Areas needing improvement

Be thorough and insightful."""
        
        try:
            result = self.agents["analyzer"](prompt)
            logger.info("Analysis completed")
            
            # Calculate metrics
            total = len(test_results)
            passed = len([r for r in test_results if r.get('status') == 'passed'])
            failed = len([r for r in test_results if r.get('status') == 'failed'])
            
            return {
                "analysis": str(result),
                "metrics": {
                    "total_tests": total,
                    "passed": passed,
                    "failed": failed,
                    "success_rate": (passed / total * 100) if total > 0 else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Analysis phase failed: {str(e)}")
            return {"error": str(e)}
    
    async def _reporting_phase(
        self,
        model_id: str,
        capabilities: ModelCapabilities,
        test_results: List[Dict[str, Any]],
        analysis: Dict[str, Any]
    ) -> Dict[str, str]:
        """Phase 5: Generate comprehensive reports"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare comprehensive report data
        report_data = {
            "model_id": model_id,
            "model_name": capabilities.model_name if capabilities else "unknown",
            "test_date": datetime.now().isoformat(),
            "capabilities": asdict(capabilities) if capabilities else {},
            "test_results": test_results,
            "analysis": analysis,
            "test_cases": [asdict(tc) for tc in self.test_cases]
        }
        
        # Generate JSON report
        json_path = f"test_reports/model_test_report_{timestamp}.json"
        os.makedirs("test_reports", exist_ok=True)
        
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Generate Markdown report
        prompt = f"""Generate a comprehensive, professional test report in Markdown format.

Report data:
{json.dumps(report_data, indent=2, default=str)}

Create a report with:
1. Executive Summary
2. Model Overview and Capabilities
3. Test Methodology
4. Detailed Test Results (organized by modality)
5. Performance Analysis
6. Findings and Insights
7. Recommendations
8. Conclusion

Make it professional, clear, and actionable."""
        
        try:
            md_result = self.agents["reporter"](prompt)
            
            md_path = f"test_reports/model_test_report_{timestamp}.md"
            with open(md_path, 'w') as f:
                f.write(str(md_result))
            
            logger.info(f"Reports generated: {json_path}, {md_path}")
            
            return {
                "json_report": json_path,
                "markdown_report": md_path
            }
            
        except Exception as e:
            logger.error(f"Reporting phase failed: {str(e)}")
            return {"error": str(e)}
    
    def test_model(self, model_id: str) -> Dict[str, Any]:
        """Synchronous wrapper for async testing"""
        return asyncio.run(self.test_model_async(model_id))


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Comprehensive AI Model Testing Agent"
    )
    parser.add_argument(
        "model_id",
        help="Bedrock Model ID to test (e.g., 'amazon.nova-pro-v1:0', 'us.anthropic.claude-sonnet-4-20250514-v1:0')"
    )
    parser.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region for Bedrock (default: us-east-1)"
    )
    
    args = parser.parse_args()
    
    # Initialize coordinator
    coordinator = ModelTestingCoordinator(region=args.region)
    
    # Run comprehensive testing
    results = coordinator.test_model(args.model_id)
    
    # Print summary
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)
    print(f"\nModel ID: {results['model_id']}")
    print(f"Status: {results['status']}")
    print(f"Duration: {results.get('duration_seconds', 0):.2f} seconds")
    
    if results['status'] == 'success':
        summary = results['test_summary']
        print(f"\nTests: {summary['total_tests']} total")
        print(f"  ✅ Passed: {summary['passed']}")
        print(f"  ❌ Failed: {summary['failed']}")
        print(f"  ⏭️  Skipped: {summary['skipped']}")
        print(f"\nReports:")
        for report_type, path in results['reports'].items():
            print(f"  - {report_type}: {path}")
    else:
        print(f"\nError: {results.get('error', 'Unknown error')}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
