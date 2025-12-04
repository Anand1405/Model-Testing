"""
Quick Start Script for Model Testing Agent
Demonstrates basic and advanced usage patterns
"""

import os
import sys
import json
from datetime import datetime
from enhanced_orchestrator import ModelTestingCoordinator


def print_banner(text):
    """Print a formatted banner"""
    width = 80
    print("\n" + "="*width)
    print(text.center(width))
    print("="*width + "\n")


def example_1_basic_usage():
    """Example 1: Basic model testing"""
    print_banner("Example 1: Basic Model Testing")
    
    print("This example demonstrates basic usage of the Model Testing Agent.\n")
    
    # Initialize coordinator
    print("📦 Initializing Model Testing Coordinator...")
    coordinator = ModelTestingCoordinator(region="us-west-2")
    print("✅ Coordinator initialized\n")
    
    # Test a model
    model_name = "Amazon Nova 2 Omni"
    print(f"🧪 Starting comprehensive test for: {model_name}")
    print("This will take several minutes...\n")
    
    results = coordinator.test_model(model_name)
    
    # Display results
    print_banner("Test Results")
    print(f"Model: {results['model_name']}")
    print(f"Status: {results['status']}")
    print(f"Duration: {results.get('duration_seconds', 0):.2f} seconds\n")
    
    if results['status'] == 'success':
        summary = results['test_summary']
        print("Test Summary:")
        print(f"  Total Tests: {summary['total_tests']}")
        print(f"  ✅ Passed: {summary['passed']}")
        print(f"  ❌ Failed: {summary['failed']}")
        print(f"  ⏭️  Skipped: {summary['skipped']}\n")
        
        print("Generated Reports:")
        for report_type, path in results['reports'].items():
            print(f"  📄 {report_type}: {path}")
    else:
        print(f"❌ Error: {results.get('error')}")


def example_2_async_usage():
    """Example 2: Async testing for better performance"""
    print_banner("Example 2: Asynchronous Testing")
    
    import asyncio
    
    async def test_multiple_models():
        """Test multiple models asynchronously"""
        coordinator = ModelTestingCoordinator()
        
        models = [
            "Amazon Nova 2 Omni",
            # Add more models as needed
        ]
        
        print(f"Testing {len(models)} model(s) asynchronously...\n")
        
        # In production, you could run these in parallel
        results = []
        for model in models:
            print(f"Testing {model}...")
            result = await coordinator.test_model_async(model)
            results.append(result)
            print(f"✅ {model} completed\n")
        
        return results
    
    # Run async tests
    results = asyncio.run(test_multiple_models())
    
    print_banner("Async Test Results")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['model_name']}")
        print(f"   Status: {result['status']}")
        if result['status'] == 'success':
            summary = result['test_summary']
            print(f"   Success Rate: {summary['passed']}/{summary['total_tests']}")


def example_3_custom_configuration():
    """Example 3: Custom configuration and selective testing"""
    print_banner("Example 3: Custom Configuration")
    
    print("This example shows how to customize the testing process.\n")
    
    # Initialize with custom settings
    coordinator = ModelTestingCoordinator(region="us-west-2")
    
    # Modify test cases to focus on specific modalities
    print("📝 Customizing test cases...")
    
    # Filter to only text and image tests
    original_count = len(coordinator.test_cases)
    coordinator.test_cases = [
        tc for tc in coordinator.test_cases 
        if tc.modality.value in ['text', 'image']
    ]
    filtered_count = len(coordinator.test_cases)
    
    print(f"   Original test cases: {original_count}")
    print(f"   Filtered test cases: {filtered_count}")
    print(f"   Focus: Text and Image modalities only\n")
    
    # Run tests
    model_name = "Amazon Nova 2 Omni"
    print(f"🧪 Testing {model_name} with custom configuration...\n")
    
    results = coordinator.test_model(model_name)
    
    print_banner("Custom Test Results")
    if results['status'] == 'success':
        summary = results['test_summary']
        print(f"Completed {summary['total_tests']} targeted tests")
        print(f"Success Rate: {summary['passed']}/{summary['total_tests']}")


def example_4_programmatic_access():
    """Example 4: Programmatic access to detailed results"""
    print_banner("Example 4: Programmatic Result Analysis")
    
    coordinator = ModelTestingCoordinator()
    
    # Run tests
    model_name = "Amazon Nova 2 Omni"
    print(f"Testing {model_name}...\n")
    results = coordinator.test_model(model_name)
    
    if results['status'] == 'success':
        # Access detailed test case results
        print("Detailed Test Analysis:\n")
        
        passed_tests = [tc for tc in coordinator.test_cases if tc.status.value == 'passed']
        failed_tests = [tc for tc in coordinator.test_cases if tc.status.value == 'failed']
        
        print(f"✅ Passed Tests ({len(passed_tests)}):")
        for tc in passed_tests[:5]:  # Show first 5
            print(f"   • {tc.name} ({tc.duration_seconds:.2f}s)")
        
        if len(passed_tests) > 5:
            print(f"   ... and {len(passed_tests) - 5} more\n")
        else:
            print()
        
        if failed_tests:
            print(f"❌ Failed Tests ({len(failed_tests)}):")
            for tc in failed_tests:
                print(f"   • {tc.name}")
                print(f"     Error: {tc.error_message}\n")
        
        # Access model capabilities
        if coordinator.model_capabilities:
            print("Model Capabilities:")
            caps = coordinator.model_capabilities
            print(f"   Model ID: {caps.model_id}")
            print(f"   Input Modalities: {', '.join(caps.input_modalities)}")
            print(f"   Output Modalities: {', '.join(caps.output_modalities)}")
            print(f"   Max Tokens: {caps.max_tokens}")
            print(f"   Streaming: {caps.supports_streaming}")
            print(f"   Tool Use: {caps.supports_tool_use}")
            print(f"   Extended Thinking: {caps.supports_extended_thinking}")


def example_5_integration():
    """Example 5: Integration with existing workflows"""
    print_banner("Example 5: Workflow Integration")
    
    print("This example shows how to integrate with existing workflows.\n")
    
    def test_and_notify(model_name: str, notification_callback):
        """Test a model and notify on completion"""
        coordinator = ModelTestingCoordinator()
        
        print(f"📊 Testing {model_name}...")
        results = coordinator.test_model(model_name)
        
        # Call notification callback
        notification_callback(results)
        
        return results
    
    def send_notification(results):
        """Example notification function"""
        status = results['status']
        model = results['model_name']
        
        if status == 'success':
            summary = results['test_summary']
            message = f"""
Model Testing Complete! ✅

Model: {model}
Status: {status}
Success Rate: {summary['passed']}/{summary['total_tests']}
Duration: {results.get('duration_seconds', 0):.2f}s

Reports available at:
{results.get('reports', {})}
"""
        else:
            message = f"""
Model Testing Failed! ❌

Model: {model}
Error: {results.get('error', 'Unknown error')}
"""
        
        print("\n📧 Notification would be sent:")
        print(message)
        
        # In production, you would send email, Slack message, etc.
        # send_email(message)
        # send_slack(message)
        # post_to_webhook(message)
    
    # Run test with notification
    test_and_notify("Amazon Nova 2 Omni", send_notification)


def example_6_error_handling():
    """Example 6: Robust error handling"""
    print_banner("Example 6: Error Handling")
    
    print("This example demonstrates proper error handling.\n")
    
    def safe_test_model(model_name: str, max_retries: int = 3):
        """Test with retry logic"""
        coordinator = ModelTestingCoordinator()
        
        for attempt in range(max_retries):
            try:
                print(f"Attempt {attempt + 1}/{max_retries}...")
                results = coordinator.test_model(model_name)
                
                if results['status'] == 'success':
                    print("✅ Test completed successfully!")
                    return results
                else:
                    print(f"⚠️ Test failed: {results.get('error')}")
                    if attempt < max_retries - 1:
                        print("   Retrying...\n")
                    
            except Exception as e:
                print(f"❌ Exception occurred: {str(e)}")
                if attempt < max_retries - 1:
                    print("   Retrying...\n")
                else:
                    print("   Max retries reached.")
                    return {
                        'model_name': model_name,
                        'status': 'failed',
                        'error': str(e)
                    }
        
        return {
            'model_name': model_name,
            'status': 'failed',
            'error': 'Max retries exceeded'
        }
    
    # Test with error handling
    results = safe_test_model("Amazon Nova 2 Omni")
    
    print("\nFinal Result:")
    print(f"Status: {results['status']}")


def example_7_batch_testing():
    """Example 7: Batch testing multiple models"""
    print_banner("Example 7: Batch Testing")
    
    print("Testing multiple models in batch mode.\n")
    
    # Define models to test
    models_to_test = [
        "Amazon Nova 2 Omni",
        # Add more models as they become available
    ]
    
    # Results storage
    all_results = []
    
    # Test each model
    coordinator = ModelTestingCoordinator()
    
    for i, model_name in enumerate(models_to_test, 1):
        print(f"\n[{i}/{len(models_to_test)}] Testing {model_name}")
        print("-" * 60)
        
        results = coordinator.test_model(model_name)
        all_results.append(results)
        
        # Print summary
        if results['status'] == 'success':
            summary = results['test_summary']
            print(f"✅ Passed: {summary['passed']}/{summary['total_tests']}")
        else:
            print(f"❌ Failed: {results.get('error')}")
    
    # Generate comparison report
    print_banner("Batch Testing Summary")
    
    print("Model Comparison:\n")
    for result in all_results:
        model = result['model_name']
        status = result['status']
        
        if status == 'success':
            summary = result['test_summary']
            success_rate = (summary['passed'] / summary['total_tests'] * 100)
            print(f"  {model}:")
            print(f"    Status: ✅ {status}")
            print(f"    Success Rate: {success_rate:.1f}%")
            print(f"    Duration: {result.get('duration_seconds', 0):.2f}s\n")
        else:
            print(f"  {model}:")
            print(f"    Status: ❌ {status}")
            print(f"    Error: {result.get('error', 'Unknown')}\n")


def main():
    """Main function to run examples"""
    examples = {
        "1": ("Basic Usage", example_1_basic_usage),
        "2": ("Async Usage", example_2_async_usage),
        "3": ("Custom Configuration", example_3_custom_configuration),
        "4": ("Programmatic Access", example_4_programmatic_access),
        "5": ("Workflow Integration", example_5_integration),
        "6": ("Error Handling", example_6_error_handling),
        "7": ("Batch Testing", example_7_batch_testing),
    }
    
    print_banner("Model Testing Agent - Quick Start Examples")
    
    print("Available Examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    print("  0. Run All Examples")
    print("  q. Quit\n")
    
    choice = input("Select an example (1-7, 0, or q): ").strip()
    
    if choice == 'q':
        print("\nGoodbye!")
        return
    
    if choice == '0':
        print("\n🚀 Running all examples...\n")
        for name, func in examples.values():
            try:
                func()
                input("\nPress Enter to continue to next example...")
            except KeyboardInterrupt:
                print("\n\nExamples interrupted by user.")
                return
            except Exception as e:
                print(f"\n❌ Error in example: {str(e)}")
                input("\nPress Enter to continue...")
    elif choice in examples:
        name, func = examples[choice]
        try:
            func()
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
    else:
        print(f"\n❌ Invalid choice: {choice}")
    
    print("\n" + "="*80)
    print("Examples completed! Check the generated reports for detailed results.")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()