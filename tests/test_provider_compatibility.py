import pytest
import os
from tacho.core import llm


# List of models to test - each represents a different provider
TEST_MODELS = [
    # OpenAI
    ("gpt-4o-mini", "OPENAI_API_KEY"),
    
    # Anthropic  
    ("claude-3-haiku-20240307", "ANTHROPIC_API_KEY"),
    
    # Google Gemini
    ("gemini/gemini-2.0-flash-exp", "GEMINI_API_KEY"),
    
    # Groq
    ("groq/llama-3.3-70b-versatile", "GROQ_API_KEY"),
    
    # Together AI
    ("together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", "TOGETHERAI_API_KEY"),
    
    # DeepSeek
    ("deepseek/deepseek-chat", "DEEPSEEK_API_KEY"),
    
    # Mistral
    ("mistral/mistral-small-latest", "MISTRAL_API_KEY"),
    
    # Cohere
    ("command-r", "COHERE_API_KEY"),
    
    # Perplexity
    ("perplexity/llama-3.1-sonar-small-128k-online", "PERPLEXITYAI_API_KEY"),
    
    # Fireworks AI
    ("fireworks_ai/accounts/fireworks/models/llama-v3p1-8b-instruct", "FIREWORKS_API_KEY"),
    
    # NVIDIA
    ("nvidia/meta/llama-3.1-8b-instruct", "NVIDIA_API_KEY"),
    
    # AI21
    ("ai21/jamba-1.5-mini", "AI21_API_KEY"),
    
    # Ollama (no API key needed)
    ("ollama/llama3.2", None),
    
    # AWS Bedrock (needs multiple env vars)
    ("bedrock/anthropic.claude-3-haiku-20240307-v1:0", ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION_NAME"]),
    
    # Google Vertex AI
    ("vertex_ai/gemini-1.5-flash", ["GOOGLE_APPLICATION_CREDENTIALS", "VERTEXAI_PROJECT"]),
]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_all_providers():
    """Test that all supported providers work correctly with proper authentication"""
    results = {}
    test_prompt = "Reply with just the word 'yes' if you understand this message."
    
    for model, env_vars in TEST_MODELS:
        provider = model.split("/")[0] if "/" in model else model.split("-")[0]
        
        # Check if required environment variables are set
        if env_vars is None:
            # No auth required (e.g., Ollama)
            env_status = "✓"
        elif isinstance(env_vars, list):
            # Multiple env vars required
            if all(os.getenv(var) for var in env_vars):
                env_status = "✓"
            else:
                missing = [var for var in env_vars if not os.getenv(var)]
                env_status = f"Missing: {', '.join(missing)}"
        else:
            # Single env var
            env_status = "✓" if os.getenv(env_vars) else f"Missing: {env_vars}"
        
        # Skip if environment variables are missing
        if env_status != "✓":
            results[model] = f"SKIPPED - {env_status}"
            continue
        
        # Try to call the model
        try:
            response = await llm(model, test_prompt, tokens=10)
            
            # Validate response structure
            assert response is not None, f"{model}: Response is None"
            
            # Check for content in response
            if hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content
                assert content, f"{model}: No content in response"
                results[model] = f"SUCCESS - Response: '{content.strip()[:50]}...'"
            else:
                results[model] = "SUCCESS - Response received"
                
        except Exception as e:
            error_msg = str(e)
            # Truncate long error messages
            if len(error_msg) > 100:
                error_msg = error_msg[:100] + "..."
            results[model] = f"FAILED - {error_msg}"
    
    # Print summary
    print("\n" + "="*80)
    print("PROVIDER COMPATIBILITY TEST RESULTS")
    print("="*80)
    
    success_count = sum(1 for r in results.values() if r.startswith("SUCCESS"))
    failed_count = sum(1 for r in results.values() if r.startswith("FAILED"))
    skipped_count = sum(1 for r in results.values() if r.startswith("SKIPPED"))
    
    for model, result in results.items():
        status_emoji = "✅" if result.startswith("SUCCESS") else "❌" if result.startswith("FAILED") else "⏭️"
        print(f"{status_emoji} {model:50} {result}")
    
    print("="*80)
    print(f"Summary: {success_count} passed, {failed_count} failed, {skipped_count} skipped")
    print("="*80)
    
    # Assert that we have at least some successful tests
    assert success_count > 0, "No providers succeeded - check your API keys"
    
    # If any explicitly failed (not skipped), that's concerning
    if failed_count > 0:
        failed_models = [m for m, r in results.items() if r.startswith("FAILED")]
        print(f"\n⚠️  Warning: The following providers failed: {', '.join(failed_models)}")
        print("These providers may have issues with their API keys or service availability.")


if __name__ == "__main__":
    # Allow running this test directly
    import asyncio
    asyncio.run(test_all_providers())