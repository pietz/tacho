import pytest
import os
from unittest.mock import patch, AsyncMock
from tacho.core import llm


# List of providers to test with their corresponding models
PROVIDER_MODELS = {
    "openai": {
        "model": "gpt-4o-mini",
        "env_var": "OPENAI_API_KEY",
        "test_model": "gpt-3.5-turbo"  # Cheaper alternative for testing
    },
    "anthropic": {
        "model": "claude-sonnet-4-20250514",
        "env_var": "ANTHROPIC_API_KEY",
        "test_model": "claude-sonnet-4-20250514"
    },
    "gemini": {
        "model": "gemini/gemini-2.5-flash",
        "env_var": "GEMINI_API_KEY", 
        "test_model": "gemini/gemini-2.5-flash"
    },
    "bedrock": {
        "model": "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
        "env_var": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
        "test_model": "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
        "region": "AWS_REGION_NAME"
    },
    "vertex": {
        "model": "vertex_ai/gemini-1.5-flash",
        "env_var": "GOOGLE_APPLICATION_CREDENTIALS",  # Path to service account JSON
        "test_model": "vertex_ai/gemini-1.5-flash",
        "project": "VERTEXAI_PROJECT"
    }
}


class TestLLMProviders:
    """Unit tests for LLM function with mocked responses"""
    
    @pytest.mark.parametrize("provider,config", PROVIDER_MODELS.items())
    @pytest.mark.asyncio
    async def test_llm_provider_call_structure(self, provider, config):
        """Test that llm function calls litellm.acompletion with correct parameters"""
        with patch("litellm.acompletion") as mock_completion:
            mock_response = AsyncMock()
            mock_completion.return_value = mock_response
            
            model = config["model"]
            prompt = "Test prompt"
            max_tokens = 10
            
            result = await llm(model, prompt, max_tokens)
            
            # Verify the call was made correctly
            mock_completion.assert_called_once_with(
                model,
                [{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            assert result == mock_response

    @pytest.mark.asyncio
    async def test_llm_without_max_tokens(self):
        """Test that llm function works without max_tokens parameter"""
        with patch("litellm.acompletion") as mock_completion:
            mock_response = AsyncMock()
            mock_completion.return_value = mock_response
            
            result = await llm("gpt-4o-mini", "Test prompt")
            
            mock_completion.assert_called_once_with(
                "gpt-4o-mini",
                [{"role": "user", "content": "Test prompt"}],
                max_tokens=None
            )


@pytest.mark.integration
class TestLLMProvidersIntegration:
    """Integration tests that require actual API keys"""
    
    def _has_required_env_vars(self, config):
        """Check if required environment variables are set for a provider"""
        env_vars = config.get("env_var")
        if env_vars is None:
            return True  # No env vars required (e.g., Ollama)
        
        if isinstance(env_vars, list):
            return all(os.getenv(var) for var in env_vars)
        else:
            return bool(os.getenv(env_vars))
    
    def _get_skip_reason(self, provider, config):
        """Get skip reason for a provider"""
        env_vars = config.get("env_var")
        if env_vars is None:
            return None
        
        if isinstance(env_vars, list):
            missing = [var for var in env_vars if not os.getenv(var)]
            return f"Missing environment variables: {', '.join(missing)}"
        else:
            return f"Missing {env_vars} environment variable"
    
    @pytest.mark.parametrize("provider,config", PROVIDER_MODELS.items())
    @pytest.mark.asyncio
    async def test_provider_basic_completion(self, provider, config):
        """Test basic completion with each provider"""
        if not self._has_required_env_vars(config):
            pytest.skip(self._get_skip_reason(provider, config))
        
        # Use test model (usually cheaper/smaller)
        model = config["test_model"]
        prompt = "Reply with just 'yes' if you can read this message."
        
        try:
            response = await llm(model, prompt, tokens=5)
            
            # Basic assertions
            assert response is not None
            assert hasattr(response, 'choices') or hasattr(response, 'usage')
            
            # If response has choices, check content
            if hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content
                assert content is not None
                assert len(content) > 0
                
        except Exception as e:
            pytest.fail(f"Provider {provider} failed with error: {str(e)}")
    
    @pytest.mark.parametrize("provider,config", PROVIDER_MODELS.items())
    @pytest.mark.asyncio
    async def test_provider_with_longer_response(self, provider, config):
        """Test providers with longer responses to verify token counting"""
        if not self._has_required_env_vars(config):
            pytest.skip(self._get_skip_reason(provider, config))
        
        model = config["test_model"]
        prompt = "Count from 1 to 20 with spaces between numbers."
        
        try:
            response = await llm(model, prompt, tokens=50)
            
            # Check usage information
            if hasattr(response, 'usage'):
                assert response.usage is not None
                if hasattr(response.usage, 'completion_tokens'):
                    assert response.usage.completion_tokens > 0
                    assert response.usage.completion_tokens <= 50
            
            # Check response content
            if hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content
                assert len(content) > 10  # Should have some substantial content
                
        except Exception as e:
            pytest.fail(f"Provider {provider} failed with longer response: {str(e)}")
    
    @pytest.mark.parametrize("provider,config", [
        (p, c) for p, c in PROVIDER_MODELS.items() 
        if p in ["openai", "anthropic", "gemini", "groq"]  # Fast providers only
    ])
    @pytest.mark.asyncio 
    async def test_provider_error_handling(self, provider, config):
        """Test error handling for invalid models"""
        if not self._has_required_env_vars(config):
            pytest.skip(self._get_skip_reason(provider, config))
        
        # Use an invalid model name
        invalid_model = f"{provider}/invalid-model-name-12345"
        prompt = "Test"
        
        with pytest.raises(Exception):
            await llm(invalid_model, prompt, tokens=5)


@pytest.mark.integration
class TestProviderSpecificFeatures:
    """Test provider-specific features and configurations"""
    
    @pytest.mark.asyncio
    async def test_ollama_local_connection(self):
        """Test Ollama with custom base URL if configured"""
        if not os.getenv("OLLAMA_API_BASE"):
            pytest.skip("OLLAMA_API_BASE not configured")
        
        # Ollama should work without API key
        response = await llm("ollama/llama3.2", "Reply with 'ok'", tokens=5)
        assert response is not None
    
    @pytest.mark.asyncio
    async def test_bedrock_with_region(self):
        """Test AWS Bedrock with region configuration"""
        required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION_NAME"]
        if not all(os.getenv(var) for var in required_vars):
            pytest.skip(f"Missing AWS credentials: {required_vars}")
        
        response = await llm(
            "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
            "Reply with 'ok'",
            tokens=5
        )
        assert response is not None
    
    @pytest.mark.asyncio
    async def test_vertex_with_project(self):
        """Test Google Vertex AI with project configuration"""
        if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or not os.getenv("VERTEXAI_PROJECT"):
            pytest.skip("Missing Google Cloud credentials or project")
        
        response = await llm(
            "vertex_ai/gemini-1.5-flash",
            "Reply with 'ok'",
            tokens=5
        )
        assert response is not None


def pytest_collection_modifyitems(config, items):
    """Add markers to tests based on their requirements"""
    for item in items:
        # Mark all tests in TestLLMProvidersIntegration as integration tests
        if "Integration" in item.cls.__name__:
            item.add_marker(pytest.mark.integration)