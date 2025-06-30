import os

import pytest
from litellm import (
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    RateLimitError,
    APIConnectionError,
)

from tacho.ai import llm


# Comprehensive list of models to test across providers
# Format: (model_name, [required_env_vars])
TEST_MODELS = [
    # OpenAI models
    ("gpt-4o-mini", ["OPENAI_API_KEY"]),
    ("openai/gpt-4.1-mini", ["OPENAI_API_KEY"]),
    # OpenAI reasoning models
    ("o4-mini", ["OPENAI_API_KEY"]),
    # Anthropic models
    ("claude-sonnet-4-20250514", ["ANTHROPIC_API_KEY"]),
    ("anthropic/claude-sonnet-4-20250514", ["ANTHROPIC_API_KEY"]),
    # Google models
    ("gemini/gemini-2.0-flash", ["GEMINI_API_KEY"]),
    # AWS Bedrock
    (
        "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",
        ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION_NAME"],
    ),
    # Google Vertex AI
    (
        "vertex_ai/gemini-2.0-flash",
        ["GOOGLE_APPLICATION_CREDENTIALS", "VERTEXAI_PROJECT", "VERTEXAI_LOCATION"],
    ),
    # Local models
    ("ollama_chat/deepseek-r1", []),
]


def get_env_vars(env_var_names: list[str]) -> dict[str, str | None]:
    """Get all required environment variables."""
    env_vars = {}
    for var_name in env_var_names:
        # Special handling for AWS_REGION_NAME which has a default
        if var_name == "AWS_REGION_NAME":
            env_vars[var_name] = os.getenv(var_name, "us-east-1")
        else:
            env_vars[var_name] = os.getenv(var_name)
    return env_vars


def has_required_env_vars(env_vars: dict[str, str | None]) -> bool:
    """Check if all required environment variables are set."""
    return all(value is not None for value in env_vars.values())


@pytest.mark.integration
@pytest.mark.parametrize("model_name,env_var_names", TEST_MODELS)
@pytest.mark.asyncio
async def test_provider_models(model_name, env_var_names):
    """Test various model names across different providers with minimal token usage."""
    env_vars = get_env_vars(env_var_names)

    # Skip test if required environment variables are not set
    if not has_required_env_vars(env_vars):
        missing_vars = [k for k, v in env_vars.items() if v is None]
        pytest.skip(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )

    # Try to call the model with minimal settings
    try:
        response = await llm(model_name, "Hi", tokens=100)

        # Basic assertions about the response
        assert response is not None
        assert hasattr(response, "choices")
        assert len(response.choices) > 0
        assert hasattr(response.choices[0], "message")
        assert hasattr(response.choices[0].message, "content")

        # The response content should exist (even if truncated due to max_tokens=1)
        assert response.choices[0].message.content is not None

    except AuthenticationError as e:
        # Authentication errors indicate the model name is valid but credentials are wrong
        pytest.fail(
            f"Authentication failed for {model_name}: {str(e)}. Check your API key."
        )

    except BadRequestError as e:
        # Some BadRequestErrors are expected for certain model/provider combinations
        error_msg = str(e).lower()

        # Known cases where model names might not be available or have changed
        if any(
            phrase in error_msg
            for phrase in [
                "model not found",
                "does not exist",
                "invalid model",
                "not available",
                "not supported",
                "no model",
                "unknown model",
            ]
        ):
            pytest.skip(f"Model {model_name} not available: {str(e)}")
        else:
            # Other bad request errors should fail the test
            pytest.fail(f"Bad request for {model_name}: {str(e)}")

    except NotFoundError as e:
        # Model doesn't exist - this is useful information
        pytest.skip(f"Model {model_name} not found: {str(e)}")

    except RateLimitError as e:
        # Rate limit errors mean the model is valid but we're hitting limits
        pytest.skip(f"Rate limit hit for {model_name}: {str(e)}")

    except APIConnectionError as e:
        # Connection errors might indicate service issues
        # Special handling for Ollama models
        if model_name.startswith(("ollama/", "ollama_chat/")):
            pytest.skip(
                f"Ollama server not running (expected at localhost:11434): {str(e)}"
            )
        else:
            pytest.skip(f"Connection error for {model_name}: {str(e)}")

    except Exception as e:
        # Catch any other exceptions and provide helpful information
        error_type = type(e).__name__
        pytest.fail(f"Unexpected {error_type} for {model_name}: {str(e)}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_invalid_model_handling():
    """Test that invalid model names raise appropriate exceptions."""
    # Test completely invalid model name
    with pytest.raises((BadRequestError, NotFoundError)):
        await llm("completely-invalid-model-xyz", "Hi", tokens=1)

    # Test model without provider prefix when required
    with pytest.raises((BadRequestError, NotFoundError)):
        await llm("gemini-pro", "Hi", tokens=1)  # Should be gemini/gemini-pro

    # Test invalid provider prefix
    with pytest.raises((BadRequestError, NotFoundError)):
        await llm("invalid-provider/gpt-4", "Hi", tokens=1)
