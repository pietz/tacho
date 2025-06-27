import os
import pytest
from typer.testing import CliRunner
from dotenv import load_dotenv

from tacho.cli import app

load_dotenv()

runner = CliRunner()


class TestCLIIntegration:
    def test_help_command(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "CLI tool for measuring LLM inference speeds" in result.stdout
        
    def test_invalid_model_error(self):
        result = runner.invoke(app, ["invalid-model-xyz"])
        assert result.exit_code == 1
        assert "✗ invalid-model-xyz" in result.stdout
        
    def test_benchmark_single_model(self):
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
            
        result = runner.invoke(app, ["gpt-4.1-mini", "--runs=1", "--lim=50"])
        assert result.exit_code == 0
        assert "tok/s" in result.stdout
        assert "gpt-4.1-mini" in result.stdout
        
    def test_benchmark_multiple_models(self):
        if not os.getenv("OPENAI_API_KEY") or not os.getenv("GEMINI_API_KEY"):
            pytest.skip("API keys not set")
            
        result = runner.invoke(app, [
            "gpt-4.1-mini", 
            "gemini/gemini-2.5-flash",
            "--runs=1", 
            "--lim=50"
        ])
        assert result.exit_code == 0
        assert "gpt-4.1-mini" in result.stdout
        assert "gemini-2.5-flash" in result.stdout
        assert result.stdout.count("tok/s") >= 2
        
    def test_benchmark_alias_command(self):
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
            
        result = runner.invoke(app, ["benchmark", "gpt-4.1-mini", "--runs=1", "--lim=50"])
        assert result.exit_code == 0
        assert "tok/s" in result.stdout
        
    def test_error_handling_no_api_key(self):
        original_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            result = runner.invoke(app, ["gpt-4.1-mini", "--runs=1", "--lim=50"])
            assert result.exit_code == 1
            assert "✗ gpt-4.1-mini" in result.stdout
        finally:
            if original_key:
                os.environ["OPENAI_API_KEY"] = original_key
                
    def test_ping_single_model(self):
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
            
        result = runner.invoke(app, ["ping", "gpt-4.1-mini"])
        assert result.exit_code == 0
        assert "✓ gpt-4.1-mini" in result.stdout
        assert "models are accessible" in result.stdout
        
    def test_ping_multiple_models(self):
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
            
        result = runner.invoke(app, ["ping", "gpt-4.1-mini", "invalid-model-xyz"])
        assert result.exit_code == 0
        assert "✓ gpt-4.1-mini" in result.stdout
        assert "✗ invalid-model-xyz" in result.stdout
        assert "1/2 models are accessible" in result.stdout
        
    def test_ping_all_invalid_models(self):
        result = runner.invoke(app, ["ping", "invalid-model-1", "invalid-model-2"])
        assert result.exit_code == 1
        assert "✗ invalid-model-1" in result.stdout
        assert "✗ invalid-model-2" in result.stdout
        assert "No models are accessible" in result.stdout
                
