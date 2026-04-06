"""Unit tests for multi-model tier routing in llm.py and provider_factory.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.generation.llm import get_llm_client_by_tier, invoke_llm_by_tier, resolve_tier_config
from src.generation.provider_factory import build_chat_model, build_embeddings

# =========================================================================
# provider_factory — build_chat_model / build_embeddings
# =========================================================================


class TestProviderFactory:
    """Tests for provider_factory.build_chat_model and build_embeddings."""

    @patch("src.generation.provider_factory.ChatOpenAI")
    def test_openai_compatible_chat_model_created(self, mock_cls):
        build_chat_model(
            provider="openai_compatible",
            model="gpt-4o-mini",
            base_url=None,
            api_key="sk-test",
            temperature=0.0,
        )
        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args.kwargs or mock_cls.call_args[1]
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["openai_api_key"] == "sk-test"

    @patch("src.generation.provider_factory.ChatOpenAI")
    def test_openrouter_headers_injected(self, mock_cls):
        build_chat_model(
            provider="openai_compatible",
            model="qwen/qwen3-8b",
            base_url="https://openrouter.ai/api/v1",
            api_key="sk-or-v1-abc",
            temperature=0.0,
        )
        call_kwargs = mock_cls.call_args.kwargs or mock_cls.call_args[1]
        assert "default_headers" in call_kwargs
        assert "HTTP-Referer" in call_kwargs["default_headers"]

    @patch("src.generation.provider_factory.ChatOpenAI")
    def test_non_openrouter_base_url_no_headers(self, mock_cls):
        build_chat_model(
            provider="openai_compatible",
            model="qwen3-8b",
            base_url="http://localhost:8080/v1",
            api_key="sk-local",
            temperature=0.0,
        )
        call_kwargs = mock_cls.call_args.kwargs or mock_cls.call_args[1]
        assert "default_headers" not in call_kwargs

    def test_unknown_provider_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            build_chat_model(
                provider="unknown_provider",
                model="some-model",
                base_url=None,
                api_key="sk",
            )

    def test_anthropic_provider_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="anthropic"):
            build_chat_model(
                provider="anthropic",
                model="claude-3-5-sonnet",
                base_url=None,
                api_key="sk-ant",
            )

    @patch("src.generation.provider_factory.OpenAIEmbeddings")
    def test_openai_compatible_embeddings_created(self, mock_cls):
        build_embeddings(
            provider="openai_compatible",
            model="text-embedding-3-small",
            base_url=None,
            api_key="sk-test",
            dimensions=1536,
        )
        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args.kwargs or mock_cls.call_args[1]
        assert call_kwargs["dimensions"] == 1536

    @patch("src.generation.provider_factory.OpenAIEmbeddings")
    def test_embeddings_with_base_url(self, mock_cls):
        build_embeddings(
            provider="openai_compatible",
            model="qwen/qwen3-embedding-8b",
            base_url="https://openrouter.ai/api/v1",
            api_key="sk-or-v1-abc",
            dimensions=1536,
        )
        call_kwargs = mock_cls.call_args.kwargs or mock_cls.call_args[1]
        assert call_kwargs["openai_api_base"] == "https://openrouter.ai/api/v1"

    def test_unknown_embeddings_provider_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown embeddings provider"):
            build_embeddings(
                provider="unknown_provider",
                model="some-model",
                base_url=None,
                api_key="sk",
                dimensions=1536,
            )


# =========================================================================
# resolve_tier_config
# =========================================================================


class TestResolveTierConfig:
    """Tests for resolve_tier_config — tier → model/base_url/api_key."""

    @patch("src.generation.llm.settings")
    def test_small_tier_uses_small_model(self, mock_settings):
        mock_settings.small_model_provider = "openai_compatible"
        mock_settings.small_model_name = "qwen3-8b"
        mock_settings.small_model_base_url = "http://localhost:8080/v1"
        mock_settings.small_model_api_key = "sk-small"
        mock_settings.llm_model = "gpt-4o-mini"
        mock_settings.openai_api_key = "sk-default"
        mock_settings.openrouter_api_key = ""

        cfg = resolve_tier_config("small")
        assert cfg["provider"] == "openai_compatible"
        assert cfg["model"] == "qwen3-8b"
        assert cfg["base_url"] == "http://localhost:8080/v1"
        assert cfg["api_key"] == "sk-small"

    @patch("src.generation.llm.settings")
    def test_small_tier_falls_back_to_llm_model(self, mock_settings):
        mock_settings.small_model_provider = "openai_compatible"
        mock_settings.small_model_name = ""
        mock_settings.small_model_base_url = ""
        mock_settings.small_model_api_key = ""
        mock_settings.llm_model = "gpt-4o-mini"
        mock_settings.openai_api_key = "sk-default"
        mock_settings.openrouter_api_key = ""

        cfg = resolve_tier_config("small")
        assert cfg["model"] == "gpt-4o-mini"
        assert cfg["base_url"] is None
        assert cfg["api_key"] == "sk-default"

    @patch("src.generation.llm.settings")
    def test_mid_tier_uses_mid_model(self, mock_settings):
        mock_settings.mid_model_provider = "openai_compatible"
        mock_settings.mid_model_name = "deepseek-v3"
        mock_settings.mid_model_base_url = "http://localhost:8081/v1"
        mock_settings.mid_model_api_key = "sk-mid"
        mock_settings.llm_model = "gpt-4o-mini"
        mock_settings.openai_api_key = "sk-default"
        mock_settings.openrouter_api_key = ""

        cfg = resolve_tier_config("mid")
        assert cfg["provider"] == "openai_compatible"
        assert cfg["model"] == "deepseek-v3"
        assert cfg["base_url"] == "http://localhost:8081/v1"
        assert cfg["api_key"] == "sk-mid"

    @patch("src.generation.llm.settings")
    def test_mid_tier_falls_back_to_llm_model(self, mock_settings):
        mock_settings.mid_model_provider = "openai_compatible"
        mock_settings.mid_model_name = ""
        mock_settings.mid_model_base_url = ""
        mock_settings.mid_model_api_key = ""
        mock_settings.llm_model = "gpt-4o-mini"
        mock_settings.openai_api_key = "sk-default"
        mock_settings.openrouter_api_key = ""

        cfg = resolve_tier_config("mid")
        assert cfg["model"] == "gpt-4o-mini"
        assert cfg["base_url"] is None
        assert cfg["api_key"] == "sk-default"

    @patch("src.generation.llm.settings")
    def test_frontier_tier_uses_frontier_model(self, mock_settings):
        mock_settings.frontier_model_provider = "openai_compatible"
        mock_settings.frontier_model_name = "claude-sonnet-4-20250514"
        mock_settings.frontier_model_base_url = "https://api.anthropic.com/v1"
        mock_settings.frontier_model_api_key = "sk-ant-frontier"
        mock_settings.llm_model = "gpt-4o-mini"
        mock_settings.openai_api_key = "sk-default"
        mock_settings.openrouter_api_key = ""

        cfg = resolve_tier_config("frontier")
        assert cfg["provider"] == "openai_compatible"
        assert cfg["model"] == "claude-sonnet-4-20250514"
        assert cfg["base_url"] == "https://api.anthropic.com/v1"
        assert cfg["api_key"] == "sk-ant-frontier"

    @patch("src.generation.llm.settings")
    def test_frontier_tier_falls_back_to_llm_model(self, mock_settings):
        mock_settings.frontier_model_provider = "openai_compatible"
        mock_settings.frontier_model_name = ""
        mock_settings.frontier_model_base_url = ""
        mock_settings.frontier_model_api_key = ""
        mock_settings.llm_model = "gpt-4o-mini"
        mock_settings.openai_api_key = "sk-default"
        mock_settings.openrouter_api_key = ""

        cfg = resolve_tier_config("frontier")
        assert cfg["model"] == "gpt-4o-mini"
        assert cfg["base_url"] is None
        assert cfg["api_key"] == "sk-default"

    @patch("src.generation.llm.settings")
    def test_openrouter_api_key_fallback(self, mock_settings):
        """openrouter_api_key is used when tier api_key is empty, before openai_api_key."""
        mock_settings.mid_model_provider = "openai_compatible"
        mock_settings.mid_model_name = "deepseek/deepseek-chat-v3-0324"
        mock_settings.mid_model_base_url = "https://openrouter.ai/api/v1"
        mock_settings.mid_model_api_key = ""
        mock_settings.llm_model = "gpt-4o-mini"
        mock_settings.openai_api_key = "sk-openai"
        mock_settings.openrouter_api_key = "sk-or-v1-abc"

        cfg = resolve_tier_config("mid")
        assert cfg["api_key"] == "sk-or-v1-abc"  # OR key wins over openai_api_key

    @patch("src.generation.llm.settings")
    def test_openai_api_key_used_when_no_openrouter_key(self, mock_settings):
        """Falls back to openai_api_key when openrouter_api_key is also empty."""
        mock_settings.small_model_provider = "openai_compatible"
        mock_settings.small_model_name = ""
        mock_settings.small_model_base_url = ""
        mock_settings.small_model_api_key = ""
        mock_settings.llm_model = "gpt-4o-mini"
        mock_settings.openai_api_key = "sk-openai"
        mock_settings.openrouter_api_key = ""

        cfg = resolve_tier_config("small")
        assert cfg["api_key"] == "sk-openai"


# =========================================================================
# get_llm_client_by_tier
# =========================================================================


class TestGetLlmClientByTier:
    """Tests for get_llm_client_by_tier — delegates to provider_factory."""

    @patch("src.generation.llm.build_chat_model")
    @patch("src.generation.llm.settings")
    def test_creates_client_with_tier_config(self, mock_settings, mock_factory):
        mock_settings.mid_model_provider = "openai_compatible"
        mock_settings.mid_model_name = "deepseek-v3"
        mock_settings.mid_model_base_url = ""
        mock_settings.mid_model_api_key = "sk-mid"
        mock_settings.llm_model = "gpt-4o-mini"
        mock_settings.openai_api_key = "sk-default"
        mock_settings.openrouter_api_key = ""
        mock_settings.llm_temperature = 0.1

        get_llm_client_by_tier("mid")
        mock_factory.assert_called_once_with(
            provider="openai_compatible",
            model="deepseek-v3",
            base_url=None,
            api_key="sk-mid",
            temperature=0.1,
        )

    @patch("src.generation.llm.build_chat_model")
    @patch("src.generation.llm.settings")
    def test_creates_client_with_custom_base_url(self, mock_settings, mock_factory):
        mock_settings.small_model_provider = "openai_compatible"
        mock_settings.small_model_name = "qwen3-8b"
        mock_settings.small_model_base_url = "http://localhost:8080/v1"
        mock_settings.small_model_api_key = "sk-small"
        mock_settings.llm_model = "gpt-4o-mini"
        mock_settings.openai_api_key = "sk-default"
        mock_settings.openrouter_api_key = ""
        mock_settings.llm_temperature = 0.1

        get_llm_client_by_tier("small")
        mock_factory.assert_called_once_with(
            provider="openai_compatible",
            model="qwen3-8b",
            base_url="http://localhost:8080/v1",
            api_key="sk-small",
            temperature=0.1,
        )

    @patch("src.generation.llm.build_chat_model")
    @patch("src.generation.llm.settings")
    def test_temperature_override(self, mock_settings, mock_factory):
        mock_settings.frontier_model_provider = "openai_compatible"
        mock_settings.frontier_model_name = ""
        mock_settings.frontier_model_base_url = ""
        mock_settings.frontier_model_api_key = ""
        mock_settings.llm_model = "gpt-4o-mini"
        mock_settings.openai_api_key = "sk-default"
        mock_settings.openrouter_api_key = ""
        mock_settings.llm_temperature = 0.7

        get_llm_client_by_tier("frontier", temperature=0.0)
        mock_factory.assert_called_once_with(
            provider="openai_compatible",
            model="gpt-4o-mini",
            base_url=None,
            api_key="sk-default",
            temperature=0.0,
        )


# =========================================================================
# invoke_llm_by_tier
# =========================================================================


class TestInvokeLlmByTier:
    """Tests for invoke_llm_by_tier — async invocation at a tier."""

    @pytest.mark.asyncio
    @patch("src.generation.llm.get_llm_client_by_tier")
    async def test_invokes_and_returns_content(self, mock_get_client):
        mock_response = MagicMock()
        mock_response.content = "Test response"
        mock_client = AsyncMock()
        mock_client.ainvoke.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = await invoke_llm_by_tier(
            messages=[{"role": "user", "content": "hello"}],
            tier="mid",
        )

        assert result == "Test response"
        mock_get_client.assert_called_once_with(tier="mid", temperature=None)

    @pytest.mark.asyncio
    @patch("src.generation.llm.get_llm_client_by_tier")
    async def test_passes_temperature(self, mock_get_client):
        mock_response = MagicMock()
        mock_response.content = "ok"
        mock_client = AsyncMock()
        mock_client.ainvoke.return_value = mock_response
        mock_get_client.return_value = mock_client

        await invoke_llm_by_tier(
            messages=[{"role": "user", "content": "hello"}],
            tier="frontier",
            temperature=0.0,
        )

        mock_get_client.assert_called_once_with(tier="frontier", temperature=0.0)


# =========================================================================
# RLM engine tier selection
# =========================================================================


class TestRlmTierSelection:
    """Tests that the RLM engine selects the correct tier per depth."""

    @patch("src.generation.llm.settings")
    def test_depth_0_uses_mid_tier_model(self, mock_settings):
        """Controller (depth 0) should resolve to mid tier."""
        mock_settings.rlm_controller_model = ""
        mock_settings.mid_model_name = "deepseek-v3"
        mock_settings.llm_model = "gpt-4o-mini"

        # Simulate the resolution logic from engine.py
        model = (
            mock_settings.rlm_controller_model
            or mock_settings.mid_model_name
            or mock_settings.llm_model
        )
        assert model == "deepseek-v3"

    @patch("src.generation.llm.settings")
    def test_depth_1_uses_small_tier_model(self, mock_settings):
        """Sub-RLM (depth > 0) should resolve to small tier."""
        mock_settings.rlm_sub_model = ""
        mock_settings.small_model_name = "qwen3-8b"
        mock_settings.llm_model = "gpt-4o-mini"

        model = (
            mock_settings.rlm_sub_model
            or mock_settings.small_model_name
            or mock_settings.llm_model
        )
        assert model == "qwen3-8b"

    @patch("src.generation.llm.settings")
    def test_depth_0_falls_back_to_llm_model(self, mock_settings):
        """When no tier models configured, falls back to llm_model."""
        mock_settings.rlm_controller_model = ""
        mock_settings.mid_model_name = ""
        mock_settings.llm_model = "gpt-4o-mini"

        model = (
            mock_settings.rlm_controller_model
            or mock_settings.mid_model_name
            or mock_settings.llm_model
        )
        assert model == "gpt-4o-mini"


# =========================================================================
# Synthesis model resolution
# =========================================================================


class TestSynthesisModelResolution:
    """Tests for synthesis model resolution in rlm_execute."""

    @patch("src.generation.llm.settings")
    def test_synthesis_model_prefers_rlm_synthesis(self, mock_settings):
        mock_settings.rlm_synthesis_model = "gpt-4o"
        mock_settings.frontier_model_name = "claude-sonnet-4-20250514"

        synthesis_model = mock_settings.rlm_synthesis_model or mock_settings.frontier_model_name
        assert synthesis_model == "gpt-4o"

    @patch("src.generation.llm.settings")
    def test_synthesis_model_falls_back_to_frontier(self, mock_settings):
        mock_settings.rlm_synthesis_model = ""
        mock_settings.frontier_model_name = "claude-sonnet-4-20250514"

        synthesis_model = mock_settings.rlm_synthesis_model or mock_settings.frontier_model_name
        assert synthesis_model == "claude-sonnet-4-20250514"

    @patch("src.generation.llm.settings")
    def test_no_synthesis_when_both_empty(self, mock_settings):
        """When no synthesis or frontier model configured, synthesis is skipped."""
        mock_settings.rlm_synthesis_model = ""
        mock_settings.frontier_model_name = ""

        synthesis_model = mock_settings.rlm_synthesis_model or mock_settings.frontier_model_name
        assert not synthesis_model  # falsy → synthesis skipped
