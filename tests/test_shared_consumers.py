"""Contract pieces other workspace projects build on when they use this AIService.

Consumers (bandido, impostor, book-watch, lamplight, calibre summarizer, story atlas,
mathforge, music writer) pick a provider by name, may bring their own key and
endpoint, and must never be handed another provider's key.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_suite import providers as cli
from ai_suite import service as api


def write_config(folder, **values):
    path = Path(folder) / "config.json"
    path.write_text(json.dumps({"provider": "hyper", "writing_model": "m", "api_key_env": "HYPER_API_KEY",
                                "groq_rate_state_path": str(Path(folder) / "groq.json"), **values}))
    return str(path)


def build(folder, **kwargs):
    return api.AIService(write_config(folder), str(Path(folder) / "usage.json"),
                         allow_auth_prompt=False, client_max_retries=0, **kwargs)


class ConfigOverrideTests(unittest.TestCase):
    def test_overrides_merge_over_the_file(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"HYPER_API_KEY": "env-key"}):
            service = build(folder, config_overrides={"base_url": "https://gateway.example/v1", "timeout": 30})
        self.assertEqual((service.base_url, service.timeout), ("https://gateway.example/v1", 30))
        # Was "env-key": a consumer's own endpoint must not receive the provider's key
        # unless it asks for it (critic finding H1) -- by api_key or api_key_env override.
        self.assertEqual(service.api_key, "")
        self.assertEqual(service.writing_model, "m")
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"HYPER_API_KEY": "env-key"}):
            service = build(folder, config_overrides={"base_url": "https://gateway.example/v1",
                                                      "api_key_env": "HYPER_API_KEY"})
        self.assertEqual(service.api_key, "env-key")

    def test_explicit_key_beats_every_environment_fallback(self):
        env = {"HYPER_API_KEY": "", "AI_API_KEY": "other", "OPENAI_API_KEY": "openai-key"}
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, env):
            service = build(folder, config_overrides={"api_key": "consumer-key"})
        self.assertEqual(service.api_key, "consumer-key")

    def test_no_overrides_keeps_existing_behaviour(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"HYPER_API_KEY": "env-key"}):
            service = build(folder)
        self.assertEqual((service.api_key, service.base_url), ("env-key", "https://hyper.charm.land/v1"))


class FailFastTests(unittest.TestCase):
    """Interactive consumers (a served report, a game turn) cannot wait hours for a reset."""

    def client(self, reply):
        client = object.__new__(api.AIService)
        client.provider_label, client.writing_model, client.review_model = "hyper", "m", "m"
        calls = []

        def once(*args):
            # Bounded: a regression that waits again must fail here, not hang the suite.
            calls.append(args)
            if len(calls) > 3:
                raise AssertionError("generate_content kept waiting instead of raising")
            return reply() if callable(reply) else reply
        client._generate_content_once = once
        return client

    def test_limit_notice_raises_instead_of_waiting(self):
        client = self.client("You've hit your usage limit. Try again at 5pm.")
        with patch.object(api.time, "sleep") as sleep:
            with self.assertRaises(api.ProviderLimitReached) as caught:
                client.generate_content("p", wait_for_limits=False)
        sleep.assert_not_called()
        self.assertIn("usage limit", str(caught.exception))

    def test_limit_error_raises_instead_of_waiting(self):
        def boom():
            raise RuntimeError("429 rate limit exceeded")
        with patch.object(api.time, "sleep") as sleep:
            with self.assertRaises(api.ProviderLimitReached):
                self.client(boom).generate_content("p", wait_for_limits=False)
        sleep.assert_not_called()

    def test_claude_cli_bad_model_notice_is_an_error_not_text(self):
        with self.assertRaisesRegex(RuntimeError, "selected model"):
            self.client("There's an issue with the selected model (x). It may not exist.").generate_content("p")

    def test_pollinations_out_of_credit_reply_is_an_error_not_text(self):
        with self.assertRaisesRegex(RuntimeError, "enough credits"):
            self.client("The account behind this API key doesn't have enough credits. This model needs "
                        "paid Pollen.").generate_content("p")

    def test_normal_text_is_returned(self):
        self.assertEqual(self.client("fine").generate_content("p", wait_for_limits=False), "fine")


class Chunk:
    def __init__(self, text="", finish=None, reasoning=None):
        delta = type("D", (), {"content": text, "reasoning_content": reasoning})()
        self.choices = [type("C", (), {"delta": delta, "finish_reason": finish})()]


class ChatOptionTests(unittest.TestCase):
    """lamplight's router needs system prompts, temperature and streamed replies."""

    def setUp(self):
        sleep = patch.object(api.time, "sleep")  # the service's retry delays must not run for real
        sleep.start()
        self.addCleanup(sleep.stop)

    def service(self, folder, replies, **config):
        service = build(folder, config_overrides=config)
        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            return replies.pop(0)
        service.client = type("Client", (), {})()
        service.client.chat = type("Chat", (), {})()
        service.client.chat.completions = type("Completions", (), {"create": staticmethod(create)})()
        return service, calls

    def test_dead_model_or_empty_account_fails_without_retrying(self):
        missing = RuntimeError("Error code: 404 - model not found: x")
        missing.status_code = 404
        for error in (missing, RuntimeError("Upstream request failed: Insufficient account funds")):
            with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"HYPER_API_KEY": "k"}):
                service, calls = self.service(folder, [])
                def create(**kwargs):
                    calls.append(kwargs)
                    raise error
                service.client.chat.completions.create = create
                with self.assertRaises(RuntimeError):
                    service.generate_content("hi", max_retries=4)
            self.assertEqual(len(calls), 1, error)

    def test_system_and_temperature_reach_chat_endpoints(self):
        reply = type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": "ok"})(),
                                                           "finish_reason": "stop"})()]})()
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"HYPER_API_KEY": "k"}):
            service, calls = self.service(folder, [reply])
            self.assertEqual(service.generate_content("hi", system="be brief", temperature=0.0), "ok")
        self.assertEqual(calls[0]["messages"], [{"role": "system", "content": "be brief"},
                                                {"role": "user", "content": "hi"}])
        self.assertEqual(calls[0]["temperature"], 0.0)

    def test_oauth_never_sends_temperature(self):
        # ChatGPT's backend rejects it: the proxy drops the stream mid-body.
        reply = type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": "ok"})(),
                                                           "finish_reason": "stop"})()]})()
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"HYPER_API_KEY": "k"}):
            service, calls = self.service(folder, [reply])
            service.provider = "openai-oauth"
            self.assertEqual(service.generate_content("hi", temperature=0.8), "ok")
        self.assertNotIn("temperature", calls[0])

    def test_last_usage_is_what_the_provider_reported(self):
        usage = type("U", (), {"prompt_tokens": 12, "completion_tokens": 5})()
        reply = type("R", (), {"usage": usage, "choices": [type("C", (), {
            "message": type("M", (), {"content": "ok"})(), "finish_reason": "stop"})()]})()
        bare = type("R", (), {"choices": reply.choices})()
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"HYPER_API_KEY": "k"}):
            service, _calls = self.service(folder, [reply, bare])
            service.generate_content("hi")
            self.assertEqual(service.last_usage, {"prompt_tokens": 12, "completion_tokens": 5})
            service.generate_content("hi")
            self.assertIsNone(service.last_usage)  # never an estimate dressed up as billing

    def test_no_delay_after_the_last_attempt(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"HYPER_API_KEY": "k"}):
            service, _calls = self.service(folder, [])  # empty: every call raises IndexError
            with self.assertRaises(IndexError):
                service.generate_content("hi", max_retries=1)
        api.time.sleep.assert_not_called()

    def test_defaults_send_neither(self):
        reply = type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": "ok"})(),
                                                           "finish_reason": "stop"})()]})()
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"HYPER_API_KEY": "k"}):
            service, calls = self.service(folder, [reply])
            service.generate_content("hi")
        self.assertEqual(calls[0]["messages"], [{"role": "user", "content": "hi"}])
        self.assertNotIn("temperature", calls[0])
        self.assertNotIn("stream", calls[0])

    def test_stream_config_streams_and_joins_text_only(self):
        stream = [Chunk("Hel", reasoning="thinking"), Chunk("lo"), Chunk("", finish="stop")]
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"HYPER_API_KEY": "k"}):
            service, calls = self.service(folder, [stream], stream=True)
            self.assertEqual(service.generate_content("hi"), "Hello")
        self.assertTrue(calls[0]["stream"])

    def test_truncated_stream_retries_with_a_larger_cap(self):
        cut = [Chunk("half", finish="length")]
        whole = [Chunk("whole", finish="stop")]
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"HYPER_API_KEY": "k"}), \
                patch.object(api.time, "sleep"):
            service, calls = self.service(folder, [cut, whole], stream=True)
            self.assertEqual(service.generate_content("hi", max_retries=2), "whole")
        self.assertGreater(calls[1]["max_tokens"], calls[0]["max_tokens"])

    def test_cli_providers_get_the_system_prompt_natively(self):
        # Was "prepended"; the CLI transports now take it as a real system prompt
        # (claude's --append-system-prompt), see test_shared_portable.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text(json.dumps({"provider": "claude", "writing_model": "sonnet",
                                        "groq_rate_state_path": str(Path(folder) / "groq.json")}))
            with patch.object(api, "claude_executable", return_value="claude"):
                service = api.AIService(str(path), str(Path(folder) / "usage.json"), allow_auth_prompt=False)
            with patch.object(api, "claude_chat", return_value="done") as chat:
                service.generate_content("hi", system="rules")
        self.assertEqual((chat.call_args.args[1], chat.call_args.kwargs["system"]), ("hi", "rules"))

    def test_a_streamed_timeout_is_per_chunk_so_cli_calls_get_a_whole_reply_budget(self):
        # article-writer sends timeout 300 with stream on; the OpenCode CLI prints the
        # text only once complete, so 300s was the whole call and killed long prompts.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text(json.dumps({"provider": "opencode", "writing_model": "m",
                                        "groq_rate_state_path": str(Path(folder) / "groq.json")}))
            with patch.object(api, "opencode_executable", return_value="opencode"):
                streamed = api.AIService(str(path), str(Path(folder) / "usage.json"), allow_auth_prompt=False,
                                         config_overrides={"timeout": 300, "stream": True})
                plain = api.AIService(str(path), str(Path(folder) / "usage.json"), allow_auth_prompt=False,
                                      config_overrides={"timeout": 300})
            with patch.object(api, "opencode_chat", return_value="done") as chat:
                streamed.generate_content("hi")
                plain.generate_content("hi")
        self.assertEqual([c.kwargs["timeout"] for c in chat.call_args_list], [api.CLI_TIMEOUT, 300])


class NvidiaProviderTests(unittest.TestCase):
    def test_nvidia_nim_is_a_catalogue_provider(self):
        """Bandido's NVIDIA NIM players moved here; the OpenAI-compatible branch serves it."""
        self.assertIn("nvidia", cli.CATALOGUE_PROVIDERS)
        data = json.loads(Path(cli.PROVIDER_CONFIG_MAP["nvidia"]).read_text(encoding="utf-8"))
        self.assertEqual(data["base_url"], "https://integrate.api.nvidia.com/v1")
        self.assertEqual(data["api_key_env"], "NVIDIA_API_KEY")
        self.assertIn(data["writing_model"], data["models"])
        with patch.object(cli, "_live_model_ids", return_value=None):
            self.assertIn("moonshotai/kimi-k3", cli._provider_models("nvidia"))
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "nv-key"}):
            service = api.AIService(cli.provider_config_path("nvidia"), allow_auth_prompt=False)
        self.assertEqual((service.api_key, service.base_url, service.provider_label),
                         ("nv-key", "https://integrate.api.nvidia.com/v1", "integrate.api.nvidia.com"))



class FreeTierProviderTests(unittest.TestCase):
    def test_free_tier_providers_use_only_their_own_key(self):
        for provider in ("cerebras", "mistral", "cloudflare", "sambanova", "chutes", "pollinations",
                         "ollama", "lmstudio"):
            data = json.loads(Path(cli.provider_config_path(provider)).read_text(encoding="utf-8"))
            if provider in cli.CATALOGUE_PROVIDERS:
                self.assertIn(data["writing_model"], data["models"], provider)
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "other", data["api_key_env"]: "own"}):
                service = api.AIService(cli.provider_config_path(provider), allow_auth_prompt=False)
            self.assertEqual((service.api_key, service.base_url), ("own", data["base_url"]), provider)

    def test_lowercased_menu_picks_go_out_in_the_configs_spelling(self):
        with patch.dict(os.environ, {"AI_WRITING_MODEL": "minimax-m3", "AI_REVIEW_MODEL": "deepseek-v3.2"}):
            service = api.AIService(cli.provider_config_path("sambanova"), allow_auth_prompt=False)
        self.assertEqual((service.writing_model, service.review_model), ("MiniMax-M3", "DeepSeek-V3.2"))
        self.assertEqual(service._max_output("DeepSeek-V3.2"), 7168)

    def test_local_servers_offer_every_loaded_model(self):
        with patch.object(cli, "_live_model_ids", return_value={"qwen3:32b"}),                 patch.object(cli, "_model_facts", return_value={}):
            self.assertEqual(cli._load_catalogue("ollama"), {"qwen3:32b": ["qwen3:32b", cli.LOCAL_MAX_OUTPUT]})
            # Hosted providers still ignore a listing that shares nothing with the curated list.
            self.assertIn("gpt-oss-120b", cli._load_catalogue("cerebras"))


class ProviderConfigPathTests(unittest.TestCase):
    def test_prefers_the_local_copy(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / "ai_config_x.json"
            base.write_text("{}")
            with patch.dict(cli.PROVIDER_CONFIG_MAP, {"x": str(base)}):
                self.assertEqual(cli.provider_config_path("x"), str(base))
                local = base.with_name("ai_config_x.local.json")
                local.write_text("{}")
                self.assertEqual(cli.provider_config_path("x"), str(local))

    def test_codex_alias_and_unknown_provider(self):
        self.assertEqual(cli.provider_config_path("codex"), cli.provider_config_path("openai-oauth"))
        with self.assertRaises(KeyError):
            cli.provider_config_path("nope")




class ConsoleRoutingTests(unittest.TestCase):
    def test_consumer_routes_startup_and_retry_output_without_redirecting_stdout(self):
        import contextlib
        import io

        messages, output = [], io.StringIO()
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(api, "claude_executable", return_value="claude"), \
             patch.object(api, "claude_chat", side_effect=[RuntimeError("temporary failure"), "answer"]), \
             patch.object(api.time, "sleep"), contextlib.redirect_stdout(output):
            service = api.AIService(
                usage_state_path=str(Path(folder) / "usage.json"),
                config_overrides={"provider": "claude"}, log=messages.append)
            self.assertEqual(service.generate_content("p", max_retries=2), "answer")
            print("foreground remains visible")
        self.assertEqual(output.getvalue(), "foreground remains visible\n")
        self.assertTrue(any("initialized" in m for m in messages))
        self.assertTrue(any("temporary failure" in m for m in messages))

    def test_oauth_cold_start_routes_child_output_and_keeps_failure_detail(self):
        import contextlib
        import io

        messages, output = [], io.StringIO()
        def launch(*args, **kwargs):
            self.assertTrue(kwargs.get("capture_output"), "proxy subprocess leaks to terminal")
            self.assertTrue(kwargs.get("text"))
            return api.subprocess.CompletedProcess(args[0], 0, "proxy ready\n", "startup notice\n")

        with patch.object(api, "_openai_oauth_proxy_running", side_effect=[False, True]), \
             patch.object(api.shutil, "which", return_value="npx"), \
             patch.object(api.subprocess, "run", side_effect=launch), contextlib.redirect_stdout(output):
            api.ensure_openai_oauth_proxy(log=messages.append)
        self.assertEqual(output.getvalue(), "")
        self.assertIn("proxy ready", messages)
        self.assertIn("startup notice", messages)
        error = api.subprocess.CalledProcessError(1, "npx", output="proxy failed", stderr="port busy")
        with patch.object(api, "_openai_oauth_proxy_running", return_value=False), \
             patch.object(api.shutil, "which", return_value="npx"), \
             patch.object(api.subprocess, "run", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "port busy"):
                api.ensure_openai_oauth_proxy(log=messages.append)

    def test_oauth_service_forwards_its_diagnostic_callback(self):
        messages = []
        with tempfile.TemporaryDirectory() as folder, patch.object(api, "ensure_openai_oauth_proxy") as ensure:
            api.AIService(usage_state_path=str(Path(folder) / "usage.json"),
                          config_overrides={"provider": "openai-oauth", "use_openai_client": False},
                          log=messages.append)
        ensure.assert_called_once_with(log=messages.append)


if __name__ == "__main__":
    unittest.main()
