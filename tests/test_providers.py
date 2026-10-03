"""Provider wiring for the Claude Code CLI and hyper.charm.land.

What breaks silently here: sending one provider's model name to another (a 404
that looks like an outage), and shelling out to `claude` with the prompt in
argv, which dies at Windows' 32k command-line cap only once chapters get long.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from ai_suite import providers as cli
from ai_suite import service as ai_service


def _service(config: dict) -> ai_service.AIService:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AI_BASE_URL", None)
            os.environ.pop("AI_WRITING_MODEL", None)
            os.environ.pop("AI_REVIEW_MODEL", None)
            return ai_service.AIService(config_path=str(path))


class ClaudeProviderTests(unittest.TestCase):
    def test_prompt_goes_through_stdin_not_argv(self):
        """A chapter-sized prompt in argv is 'The command line is too long'."""
        completed = subprocess.CompletedProcess([], 0, stdout="chapter text", stderr="")
        with patch.object(ai_service, "claude_executable", return_value="claude"), \
                patch.object(ai_service, "_run_cli", return_value=completed) as run:
            out = ai_service.claude_chat("opus", "x" * 40000)
        self.assertEqual(out, "chapter text")
        args, kwargs = run.call_args
        self.assertNotIn("x" * 40000, args[0])
        self.assertEqual(args[1], "x" * 40000)
        self.assertIn("--model", args[0])
        self.assertIn("opus", args[0])
        self.assertIn("-p", args[0])

    def test_role_aliases_map_to_cli_names(self):
        completed = subprocess.CompletedProcess([], 0, stdout="ok", stderr="")
        with patch.object(ai_service, "claude_executable", return_value="claude"), \
                patch.object(ai_service, "_run_cli", return_value=completed) as run:
            ai_service.claude_chat("pro", "hello")
            self.assertIn("opus", run.call_args[0][0])
            ai_service.claude_chat("claude-opus-5", "hello")
            self.assertIn("claude-opus-5", run.call_args[0][0])

    def test_empty_output_raises_with_the_stderr_reason(self):
        completed = subprocess.CompletedProcess([], 0, stdout="  \n", stderr="not logged in")
        with patch.object(ai_service, "claude_executable", return_value="claude"), \
                patch.object(ai_service, "_run_cli", return_value=completed):
            with self.assertRaises(RuntimeError) as ctx:
                ai_service.claude_chat("opus", "hello")
        self.assertIn("not logged in", str(ctx.exception))

    def test_service_needs_no_key_and_no_endpoint(self):
        service = _service({"provider": "claude", "writing_model": "opus",
                            "review_model": "sonnet"})
        self.assertEqual(service.base_url, "")
        self.assertIsNone(service.client)
        self.assertIsNone(service.session)
        self.assertFalse(service.has_budget_pause())

    def test_generate_content_routes_to_the_cli(self):
        service = _service({"provider": "claude", "writing_model": "opus",
                            "review_model": "sonnet"})
        with patch.object(ai_service, "claude_chat", return_value="prose") as chat:
            self.assertEqual(service.generate_content("write", model_type="writing"), "prose")
        self.assertEqual(chat.call_args[0][0], "opus")


class HyperProviderTests(unittest.TestCase):
    def test_base_url_and_key_env_default_to_hyper(self):
        with patch.dict(os.environ, {"HYPER_API_KEY": "sk-test"}, clear=False):
            service = _service({"provider": "hyper", "writing_model": "qwen3.8-max",
                                "review_model": "qwen3.8-max"})
        self.assertEqual(service.base_url, "https://hyper.charm.land/v1")
        self.assertEqual(service.api_key, "sk-test")
        self.assertEqual(service._api_key_env_name(), "HYPER_API_KEY")

    def test_completion_cap_uses_max_tokens(self):
        """hyper rejects the newer `max_completion_tokens` spelling with a 400."""
        with patch.dict(os.environ, {"HYPER_API_KEY": "sk-test"}, clear=False):
            service = _service({"provider": "hyper", "writing_model": "qwen3.8-max",
                                "review_model": "qwen3.8-max",
                                "models": {"qwen3.8-max": {"max_output": 65536}}})
        service.client = Mock()
        service.client.chat.completions.create.return_value = {
            "choices": [{"message": {"content": "text"}}]}
        service.generate_content("hi", model_type="writing", max_completion_tokens=999)
        kwargs = service.client.chat.completions.create.call_args.kwargs
        # the model's whole allowance, not the caller's smaller cap
        self.assertEqual(kwargs["max_tokens"], 65536)
        self.assertNotIn("max_completion_tokens", kwargs)

    def test_judge_model_overrides_the_role_default(self):
        with patch.dict(os.environ, {"HYPER_API_KEY": "sk-test"}, clear=False):
            service = _service({"provider": "hyper", "writing_model": "qwen3.8-max",
                                "review_model": "qwen3.8-max"})
        service.client = Mock()
        service.client.chat.completions.create.return_value = {
            "choices": [{"message": {"content": "{}"}}]}
        service.generate_content("hi", model_type="review", model="glm-5.2")
        self.assertEqual(
            service.client.chat.completions.create.call_args.kwargs["model"], "glm-5.2")


class ProviderLabelTests(unittest.TestCase):
    def test_label_names_the_real_endpoint_not_the_client_branch(self):
        """opencode-go rides the 'openrouter' branch; logs must say where it goes."""
        with patch.dict(os.environ, {"OPENCODE_GO_API_KEY": "sk-test"}, clear=False):
            service = _service({"provider": "openrouter", "use_openai_client": True,
                                "writing_model": "glm-5.2", "review_model": "glm-5.2",
                                "base_url": "https://opencode.ai/zen/go/v1",
                                "api_key_env": "OPENCODE_GO_API_KEY"})
        self.assertEqual(service.base_url, "https://opencode.ai/zen/go/v1")
        self.assertEqual(service.provider_label, "opencode.ai")

    def test_label_falls_back_to_provider_without_a_base_url(self):
        service = _service({"provider": "claude", "writing_model": "opus",
                            "review_model": "sonnet"})
        self.assertEqual(service.provider_label, "claude")

    def test_real_openrouter_keeps_its_name(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test"}, clear=False):
            service = _service({"provider": "openrouter", "use_openai_client": True,
                                "writing_model": "x", "review_model": "x"})
        self.assertEqual(service.base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(service.provider_label, "openrouter")


FREE_TIER_PROVIDERS = ("cerebras", "mistral", "cloudflare", "sambanova", "chutes", "pollinations",
                       "ollama", "lmstudio")


class CatalogueTests(unittest.TestCase):
    def test_every_provider_has_a_config_that_names_the_same_provider(self):
        for provider, path in cli.PROVIDER_CONFIG_MAP.items():
            base = Path(path)
            local = base.with_name(base.stem + ".local.json")
            if not base.exists() and not local.exists():
                continue          # *.local.json files are gitignored user copies
            data = json.loads((base if base.exists() else local).read_text(encoding="utf-8"))
            expected = {"opencode-go": "openrouter", "opencode-zen": "opencode",
                        "nvidia": "openrouter", "gpt4free": "openrouter"}.get(
                provider, "openrouter" if provider in FREE_TIER_PROVIDERS else provider)
            self.assertEqual(data["provider"], expected, provider)

    def test_catalogue_models_are_listed_for_claude_and_hyper(self):
        for provider in ("claude", "hyper"):
            models = cli._provider_models(provider)
            self.assertTrue(models, provider)
            for mid, (name, max_out) in models.items():
                self.assertEqual(mid, mid.lower())
                self.assertTrue(name)
                self.assertGreaterEqual(max_out, 0)
        # Claude Code takes no completion cap, so its entries carry 0 and the
        # CLI must not export an AI_*_COMPLETION_TOKENS value from them.
        self.assertTrue(all(m[1] == 0 for m in cli._provider_models("claude").values()))

    def test_retired_models_are_dropped_only_when_the_provider_answers(self):
        cli._CATALOGUE_CACHE.clear()
        with patch.object(cli, "_live_model_ids", return_value={"glm-5.3", "kimi-k3"}):
            self.assertEqual(set(cli._load_catalogue("hyper")), {"glm-5.3", "kimi-k3"})
        with patch.object(cli, "_live_model_ids", return_value=None):
            self.assertIn("qwen3.8-max", cli._load_catalogue("hyper"))

    def test_menu_facts_come_from_models_dev_not_the_zero_cli_cap(self):
        """A CLI provider's 0 means 'send no cap', not 'can write 0 tokens'."""
        catalogue = {
            "anthropic": {"models": {
                "claude-opus-4-5": {"family": "claude-opus", "release_date": "2025-11-24"},
                "claude-opus-5-5": {"family": "claude-opus", "release_date": "2026-09-22",
                                    "limit": {"context": 1000000, "output": 128000},
                                    "cost": {"input": 4, "output": 20}},
            }},
            "openrouter": {"models": {"z-ai/glm-5.3": {"cost": {"input": 0, "output": 0}}}},
        }
        with patch.object(cli, "_models_dev", return_value=catalogue), \
                patch.object(cli, "_artificial_analysis", return_value=[]), \
                patch.dict(os.environ, {"NO_COLOR": "1"}):
            opus = cli._model_facts("claude", "opus")
            self.assertEqual(cli._facts_label(opus), "ctx 1M | $4/$20")
            # Command Code prefixes ids with a different org than OpenRouter does.
            glm = cli._model_facts("commandcode", "zai-org/glm-5.3")
            self.assertEqual(cli._facts_label(glm), "ctx ? | free")
            self.assertEqual(cli._facts_label(cli._model_facts("grok", "grok-9")),
                             "ctx ? | $?")
            # Across a menu each metric runs worst 0 to best 1; the aggregate is
            # their mean, and a metric a model lacks counts as 0.
            scales = cli._scales([opus, glm])
            self.assertEqual((scales[0]["ctx"], scales[0]["price"]), (1.0, 0.0))
            self.assertEqual(scales[1]["price"], 1.0)
            self.assertEqual([s["score"] for s in scales], [0.5, 0.5])
            self.assertEqual(cli._facts_label(opus, scales[0]), "ctx 1M | $4/$20 -> 50")

    def test_state_key_is_stable_for_the_existing_opencode_entries(self):
        self.assertEqual(cli._model_state_key("opencode-go"), "opencode_go_model")
        self.assertEqual(cli._model_state_key("opencode-zen"), "opencode_zen_model")
        self.assertEqual(cli._model_state_key("claude"), "claude_model")

    def test_hyper_judges_never_share_a_family_with_the_writers(self):
        data = json.loads(Path(cli.PROVIDER_CONFIG_MAP["hyper"]).read_text(encoding="utf-8"))
        family = lambda m: m.split("-")[0].split(".")[0].lower()
        busy = {family(data["writing_model"]), family(data["review_model"])}
        for judge in data["judge_models"]:
            self.assertNotIn(family(judge), busy,
                             "a judge from the writer's own family measures nothing")

    def test_choose_ai_remembers_one_model_per_role_in_the_callers_state(self):
        """The menu other scripts share: auto mode reuses picks without asking."""
        cli._CATALOGUE_CACHE.clear()
        env = {k: v for k, v in os.environ.items() if not k.startswith("AI_")}
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(cli, "_live_model_ids", return_value=None), \
                patch.dict(os.environ, env, clear=True), \
                patch("builtins.input", side_effect=AssertionError("auto mode asked")):
            state = Path(tmp) / "state.json"
            state.write_text(json.dumps({"provider": "hyper", "hyper_model_review": "kimi-k3"}))
            provider, config, models = cli.choose_ai(
                None, "auto", state, ("work", "review"), ("glm-5.3", "not-served"))
            self.assertEqual((provider, models), ("hyper", ["glm-5.3", "kimi-k3"]))
            self.assertTrue(config.endswith(".json"))
            self.assertEqual(os.environ["AI_WRITING_MODEL"], "glm-5.3")
            self.assertEqual(os.environ["AI_REVIEW_MODEL"], "kimi-k3")
            saved = json.loads(state.read_text())
            self.assertEqual((saved["hyper_model"], saved["hyper_model_review"]), ("glm-5.3", "kimi-k3"))

    def test_choose_ai_offers_only_the_efforts_the_model_lists(self):
        """Effort levels are per model (models.dev); a remembered one the model lacks is dropped."""
        cli._CATALOGUE_CACHE.clear()
        catalogue = {"hyper": {"models": {
            "glm-5.3": {"reasoning_options": [{"type": "toggle"}, {"type": "effort", "values": ["low", "high"]}]},
            "kimi-k3": {"reasoning_options": [{"type": "budget_tokens", "min": 1, "max": 9}]}}},
            "minimax": {"models": {"glm-5.3": {"reasoning_options": [{"type": "effort", "values": ["low"]}]}}}}
        env = {k: v for k, v in os.environ.items() if not k.startswith("AI_")}
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(cli, "_live_model_ids", return_value=None), \
                patch.object(cli, "_models_dev", return_value=catalogue), \
                patch.object(cli, "_artificial_analysis", return_value=[]), \
                patch.object(cli, "_refresh_missing_intelligence"), \
                patch.dict(os.environ, env, clear=True):
            self.assertEqual(cli._effort_levels("hyper", "glm-5.3"), ["low", "high"])
            self.assertEqual(cli._effort_levels("hyper", "kimi-k3"), [])     # thinks, but takes no level
            self.assertEqual(cli._effort_levels("minimax", "glm-5.3"), [])   # its transport sends none
            state = Path(tmp) / "state.json"
            state.write_text(json.dumps({"provider": "hyper", "hyper_model": "glm-5.3",
                                         "hyper_model_review": "kimi-k3",
                                         "hyper_effort": "high", "hyper_effort_review": "high"}))
            with patch("builtins.input", side_effect=AssertionError("auto mode asked")):
                cli.choose_ai(None, "auto", state, ("work", "review"))
            self.assertEqual(os.environ["AI_WRITING_EFFORT"], "high")
            self.assertNotIn("AI_REVIEW_EFFORT", os.environ)
            # Asked once, for the one model with levels; the caller that owns effort is never asked.
            # Each effort menu follows its own model's menu, before the next role's model.
            with patch.object(cli, "_arrow_menu", side_effect=["glm-5.3", "low", "kimi-k3"]) as menu:
                cli.choose_ai("hyper", "review", state, ("work", "review"))
            self.assertEqual([rid for rid, _ in menu.call_args_list[1].args[1]], ["default", "low", "high"])
            self.assertEqual(os.environ["AI_WRITING_EFFORT"], "low")
            self.assertEqual(json.loads(state.read_text())["hyper_effort"], "low")
            with patch.object(cli, "_arrow_menu", side_effect=["glm-5.3", "kimi-k3"]):
                cli.choose_ai("hyper", "review", state, ("work", "review"), effort=False)
            self.assertNotIn("AI_WRITING_EFFORT", os.environ)
            # A role the state has never seen starts on the first role's picks, so a
            # consumer that adds a review role keeps running what it ran before.
            state.write_text(json.dumps({"provider": "hyper", "hyper_model": "glm-5.3", "hyper_effort": "high"}))
            cli.choose_ai(None, "auto", state, ("writing", "review"))
            self.assertEqual((os.environ["AI_REVIEW_MODEL"], os.environ["AI_REVIEW_EFFORT"]), ("glm-5.3", "high"))
            # Only an answered menu is remembered: the review role still inherits next time.
            self.assertNotIn("hyper_effort_review", json.loads(state.read_text()))
            self.assertEqual(cli._effort_levels("gpt4free", "glm-5.3"), [])  # fuzzy name match, unverified server
            # models.dev unreachable: the remembered effort stands, in the run and on disk.
            with patch.object(cli, "_models_dev", return_value={}):
                cli.choose_ai(None, "auto", state, ("writing", "review"))
            self.assertEqual(os.environ["AI_WRITING_EFFORT"], "high")
            self.assertEqual(json.loads(state.read_text())["hyper_effort"], "high")

    def test_new_live_models_join_the_menu_when_models_dev_knows_their_output(self):
        cli._CATALOGUE_CACHE.clear()
        facts = {"brand-new": {"limit": {"output": 65536}}}
        with patch.object(cli, "_live_model_ids", return_value={"glm-5.3", "brand-new", "an-image-model"}), \
                patch.object(cli, "_model_facts", side_effect=lambda p, m: facts.get(m, {})):
            models = cli._load_catalogue("hyper")
        self.assertEqual(models["brand-new"], ["brand-new", 65536])
        self.assertNotIn("an-image-model", models)

    def test_claude_menu_adds_each_familys_newest_model_from_models_dev(self):
        """A Claude release reaches the menu without a config edit; aliases stay."""
        cli._CATALOGUE_CACHE.clear()
        catalogue = {"anthropic": {"models": {
            "claude-sonnet-5": {"family": "claude-sonnet", "release_date": "2026-06-29"},
            "claude-sonnet-6": {"family": "claude-sonnet", "release_date": "2027-02-01"},
            "claude-haiku-4-5": {"family": "claude-haiku", "release_date": "2025-10-15"},
            "claude-haiku-4-5-20251001": {"family": "claude-haiku", "release_date": "2025-10-15"},
            "claude-opus-5-5": {"family": "claude-opus", "release_date": "2026-09-22"},
            "claude-fable-5": {"family": "claude-fable", "release_date": "2027-03-01"},
        }}}
        with patch.object(cli, "_models_dev", return_value=catalogue):
            self.assertEqual(cli._live_model_ids("claude"),
                             {"claude-sonnet-6", "claude-haiku-4-5", "claude-opus-5-5", "claude-fable-5"})
            models = cli._load_catalogue("claude")
        self.assertEqual(models["claude-sonnet-6"], ["claude-sonnet-6", 0])
        self.assertIn("claude-sonnet-5", models)  # curated ids are never dropped
        self.assertIn("opus", models)
        # The config lists haiku only as its dated snapshot: the bare id is no new model.
        self.assertNotIn("claude-haiku-4-5", models)
        # A longer curated id (claude-fable-5-1) is a different model, not a snapshot.
        self.assertIn("claude-fable-5", models)

    def test_opencode_calls_carry_a_session_id(self):
        """Without x-opencode-session opencode.ai answers 400 MissingSessionID."""
        service = _service({"provider": "openrouter", "base_url": "https://opencode.ai/zen/go/v1",
                            "api_key": "k", "writing_model": "m", "review_model": "m"})
        headers = service._extra_headers()
        self.assertEqual(len(headers["x-opencode-session"]), 32)
        self.assertEqual(service._extra_headers(), headers)  # one session per service
        other = _service({"provider": "openrouter", "base_url": "https://openrouter.ai/api/v1",
                          "api_key": "k", "writing_model": "m", "review_model": "m"})
        self.assertNotIn("x-opencode-session", other._extra_headers())


class PublicProviderOptionsTests(unittest.TestCase):
    def test_options_expose_the_complete_ordered_catalogue(self):
        options = cli.provider_options()
        self.assertEqual(list(options), list(cli.PROVIDER_CONFIG_MAP))
        self.assertIn("commandcode", options)
        self.assertEqual(options["commandcode"]["provider"], "commandcode")
        self.assertIn("deepseek/deepseek-v4-pro", options["commandcode"]["models"])
        self.assertEqual(options["commandcode"]["writing_model"], "deepseek/deepseek-v4-pro")
        self.assertFalse(options["commandcode"]["needs_api_key"])

    def test_options_read_configs_only_and_never_ask_providers(self):
        # Consumers call this at import time; a live listing there costs seconds per provider.
        with patch.object(cli, "_live_model_ids", side_effect=AssertionError("live call")), \
             patch.object(cli, "_CATALOGUE_CACHE", {}):
            options = cli.provider_options()
        self.assertIn("claude-sonnet-5", options["commandcode"]["models"])


class Gpt4freeTests(unittest.TestCase):
    """gpt4free's local `g4f api` server: OpenAI-compatible, its own model names."""

    def test_config_targets_the_local_server_without_a_real_key(self):
        self.assertIn("gpt4free", cli.CATALOGUE_PROVIDERS)
        data = json.loads(Path(cli.PROVIDER_CONFIG_MAP["gpt4free"]).read_text(encoding="utf-8"))
        self.assertEqual(data["token_param"], "max_tokens")  # g4f's request model has no other
        self.assertIn(data["writing_model"], data["models"])
        env = {k: v for k, v in os.environ.items() if k != "G4F_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            service = _service(data)
        self.assertEqual(service.base_url, "http://127.0.0.1:1337/v1")
        # g4f hands any other Bearer token to its backends as their key, so never
        # send another provider's: G4F_API_KEY (the server's own) or nothing.
        self.assertEqual(service.api_key, "")

    def test_live_listing_skips_provider_and_image_entries(self):
        """/v1/models lists every g4f backend as a "model" (provider: true) too."""
        body = json.dumps({"data": [
            {"id": "deepseek-v4-pro", "image": False, "provider": False},
            {"id": "flux", "image": True, "provider": False},
            {"id": "PollinationsAI", "image": True, "provider": True},
        ]}).encode()
        resp = Mock(read=Mock(return_value=body))
        resp.__enter__ = Mock(return_value=resp)
        resp.__exit__ = Mock(return_value=False)
        with patch.object(cli, "urlopen", return_value=resp):
            self.assertEqual(cli._live_model_ids("gpt4free"), {"deepseek-v4-pro"})

    def test_g4f_names_find_their_models_dev_entry_and_show_free(self):
        catalogue = {
            "openrouter": {"models": {
                "openai/gpt-4o": {"limit": {"context": 1, "output": 1}},
                "meta-llama/llama-3.3-70b-instruct": {"limit": {"context": 131072, "output": 16384}},
            }},
            "openai": {"models": {"gpt-4o": {"limit": {"context": 128000, "output": 16384},
                                             "cost": {"input": 2.5, "output": 10}}}},
            "groq": {"models": {"llama-3.3-70b-versatile": {"limit": {"context": 9, "output": 9}}}},
            "alibaba": {"models": {"qwen3-235b-a22b": {"limit": {"context": 131072, "output": 32768}}}},
        }
        cli._models_dev_index.cache_clear()
        try:
            with patch.object(cli, "_models_dev", return_value=catalogue),                     patch.object(cli, "_artificial_analysis", return_value=[]),                     patch.dict(os.environ, {"NO_COLOR": "1"}):
                # The maker's own listing beats a reseller's copy of the same id.
                self.assertEqual(cli._facts_label(cli._model_facts("gpt4free", "gpt-4o")),
                                 "ctx 128K | free")
                self.assertEqual(cli._model_facts("gpt4free", "qwen-3-235b")["limit"]["output"], 32768)
                self.assertEqual(cli._model_facts("gpt4free", "llama-3.3-70b")["limit"]["context"], 131072)
                self.assertEqual(cli._model_facts("gpt4free", "flux"), {})
        finally:
            cli._models_dev_index.cache_clear()


class ArrowMenuTests(unittest.TestCase):
    ROWS = [(f"m{i:02d}", f"model {i}") for i in range(30)]

    def _run(self, keys: str, default: str = "m03", **kw) -> str:
        msvcrt = Mock(getwch=Mock(side_effect=list(keys)))
        tty = Mock(isatty=Mock(return_value=True))
        with patch.object(cli.os, "name", "nt"), patch.object(cli.os, "system"), \
                patch.dict("sys.modules", {"msvcrt": msvcrt}), \
                patch.object(cli.sys, "stdin", tty), patch.object(cli.sys, "stdout", tty), \
                patch("builtins.print") as out:
            picked = cli._arrow_menu("Models:", self.ROWS, default, **kw)
        frames = [c.args[0] for c in out.call_args_list if "Models:" in c.args[0]]
        self.assertTrue(frames)
        self.assertTrue(all(f.count("] model") <= cli.MENU_ROWS for f in frames))
        return picked

    def test_enter_takes_the_preselected_default(self):
        self.assertEqual(self._run("\r"), "m03")

    def test_scrolls_past_the_window_and_space_moves_the_selection(self):
        self.assertEqual(self._run("\xe0P" * 22 + " \r"), "m25")

    def test_typing_filters_rows_and_highlights_the_match(self):
        self.assertEqual(self._run("m2\xe0P\r"), "m21")       # m20..m29 left; Down, Enter
        self.assertEqual(self._run("M25\r"), "m25")           # case-insensitive, cursor row wins
        self.assertEqual(self._run("m29\x08\x0805\r"), "m05")  # Backspace widens again
        self.assertEqual(self._run("zz\r\x1b\r"), "m03")      # no match: Enter waits; Esc clears
        self.assertEqual(self._run("m1\x1b\x1b"), "m03")      # second Esc keeps the default
        with patch.dict(os.environ, {"NO_COLOR": ""}):
            msvcrt = Mock(getwch=Mock(side_effect=list("07\r")))
            tty = Mock(isatty=Mock(return_value=True))
            with patch.object(cli.os, "name", "nt"), patch.object(cli.os, "system"), \
                    patch.dict("sys.modules", {"msvcrt": msvcrt}), \
                    patch.object(cli.sys, "stdin", tty), patch.object(cli.sys, "stdout", tty), \
                    patch("builtins.print") as out:
                cli._arrow_menu("Models:", [(m, f"{m}  {label}") for m, label in self.ROWS], "m03")
        frame = out.call_args_list[-2].args[0]
        self.assertIn("m\033[7m07\033[27m", frame)
        self.assertNotIn("m03", frame)

    def test_deselecting_falls_back_to_the_cursor_row(self):
        self.assertEqual(self._run(" \xe0P\r"), "m04")        # unmark m03, move, Enter
        self.assertEqual(self._run(" \xe0P\xe0H \r"), "m03")  # unmark, move back, re-mark

    def test_escape_keeps_the_default_and_tab_keeps_the_cursor_row(self):
        self.assertEqual(self._run("\xe0P\x1b"), "m03")
        ids = [m for m, _ in self.ROWS]
        sorts = [("a", ids), ("z", ids[::-1])]
        # m00 -> m01, Tab to reversed order (cursor stays on m01), Up -> m02
        self.assertEqual(self._run("\xe0P\t\xe0H\r", default="", sorts=sorts), "m02")

    def test_multi_keeps_pick_order_and_allows_none(self):
        # m03,m05 preselected; toggle m03 off, move to m06 and pick it: order is pick order
        self.assertEqual(self._run(" " + "\xe0P" * 3 + " \r", default="m03,m05", multi=True),
                         "m05,m06")
        self.assertEqual(self._run(" \r", default="m03", multi=True), "")
        self.assertEqual(self._run("\x1b", default="m03,m05", multi=True), "m03,m05")

    def test_loading_label_stays_on_the_last_row_and_is_erased(self):
        import io
        screen = io.StringIO()
        screen.isatty = lambda: True
        with patch.object(cli.sys, "stdout", screen), patch.object(cli.os, "system"), \
                patch.dict(os.environ, {"NO_COLOR": "1"}):
            with cli._loading("Zen models"):
                print("Using the OpenCode CLI")  # output while fetching
            self.assertIs(cli.sys.stdout, screen)
        out = screen.getvalue()
        self.assertEqual(out, "Loading Zen models...\r\033[KUsing the OpenCode CLI\n"
                              "Loading Zen models...\r\033[K")

    def test_off_a_console_it_returns_none(self):
        with patch.object(cli.sys, "stdin", Mock(isatty=Mock(return_value=False))):
            self.assertIsNone(cli._arrow_menu("Models:", self.ROWS, "m03"))


if __name__ == "__main__":
    unittest.main()


class CliCatalogueTests(unittest.TestCase):
    CMDC = ("Available models  ·  3 models\n\nOpen Source\n\n"
            "deepseek/deepseek-v4.1-flash           V4.1 reasoning\n"
            "claude-sonnet-5                        Anthropic\n\n  footer text here\n")

    def test_commandcode_listing_skips_headings(self):
        with patch.object(cli, "_cli_listing", return_value=self.CMDC):
            self.assertEqual(cli._live_model_ids("commandcode"), {"deepseek/deepseek-v4.1-flash", "claude-sonnet-5"})

    def test_zen_uses_opencode_cli_listing(self):
        """The gateway's /models still names retired free models; OpenCode's CLI does not."""
        with patch.object(cli, "_cli_listing", return_value="opencode/big-pickle\nopencode/Nemotron-3-Ultra-Free\n"):
            self.assertEqual(cli._live_model_ids("opencode-zen"), {"big-pickle", "nemotron-3-ultra-free"})

    def test_cli_providers_list_every_served_model(self):
        for provider, served in (("commandcode", {"claude-sonnet-5", "deepseek/deepseek-v4-pro", "new/unknown-model"}),
                                 ("opencode-zen", {"big-pickle", "brand-new-free"})):
            with self.subTest(provider=provider), patch.object(cli, "_live_model_ids", return_value=served), \
                    patch.object(cli, "_model_facts", return_value={}):
                models = cli._load_catalogue(provider)
            self.assertEqual(set(models), served)

    def test_commandcode_plan_refusal_is_not_retried_and_updates_stay_out(self):
        refused = subprocess.CompletedProcess([], 4, stdout="", stderr="Error: 403 MODEL_NOT_IN_PLAN")
        with patch.object(ai_service, "commandcode_executable", return_value="cmdc"), \
                patch.object(ai_service, "_run_cli", return_value=refused) as run:
            with self.assertRaises(RuntimeError) as caught:
                ai_service.commandcode_chat("claude-sonnet-5", "hi")
        self.assertEqual(caught.exception.status_code, 403)
        self.assertIn("--no-auto-update", run.call_args.args[0])
