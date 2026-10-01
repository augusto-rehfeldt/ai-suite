import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_suite import providers as cli


class IntelligenceMatchTests(unittest.TestCase):
    def test_matches_reordered_slugs_and_best_variant(self):
        entries = [
            {"slug": "claude-4-5-sonnet", "evaluations": {"artificial_analysis_intelligence_index": 50}},
            {"slug": "claude-4-5-sonnet-thinking", "evaluations": {"artificial_analysis_intelligence_index": 60}},
            {"slug": "gpt-5-4-low", "evaluations": {"artificial_analysis_intelligence_index": 55}},
            {"slug": "gpt-5-4-high", "evaluations": {"artificial_analysis_intelligence_index": 70}},
            {"slug": "gpt-5-4-mini", "evaluations": {"artificial_analysis_intelligence_index": 40}},
            {"slug": "nvidia-nemotron-3-ultra-550b-a55b", "model_creator": {"slug": "nvidia"},
             "evaluations": {"artificial_analysis_intelligence_index": 23}},
        ]
        with patch.object(cli, "_artificial_analysis", return_value=entries):
            self.assertEqual(cli._intelligence("claude-sonnet-4-5"), 60)
            self.assertEqual(cli._intelligence("opencode/gpt-5.4"), 70)
            self.assertEqual(cli._intelligence("gpt-5.4-mini"), 40)
            self.assertEqual(cli._intelligence("nemotron-3-ultra-free"), 23)
            self.assertIsNone(cli._intelligence("gpt-4.5"))

    def test_dated_slugs_and_release_channel_suffixes(self):
        def e(slug, score):
            return {"slug": slug, "evaluations": {"artificial_analysis_intelligence_index": score}}
        entries = [e("mimo-v2-5-0424", 25), e("grok-build-0-1-06-16", 27),
                   e("deepseek-v4-flash-vision", 35), e("gpt-5-1-codex", 24), e("glm-5-1", 26)]
        with patch.object(cli, "_artificial_analysis", return_value=entries):
            self.assertEqual(cli._intelligence("mimo-v2.5-free"), 25)
            self.assertEqual(cli._intelligence("grok-build-0.1"), 27)
            self.assertEqual(cli._intelligence("deepseek-v4-flash-vision-exp"), 35)
            self.assertIsNone(cli._intelligence("gpt-5.1-codex-max"))
            self.assertIsNone(cli._intelligence("glm-5"))

    def test_glued_family_versions_match_spaced_ones(self):
        """AA writes qwen3-5, gpt4free writes qwen-3.5: the same model."""
        entries = [{"slug": "qwen3-5-397b-a17b", "evaluations": {"artificial_analysis_intelligence_index": 45}}]
        with patch.object(cli, "_artificial_analysis", return_value=entries):
            self.assertEqual(cli._intelligence("qwen-3.5-397b-a17b"), 45)

    def test_serving_tiers_share_the_base_score(self):
        def e(slug, score):
            return {"slug": slug, "evaluations": {"artificial_analysis_intelligence_index": score}}
        entries = [e("kimi-k2-7-code", 26), e("hy3", 25), e("grok-4-fast", 30), e("grok-4", 40)]
        with patch.object(cli, "_artificial_analysis", return_value=entries):
            self.assertEqual(cli._intelligence("moonshotai/kimi-k2.7-code-highspeed"), 26)
            self.assertEqual(cli._intelligence("tencent/hy3-paid"), 25)
            self.assertEqual(cli._intelligence("grok-4-fast"), 30)  # its own model when AA lists it


class MissingIntelligenceRefreshTests(unittest.TestCase):
    def run_refresh(self, age, key="k", mids=("unknown-9",)):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "aa.json"
            cache.write_text("[]")
            os.utime(cache, (time.time() - age,) * 2)
            with patch.object(cli, "AA_CACHE", cache), patch.object(cli, "_artificial_analysis", return_value=[]), \
                    patch.dict(os.environ, {"ARTIFICIAL_ANALYSIS_API_KEY": key}), \
                    patch.object(cli.subprocess, "Popen") as popen:
                return cli._refresh_missing_intelligence(mids), popen

    def test_refetches_in_background_at_most_hourly(self):
        started, popen = self.run_refresh(age=2 * 3600)
        self.assertTrue(started)
        popen.assert_called_once()
        self.assertFalse(self.run_refresh(age=60)[0])
        self.assertFalse(self.run_refresh(age=2 * 3600, key="")[0])
        self.assertFalse(self.run_refresh(age=2 * 3600, mids=())[0])


class CliListingTests(unittest.TestCase):
    def test_stale_listing_survives_a_slow_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "models"
            cache.write_text("old-model\n")
            os.utime(cache, (time.time() - 2 * 86400,) * 2)
            with patch.object(cli.subprocess, "run", side_effect=subprocess.TimeoutExpired("cmdc", 30)):
                self.assertEqual(cli._cli_listing(["cmdc", "--list-models"], cache), "old-model\n")


if __name__ == "__main__":
    unittest.main()
