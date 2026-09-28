import unittest
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


if __name__ == "__main__":
    unittest.main()
