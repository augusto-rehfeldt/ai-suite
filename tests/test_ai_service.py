"""AIService behaviour that would corrupt a run without anyone noticing.

Response normalisation and prompt budgeting are the two places where a silent
change shows up as an empty chapter or a 413 halfway through book four.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_suite import service as ai_service


def _service(config: dict) -> ai_service.AIService:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        with patch.dict(os.environ, {"AI_API_KEY": "sk-test"}, clear=False):
            os.environ.pop("AI_BASE_URL", None)
            return ai_service.AIService(config_path=str(path))


class ResponseShapeTests(unittest.TestCase):
    def setUp(self):
        self.service = _service({"provider": "http", "writing_model": "m", "review_model": "m"})

    def test_chat_completion_json(self):
        resp = {"choices": [{"message": {"content": "chapter one"}}]}
        self.assertEqual(self.service._extract_text_from_response(resp), "chapter one")

    def test_anthropic_content_blocks_drop_thinking(self):
        resp = {"content": [{"type": "thinking", "text": "let me plan"},
                            {"type": "text", "text": "the prose"}]}
        self.assertEqual(self.service._extract_text_from_response(resp), "the prose")

    def test_inline_thinking_tags_are_stripped(self):
        resp = {"choices": [{"message": {"content": "<thinking>plan</thinking>the prose"}}]}
        self.assertEqual(self.service._extract_text_from_response(resp), "the prose")

    def test_unknown_shape_returns_empty_rather_than_raising(self):
        self.assertEqual(self.service._extract_text_from_response({"weird": 1}), "")


class PromptBudgetTests(unittest.TestCase):
    def setUp(self):
        self.service = _service({"provider": "http", "writing_model": "m", "review_model": "m"})

    def test_sectioned_prompt_fits_the_budget_by_shrinking_the_largest_section(self):
        big = "word " * 20000
        prompt = self.service.build_sectioned_prompt(
            "Expand this chapter.",
            [("Chapter", big), ("Outline", "short outline")],
            max_prompt_tokens=2000,
        )
        self.assertLessEqual(self.service._estimate_tokens(prompt), 2000)
        self.assertIn("Expand this chapter.", prompt)
        self.assertIn("short outline", prompt)

    def test_usage_is_estimated_when_the_provider_reports_none(self):
        usage = self.service._extract_usage_from_response({}, "prompt here", "answer text")
        self.assertEqual(usage["estimated"], 1)
        self.assertEqual(usage["total_tokens"], usage["input_tokens"] + usage["output_tokens"])

    def test_reported_usage_wins_over_the_estimate(self):
        usage = self.service._extract_usage_from_response(
            {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}, "x" * 4000, "y" * 4000)
        self.assertEqual((usage["input_tokens"], usage["output_tokens"]), (10, 5))
        self.assertEqual(usage["estimated"], 0)


class GroqRateTests(unittest.TestCase):
    def test_daily_token_limit_error_is_recognised_with_its_numbers(self):
        service = _service({"provider": "groq", "writing_model": "m", "review_model": "m"})
        info = service._parse_groq_rate_limit_error(
            "rate_limit_exceeded: tokens per day (TPD) Limit 500000, Used 499000, "
            "Requested 3000. Please try again in 41.5s")
        self.assertEqual(info["metric"], "TPD")
        self.assertEqual((info["token_limit"], info["tokens_used"]), (500000, 499000))
        self.assertEqual(info["retry_after"], 42)

    def test_an_unrelated_error_is_not_read_as_a_rate_limit(self):
        service = _service({"provider": "groq", "writing_model": "m", "review_model": "m"})
        self.assertEqual(service._parse_groq_rate_limit_error("connection reset"), {})


if __name__ == "__main__":
    unittest.main()
