"""Offline ledger crash/concurrency and shared-client contract checks."""
import json
import multiprocessing
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ai_suite import service as api


def service(folder):
    config = Path(folder) / 'config.json'
    return api.AIService(str(config), str(Path(folder) / 'usage.json'),
                         allow_auth_prompt=False, client_max_retries=0)


def record_many(folder):
    client = service(folder)
    for _ in range(10):
        client._record_openai_usage('mini', {'total_tokens': 3})


class ServiceContractTests(unittest.TestCase):
    def test_concurrent_writers_and_corrupt_state_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'config.json').write_text(json.dumps({'provider': 'openai', 'api_key': 'fixture',
                'groq_rate_state_path': str(path / 'groq.json')}))
            with ThreadPoolExecutor(max_workers=3) as pool:
                list(pool.map(record_many, [folder] * 3))
            workers = [multiprocessing.get_context('spawn').Process(target=record_many, args=(folder,)) for _ in range(2)]
            for worker in workers: worker.start()
            for worker in workers:
                worker.join(30)
                if worker.is_alive():
                    worker.terminate()
                    worker.join()
                self.assertEqual(worker.exitcode, 0)
            client = service(folder)
            self.assertEqual(client.get_budget_status()['buckets']['mini']['tokens'], 150)
            before = (path / 'usage.json').read_bytes()
            with patch.object(api.os, 'replace', side_effect=OSError('disk failure')):
                with self.assertRaises(api.UsageStateError):
                    client._record_openai_usage('mini', {'total_tokens': 5})
            self.assertEqual((path / 'usage.json').read_bytes(), before)
            (path / 'usage.json').write_text('{broken')
            with self.assertRaises(api.UsageStateError): service(folder)
            self.assertEqual((path / 'usage.json').read_text(), '{broken')

    def test_public_options_reach_responses_and_do_not_prompt(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'config.json').write_text(json.dumps({'provider': 'openai', 'api_key': 'fixture',
                'groq_rate_state_path': str(path / 'groq.json')}))
            client = service(folder)
            client.set_reasoning_effort('high', 'low')
            client.client = Mock()
            client.client.responses.create.return_value = SimpleNamespace(output_text='answer',
                usage={'input_tokens': 1, 'output_tokens': 2, 'total_tokens': 3})
            self.assertEqual(client.generate_content('question', model_type='review', max_completion_tokens=50, max_retries=1), 'answer')
            sent = client.client.responses.create.call_args.kwargs
            self.assertEqual(sent['reasoning'], {'effort': 'low'})
            # no catalogue for this model: a caller's cap is only a floor under the role default
            self.assertEqual(sent['max_output_tokens'], client._default_completion_tokens('review'))
            requests_before = client.client.responses.create.call_count
            with patch.object(api.os, 'replace', side_effect=OSError('disk failure')):
                with self.assertRaises(api.UsageStateError):
                    client.generate_content('question', max_retries=3)
            self.assertEqual(client.client.responses.create.call_count, requests_before + 1)
            with patch('builtins.input', side_effect=AssertionError('must not prompt')):
                self.assertFalse(client._handle_401_auth_error())
            with patch.object(api, 'OpenAI') as constructor:
                api.AIService._init_client(client)
                self.assertEqual(constructor.call_args.kwargs['max_retries'], 0)
            client.provider = 'openai-oauth'
            client.client = Mock()
            client.client.chat.completions.create.return_value = {'choices': [{'message': {'content': 'answer'}}]}
            client.generate_content('question', model_type='review')
            # the ChatGPT sign-in proxy reads only `reasoning_effort`; the nested object was dropped unread
            self.assertEqual(client.client.chat.completions.create.call_args.kwargs['extra_body'], {'reasoning_effort': 'low'})

    def test_effort_goes_out_in_the_shape_each_endpoint_reads(self):
        with tempfile.TemporaryDirectory() as folder:
            def made(**config):
                # the menu's picks arrive through the environment
                with patch.dict(api.os.environ, {'AI_WRITING_EFFORT': 'high', 'AI_REVIEW_EFFORT': 'low'}), \
                        patch.object(api, 'claude_executable', return_value='claude'):
                    client = api.AIService(None, str(Path(folder) / 'usage.json'), allow_auth_prompt=False,
                                           config_overrides={'api_key': 'fixture', 'writing_model': 'w', 'review_model': 'r',
                                                             'groq_rate_state_path': str(Path(folder) / 'groq.json'), **config})
                client.client = Mock()
                client.client.chat.completions.create.return_value = {'choices': [{'message': {'content': 'answer'}}]}
                return client

            def sent(client, **kwargs):
                client.generate_content('question', **kwargs)
                return client.client.chat.completions.create.call_args.kwargs.get('extra_body')

            # OpenRouter itself takes the nested object; other hosts on its branch read reasoning_effort
            self.assertEqual(sent(made(provider='openrouter', base_url='https://openrouter.ai/api/v1')),
                             {'reasoning': {'effort': 'high'}})
            go = made(provider='openrouter', base_url='https://opencode.ai/zen/go/v1')
            self.assertEqual(sent(go, model_type='review'), {'reasoning_effort': 'low'})
            # the effort was picked for the role's own model: a judge on another model keeps its default
            self.assertIsNone(sent(go, model_type='review', model='judge'))
            # an override naming a role's model, in any case, takes that role's effort
            self.assertEqual(sent(go, model='R'), {'reasoning_effort': 'low'})
            self.assertEqual(sent(made(provider='grok', base_url='https://api.x.ai/v1')), {'reasoning_effort': 'high'})
            cli = made(provider='claude')
            with patch.object(api, 'claude_chat', return_value='done') as chat:
                cli.generate_content('question')
                cli.generate_content('question', model='judge')
            self.assertEqual([c.kwargs.get('effort') for c in chat.call_args_list], ['high', None])
            self.assertFalse(made(provider='minimax').set_reasoning_effort('high'))

    def test_usage_limit_is_waited_out_never_returned(self):
        client = object.__new__(api.AIService)
        client.provider_label = 'claude'
        # a subscription window earns the long pause; a gateway rate limit never does
        replies = (["You've hit your session limit"] * 5
                   + [RuntimeError('HTTP 429 Rate limit exceeded. Please retry after a brief wait.')] * 5
                   + ['chapter text'])

        def once(*args):
            reply = replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply

        client._generate_content_once = once
        with patch.object(api.time, 'sleep') as sleep:
            self.assertEqual(client.generate_content('p'), 'chapter text')
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [api.LIMIT_RETRY] * 4 + [api.LIMIT_PAUSE] + [api.LIMIT_RETRY] * 5)

        # prose that merely mentions a limit, other errors and metered budget stops pass through
        client._generate_content_once = lambda *a: 'She had hit the rate limit of her patience.'
        self.assertEqual(client.generate_content('p'), 'She had hit the rate limit of her patience.')
        for error in (ValueError('bad JSON'), api.IncompleteGenerationError('cut')):
            client._generate_content_once = Mock(side_effect=error)
            with self.assertRaises(type(error)):
                client.generate_content('p')

        # a stated reset is waited for directly, in its own zone, rolling over to tomorrow
        notice = "You've hit your session limit · resets 4pm (America/Argentina/Buenos_Aires)"
        at = api.datetime(2026, 9, 24, 16, 33, tzinfo=timezone.utc)  # 13:33 in Buenos Aires
        self.assertEqual(api.limit_reset_wait(notice, at), 2 * 3600 + 27 * 60 + 60)
        self.assertEqual(api.limit_reset_wait('resets 9:30am', api.datetime(2026, 9, 24, 10, 0)), 23.5 * 3600 + 60)
        self.assertIsNone(api.limit_reset_wait("You've hit your session limit"))


if __name__ == '__main__':
    unittest.main()
