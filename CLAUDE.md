# ai-suite

The workspace's one AI suite: `ai_suite.service` (AIService and its transports),
`ai_suite.providers` (the shared provider/model menu), `ai_suite/config/` (provider configs;
`*.local.json` overrides are local-only) and `.env` (provider API keys, never committed).

## Commands
- Tests: `python -B -m unittest discover -s tests -q`
- Refresh the vendored copies in consumer repos: `python sync.py` (then commit them there)
- Drift check: `python sync.py --check` (run by the workspace `check_workspace.py`)

## Distribution
Consumers import the sibling checkout (`AI_SUITE_DIR` overrides the path). Consumers with
their own GitHub repository carry a vendored `ai_suite/` copy (`sync.TARGETS`) that a fresh
clone falls back to; it is never edited in place. `pip install
"git+https://github.com/augusto-rehfeldt/ai-suite"` also works (`pyproject.toml`).
`service.py` must import on the standard library alone: the calibre plugin packs it into its
zip as `ai_service.py`. `ai_suite/.gitignore` travels with every copy so local configs and
usage state never get committed anywhere.

## Providers
`providers.PROVIDER_CONFIG_MAP` maps a provider name to its config file; anything in
`providers.CATALOGUE_PROVIDERS` also picks a model from that config's `models` block
and remembers it in the caller's state file (default `provider_state.json` here).
At menu time `providers._live_model_ids` asks the provider what it serves
(`/models`, or `cmdc --list-models`) and hides curated ids it has retired;
served ids missing from the config join the menu when models.dev reports their
output limit. Offline, the curated list shows as is.
`providers.choose_ai` is the one provider/model menu for every AIService script:
mathforge and music writer call it with their own state file and roles.
opencode.ai rejects calls without an `x-opencode-session` header;
`AIService._extra_headers` adds one per service.
Menus print context, max output and list price per 1M tokens from models.dev
(`providers._model_facts`; opencode's `~/.cache/opencode/models.json` when under a day
old, else live). Display only: the config's `max_output` is still the cap sent.
With `ARTIFICIAL_ANALYSIS_API_KEY` set, menus also show the Artificial Analysis
Intelligence Index (`providers._intelligence`, cached a day in
`~/.cache/ai-book-creator/artificial_analysis.json`). Rows read
`ctx | $in/$out | AA -> aggregate`: `_scales` puts each metric on 0..1 across the menu
(log for context and output price) and averages them into the aggregate (missing = 0);
`_gradient` colors each from red (kept bright for low vision) to bright green in 24-bit color. `_pick_model`
sorts by price first. On a Windows console the provider and model menus are
`_arrow_menu`: `MENU_ROWS` (20) rows at a time with scrolling, the default preselected,
Up/Down or W/S, PgUp/PgDn move, Space selects/deselects, Enter confirms (the cursor row if
nothing is selected), Esc keeps the default, Tab re-sorts by context, AA, then
aggregate. Off a console (piped, non-Windows) they fall back to the typed numbered menu.

- `claude` — the Claude Code CLI in print mode, on the user's subscription, no
  key. The prompt goes in **on stdin, never in argv**: Windows caps a command
  line at 32k characters and a chapter prompt blows past it. Its catalogue
  carries `max_output: 0` because the CLI takes no completion cap, and the CLI
  branch in `run()` unsets `AI_*_COMPLETION_TOKENS` when it sees a 0.
- `commandcode` — same CLI pattern as `claude` but through the Command Code CLI
  in headless mode (`cmdc -p --output-format text --model ID`, prompt on stdin).
  `cmd` is deliberately avoided: on Windows that is the system shell. Runs on
  the user's Command Code subscription, no key; catalogue model ids must be
  written lowercase (the model menu lowercases picks, and Command Code ids are
  case-sensitive).
- `opencode-zen` — config provider `opencode`: the OpenCode CLI
  (`opencode run --agent build --format json -m opencode/ID`, prompt on stdin).
  opencode.ai's free tier answers only OpenCode itself: direct API calls get
  403 FreeTierError, and so does the CLI with a custom agent or a trimmed/denied
  toolset. So the stock `build` agent runs with `XDG_CONFIG_HOME` pointed at a
  bare config (`OPENCODE_BARE_CONFIG`) whose permissions are all `ask`, which a
  headless run auto-rejects; that also keeps the user's global AGENTS.md and
  plugins out of replies. `plan` was dropped: its reminder made models answer
  with a plan instead of the draft. A 403 is never retried. With `stream` on,
  `timeout` is per chunk, so CLI calls get at least `CLI_TIMEOUT` (1800s) wall clock.
- `hyper` — hyper.charm.land, OpenAI-compatible (`HYPER_API_KEY`). Send it
  `max_tokens`, not `max_completion_tokens`; the newer spelling is a 400.
- `grok` — xAI's own API, OpenAI-compatible (`XAI_API_KEY`, `api.x.ai/v1`).
  Takes the older `max_tokens` spelling like hyper. Its `judge_models` are
  grok models because that is all the endpoint serves — same-family judges,
  so treat their humanness verdicts as weak signal there.
- Every request asks for the model's whole output allowance: the config's
  `models[<id>].max_output` (`_max_output`) when the catalogue lists the model,
  else the larger of the caller's `max_completion_tokens` and the role default.
  A caller's cap is a floor, never a ceiling -- a cap only truncates, and a
  truncated reply costs a whole retry. Retries after truncation double the cap
  but never past `max_output`.
- `generate_content(..., model=...)` overrides the role default. That parameter
  exists for the judges and nothing else.


## Shared service contract

This AIService is the workspace's one AI suite. Consumers: book writer, music writer, mathforge,
bandido, impostor, book-watch, lamplight, article-writer (via book writer's aliases), the calibre summarizer, Shelfscape and youtube-explainer (see
the workspace README's contract section). Expose options on AIService, not
private-method or SDK monkeypatches, and add them test-first in
`tests/test_shared_consumers.py` / `tests/test_shared_portable.py` here. Public:
`allow_auth_prompt`, `client_max_retries`, `config_overrides` (merged over the file,
explicit `api_key` wins; config-only when no file, then `provider` is required),
`providers.provider_config_path`, `set_reasoning_effort(writing, review)`,
`generate_content(system=, temperature=, wait_for_limits=)`, `embed()` and `last_usage`.
Config keys `stream` and `token_param` are consumer-facing too.

The module must import on the standard library alone: the calibre plugin packs this
file into its zip and Calibre's Python has no `requests` or SDKs (`_StdlibSession`).
The CLI transports (`claude_chat`, `commandcode_chat`) run from a neutral directory;
Claude Code also gets `--safe-mode --tools "" --system-prompt CLI_NEUTRAL_SYSTEM`, because a
CLI otherwise reads project/global CLAUDE.md and plugin rules into the reply. Keys never
cross providers (`_resolve_api_key`): overrides with their own endpoint get only their own
key, and a config naming `api_key_env` uses only that variable; the legacy cross-provider
fallback remains only for configs that name none. Override `base_url`, `writing_model`
and `review_model` beat the `AI_*` environment variables; `cap_is_ceiling` sends a
caller's cap as given. Metered requests sharing a
ledger serialize under an OS lock; atomic writes and UsageStateError prevent silent
accounting resets/retries. Run every consumer's workspace check together (command in the workspace AGENTS.md).

`generate_content` never returns a provider's usage-limit notice (`LIMIT_NOTICE_RE`;
the Claude Code CLI prints "You've hit your session limit" as its reply) and waits
out errors matching `LIMIT_ERROR_RE`: until the reset the notice names
(`limit_reset_wait`), else `LIMIT_RETRY` seconds with a 5-hour `LIMIT_PAUSE` every
`LIMIT_TRIES`th try, indefinitely. Only a subscription window (`LIMIT_WINDOW_RE`) earns
the long pause; a gateway rate limit (Zen's free tier 429s) retries every minute.
`_run_cli` kills the whole process tree on timeout (`taskkill /T`): killing only the
`.cmd` shim left node holding the pipes and the call hung forever. The loop sits outside `@accounted` (the body is
`_generate_content_once`) so a wait never holds the ledger lock. Metered budget
stops (`UsageLimitExceeded`, `DailyTokenBudgetExceeded`, `UsageStateError`) still raise.
