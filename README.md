# ai-suite

Shared AI client for a family of scripts (book writer, mathforge, book-watch, lamplight,
the Calibre summarizer, Shelfscape and others): one `AIService` with every provider
transport (OpenAI-compatible gateways, OpenAI OAuth proxy, Gemini, Groq, the Claude Code,
Command Code and OpenCode CLIs), usage-limit handling, retries and token accounting, plus
the interactive provider/model menu.

```python
from ai_suite import AIService, choose_ai

provider, config, models = choose_ai(state_file=Path("provider_state.json"))
text = AIService(config_path=config).generate_content("Hello", system="Be brief.")
```

## The menu

On a Windows console `choose_ai` asks for the provider, then one model per role, with
arrow-key menus: 20 rows at a time with scrolling, the remembered pick preselected.
Up/Down or W/S move, Space selects or deselects, Enter confirms (the row under the cursor
if nothing is selected), Esc keeps the default, Tab re-sorts models by price, context,
Artificial Analysis index or aggregate score. A "Loading … models" line holds the
bottom row while catalogues are fetched. Piped or off Windows it is a typed numbered
menu; `mode="auto"` asks nothing and reuses the last picks.

Model rows read `ctx | $in/$out | AA -> score` from models.dev and, with
`ARTIFICIAL_ANALYSIS_API_KEY`, the Artificial Analysis index. Defaults worth knowing:
`claude` writes on `claude-opus-5-5`, `openai-oauth` on `gpt-6-sol`.

## Install

- Next to its consumers: keep this checkout beside them as `ai-suite/` (or set
  `AI_SUITE_DIR`); they import it from there.
- As a package: `pip install "git+https://github.com/augusto-rehfeldt/ai-suite"`
  (`[sdk]` extra adds `requests`, `openai`, `google-genai`; the core runs on the standard
  library alone).
- Vendored: consumer repositories carry an `ai_suite/` copy written by `python sync.py`,
  so a fresh clone of any of them runs without this repository.

Provider keys go in `.env` here (or the environment); per-provider overrides with secrets go
in `ai_suite/config/ai_config_<provider>.local.json`. Neither is ever committed.

## Checks

```
python -B -m unittest discover -s tests -q
python sync.py --check
```
