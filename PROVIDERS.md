# LLM Provider Configuration

OpenLegion supports multiple LLM providers through a provider abstraction
layer located at `src/host/providers/`. The right provider is selected
automatically based on the model string you configure.

---

## Supported Providers

| Provider | Model prefix | Auth |
|----------|-------------|------|
| Anthropic (direct SDK) | `anthropic/...`, `claude-...` | API key or OAuth token |
| LiteLLM (100+ providers) | everything else | per-provider API key |

---

## Configuring Claude / Anthropic

### Option 1 — Claude Code subscription (recommended, no per-call cost)

If you have a Claude Pro or Claude Max subscription, you can use your
subscription quota instead of paying per API call.

1. Install the Claude CLI: https://claude.ai/download
2. Run `claude setup-token` to generate a long-lived OAuth token.
3. Add it to your `.env`:

```bash
CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-...
```

Set your default model in `config/mesh.yaml`:

```yaml
llm:
  default_model: "anthropic/claude-sonnet-4-5"
```

The token is recognized automatically — no other configuration needed.

### Option 2 — Standard Anthropic API key

```bash
OPENLEGION_SYSTEM_ANTHROPIC_API_KEY=sk-ant-...
```

Set your default model:

```yaml
llm:
  default_model: "anthropic/claude-sonnet-4-5"
```

### Credential resolution order

When an Anthropic model is used, credentials are resolved in this order:

1. `CLAUDE_CODE_OAUTH_TOKEN` env var (Claude Code subscription)
2. `OPENLEGION_SYSTEM_ANTHROPIC_API_KEY` env var (standard API key)
3. `OPENLEGION_SYSTEM_ANTHROPIC_OAUTH` env var (JSON blob with `access_token`)

---

## Configuring OpenAI

```bash
OPENLEGION_SYSTEM_OPENAI_API_KEY=sk-...
```

```yaml
llm:
  default_model: "openai/gpt-4o-mini"
```

---

## Configuring other providers

All other providers are handled by LiteLLM. Set the appropriate env var:

```bash
OPENLEGION_SYSTEM_GEMINI_API_KEY=...
OPENLEGION_SYSTEM_GROQ_API_KEY=gsk_...
OPENLEGION_SYSTEM_DEEPSEEK_API_KEY=...
OPENLEGION_SYSTEM_MISTRAL_API_KEY=...
OPENLEGION_SYSTEM_XAI_API_KEY=...
```

Use `provider/model-name` format in your config:

```yaml
llm:
  default_model: "gemini/gemini-2.5-flash"
```

For a full list of supported models: https://docs.litellm.ai/docs/providers

### Local models (Ollama — no key needed)

```yaml
llm:
  default_model: "ollama/llama3.3"
```

Make sure Ollama is running locally on port 11434.

---

## Provider abstraction layer

The provider abstraction lives in `src/host/providers/`:

| File | Purpose |
|------|---------|
| `base.py` | Abstract base class (`LLMProvider`, `LLMResponse`, `StreamChunk`) |
| `anthropic.py` | Anthropic SDK provider (API key + OAuth token support) |
| `litellm.py` | LiteLLM fallback provider (all other models) |
| `factory.py` | `get_provider(model, **credentials)` — selects provider by model string |

### Adding a new provider

1. Create `src/host/providers/your_provider.py`
2. Subclass `LLMProvider` from `base.py`
3. Implement `name`, `supports_model()`, `complete()`, and `stream()`
4. Register it in `factory.py` by inserting before `LiteLLMProvider` in `_PROVIDERS`

```python
class MyProvider(LLMProvider):

    @property
    def name(self) -> str:
        return "myprovider"

    def supports_model(self, model: str) -> bool:
        return model.startswith("myprovider/")

    async def complete(self, params: dict) -> LLMResponse:
        ...

    async def stream(self, params: dict) -> AsyncIterator[StreamChunk]:
        ...
```

The existing mesh dispatch in `credentials.py` already routes `anthropic/`
and `claude-` models to `AnthropicProvider` via the `CLAUDE_CODE_OAUTH_TOKEN`
/ `OPENLEGION_SYSTEM_ANTHROPIC_*` credential resolution. All other models
continue to use LiteLLM as before — no changes to the rest of the codebase
are required when adding a new provider that fits the standard pattern.
