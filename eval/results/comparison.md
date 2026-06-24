# Base vs Fine-tuned — Function Calling Comparison

- Prompts: **23**  |  base: `qwen36-base`  |  tuned: `qwen36-tuned`


## Metrics (%)

| Metric | Base | Tuned | Δ |
|---|---:|---:|---:|
| Appropriate call/no-call | 87.0 | 95.7 | +8.7 |
| Function-name accuracy (call cases) | 82.4 | 100.0 | +17.6 |
| Argument exact-match (call cases) | 76.5 | 94.1 | +17.6 |
| Executable against mock tools | 65.2 | 78.3 | +13.1 |

## Per-category appropriate-rate

| Category | n | Base | Tuned |
|---|---:|---:|---:|
| ambiguous | 2 | 0% | 100% |
| clarify | 3 | 100% | 67% |
| multi_arg | 5 | 100% | 100% |
| no_tool | 3 | 100% | 100% |
| process_automation | 4 | 100% | 100% |
| single_tool | 6 | 83% | 100% |

## Examples where tuned improved

- **single-05** (single_tool): _Translate 'good morning' into Japanese._
  - base → `None`
  - tuned → `{'name': 'translate_text', 'arguments': {'text': 'good morning', 'target_language': 'Japanese'}}` (expected `translate_text`)
- **ambig-01** (ambiguous): _Remind me about the Paris weather later._
  - base → `{'name': 'get_weather', 'arguments': {'city': 'Paris'}}`
  - tuned → `{'name': 'set_reminder', 'arguments': {'text': 'Check the weather in Paris', 'datetime': 'later'}}` (expected `set_reminder`)
- **ambig-02** (ambiguous): _I need the latest figures for Tesla._
  - base → `None`
  - tuned → `{'name': 'get_stock_price', 'arguments': {'symbol': 'TSLA'}}` (expected `get_stock_price`)
