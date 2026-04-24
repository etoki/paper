# Big Five -> University Entrance Exam Outcome Agent

## Overview

Generative agent simulation predicting Japanese university entrance exam
outcomes (hensachi) from each participant's Big Five personality scores.

## Design

| Component | Choice | Rationale |
|---|---|---|
| Model | `claude-opus-4-7` | Strongest available Claude, 1M context |
| Prompt language | English | Paper is English; prompt-quoting in Methods |
| Extended thinking | `adaptive`, effort=`high` | Deep 3-stage reasoning inside hidden block |
| Output format | Tool Use (`submit_prediction`) | Schema-guaranteed JSON, no regex parsing |
| Prompt caching | system prompt, ephemeral | ~90% input cost reduction across calls |
| Temperature | 1.0 (required by thinking) | N=30 sample diversity from stochasticity |
| Samples / participant | N=30 | Stable per-ID distribution estimate |

### Methodological safeguards

- **No effect-size hints in prompt.** The agent receives only Big Five values,
  the cohort context, and the hensachi scale. Causal inference (C -> learning
  -> outcome) is left entirely to pre-trained knowledge.
- **Three-stage reasoning** (backstory -> learning behavior -> outcome)
  mandated in the user prompt, executed inside extended thinking.
- **Forced structured output** via `submit_prediction` tool schema;
  retry loop re-invokes if the model fails to call the tool.

## Files

```
simulation/agent/
├── prompts.py        # System + user prompts, PREDICTION_TOOL schema
├── agent.py          # run_one(): Opus 4.7 + thinking + tool call
├── run_pilot.py      # 10 participants x 30 samples = 300 calls
├── baselines.py      # Random / OLS-C / OLS-BigFive baselines
├── reliability.py    # ICC + per-subject spread
├── load_env.py       # Minimal .env loader (no dotenv dependency)
├── .env.example      # Template (committed)
└── .env              # Real key (gitignored)
```

## Execution

```bash
# 1. API key (one-time setup)
cd simulation/agent
cp .env.example .env
# Edit .env, add ANTHROPIC_API_KEY=sk-ant-...

# 2. Smoke test (1 call, ~$0.10)
python agent.py

# 3. Baselines (no API calls, instant)
python baselines.py

# 4. Pilot (300 calls, ~$30-40, ~15-30 min)
python run_pilot.py

# 5. Reliability analysis (after pilot)
python reliability.py
```

## Full run

After pilot validation, change `N_PARTICIPANTS = 103` in `run_pilot.py`
(or create `run_full.py`). Expected: 103 x 30 = 3090 calls, ~$320-400,
1-2 hours. Consider Anthropic Message Batches API (50% discount) for cost.

## Outputs

- `simulation/data/agent_pilot_results.csv` : per-call row with Big Five,
  parsed prediction, backstory/reasoning, token usage, stop_reason
- `simulation/data/baseline_results.json` : three baselines' metrics
- `simulation/data/reliability_results.json` : ICC + spread summary
