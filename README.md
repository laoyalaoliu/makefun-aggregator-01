# makefun-aggregator-01

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/laoyalaoliu/makefun-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/laoyalaoliu/makefun-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

A Node 22 server that serves the OpenAI Chat Completions API and the Anthropic Messages API (JSON and streaming) over ten simulated free-tier providers, each speaking its own wire (OpenAI, Anthropic or Gemini). For every request it plans every free model that meets the request's requirements, best quality rating first, walks the plan when a provider refuses (402/429/5xx), learns from a refusal that a provider's budget is spent or that it is out, and returns the provider's answer in the client's wire with its text, usage and stop reason intact. It probes each provider's catalog in the provider's own wire and registers the catalog it built, byte for byte the document `node main.ts --list` prints. Local grade on the kit's three practice trials: score 99 (quality 99, served 100, wire 100, failover 100, catalog 100), every request valid, every replay byte-for-byte identical.

## Run it

```sh
node main.ts --list
MAKEFUN_PROMPT_FILE=<trial brief> MAKEFUN_ENV_URL=<trial env> node main.ts
```

---
<sub>Entered on [make.fun](https://make.fun/?ref=8j9v25ys&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
