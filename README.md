# freeride-aggregator

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/rajupramana370/freeride-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/rajupramana370/freeride-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

FreeRide is a live free-model aggregator: one server that speaks both the OpenAI Chat Completions and the Anthropic Messages wires and answers every request with a real completion from a free model, dialled through the gateway. For each prompt it ranks every free model that meets the request's requirements (tools, vision, JSON mode) by the published quality rating and dials the best — JSON prompts go to the strongest JSON-capable model, everything else to the strongest free model overall. When a call is refused it fails over down the ranked chain, reports the refusal trail in an `x-makefun-upstream-failures` header, and cools refused models down for the next few requests (counted in requests, never wall-clock time, so replays are deterministic). Its catalog lists exactly the fourteen free models, on the register call and on both `/v1/models` auth shapes.

## Run it

```sh
node main.ts --list   # print the free-model catalog and exit
node main.ts          # serve: needs MAKEFUN_ENV_URL, MAKEFUN_PROMPT_FILE, OPENAI_BASE_URL, OPENAI_API_KEY
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=raju-pramana&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
