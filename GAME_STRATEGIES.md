# Game Strategies Guide

This document explains the game-theory-inspired strategies available in the autoresearch-dgx-local meta-research framework. Each strategy orchestrates multiple research branches that cooperate and/or compete to find the lowest `val_bpb`.

## Architecture

```
Layer 3: Meta-Agent (optional)  →  tunes game parameters
Layer 2: Orchestrator           →  manages branches, strategies, migration
Layer 1: Research Agent         →  modifies train.py, runs experiments
Layer 0: Training               →  PyTorch training loop (train.py)
```

The orchestrator is deterministic Python — it schedules which branch runs, for how many experiments, and handles between-round operations (migration, adoption). The research agent is an LLM (via Ollama + Claude Code) that autonomously modifies hyperparameters and evaluates results.

## Mode: Base

```bash
bash run-dgx-game.sh --mode base
```

The original autoresearch workflow, unchanged. Single branch, single agent, runs indefinitely. No game theory. Use this as your baseline.

## Mode: Island Model

```bash
bash run-dgx-game.sh --mode island
```

Multiple research branches evolve independently, with periodic migration of successful hyperparameters between them. Inspired by island model evolutionary algorithms.

### How it works

1. **Setup**: K branches are created, each assigned a focus area (e.g., architecture, optimization, data handling)
2. **Rounds**: Each branch gets N experiments per round. The agent is told to focus on its assigned area but can explore adjacent ideas
3. **Adaptive allocation**: After each round, branches that improved more get a larger experiment budget next round. Stagnating branches get fewer (but never below `min_budget_ratio`)
4. **Migration**: After all branches run, the best-improving branch's hyperparameters are partially migrated to other branches. The migration rate controls how many parameters are transferred

### Key parameters

| Parameter | Default | Effect |
|---|---|---|
| `migration_rate` | 0.3 | Fraction of differing params to migrate (0 = no sharing, 1 = full copy) |
| `migration_select` | best_delta | Which params to migrate: `best_delta` (most changed) or `random` |
| `adaptive_allocation` | true | Adjust budgets based on improvement rates |
| `min_budget_ratio` | 0.2 | Floor: no branch gets less than 20% of mean budget |

### When to use

Best when you want broad exploration across different optimization axes. The migration mechanism lets branches benefit from each other's discoveries without losing their own exploration direction.

### Tuning tips

- **High migration_rate (0.5+)**: Branches converge faster. Good if you want to exploit a known-good region
- **Low migration_rate (0.1–0.2)**: More diversity preserved. Good for broad exploration
- **Disable adaptive_allocation**: Set to `false` for equal time per branch, useful early when you don't know which direction is most promising

## Mode: Multi-Armed Bandit (UCB1)

```bash
bash run-dgx-game.sh --mode bandit
```

Treats each exploration strategy as a slot machine arm. Uses the UCB1 algorithm to balance exploration (trying less-tested strategies) with exploitation (doubling down on what's working).

### How it works

1. **Warmup**: Each arm gets `warmup_rounds` rounds of experiments (round-robin)
2. **UCB1 selection**: After warmup, the arm with the highest UCB1 score is selected each round:
   ```
   UCB1(arm) = mean_reward + C * sqrt(ln(total_plays) / plays_for_this_arm)
   ```
3. **Reward**: Defined as `max(0, previous_best_bpb - new_best_bpb)` — only improvements count
4. **No migration**: Arms are independent. This mode tests which exploration direction yields the most improvement

### Key parameters

| Parameter | Default | Effect |
|---|---|---|
| `exploration_constant` | 1.414 | UCB1 C parameter. Higher = more exploration, lower = more exploitation |
| `warmup_rounds` | 2 | Minimum rounds per arm before UCB1 selection begins |
| `experiments_per_round` | 3 | Experiments per arm per round |

### When to use

Best when you want to discover which type of change yields the most improvement. Unlike island model, there's no sharing between arms — this is pure competition for resources.

### Tuning tips

- **High C (2.0+)**: Strongly favors exploration — arms that haven't been tried recently get priority
- **Low C (0.5–1.0)**: Favors exploitation — quickly concentrates on the best-performing arm
- **C = 1.414 (sqrt(2))**: The theoretical optimum for UCB1, a balanced default

## Mode: Iterated Coopetition

```bash
bash run-dgx-game.sh --mode coopetition
```

Two branches alternate rounds. After each pair of rounds, the loser must adopt parameters from the winner, creating a ratchet effect where good ideas propagate while maintaining competitive pressure.

### How it works

1. **Setup**: Two branches (alpha, beta) created from the same base
2. **Rounds**: Each branch gets `experiments_per_round` experiments, alternating
3. **Comparison**: After both run, improvement rates are compared. The branch with better improvement rate wins
4. **Adoption**: The loser must adopt `adoption_count` hyperparameters from the winner. The winner is unchanged
5. **Ratchet**: Over time, both branches incorporate each other's best ideas while maintaining distinct exploration paths

### Key parameters

| Parameter | Default | Effect |
|---|---|---|
| `adoption_count` | 2 | Number of params loser adopts from winner |
| `adoption_select` | best_delta | Which params: `best_delta` (most different), `random`, `worst_own` (where loser is weakest) |
| `experiments_per_round` | 4 | Experiments per branch per round |

### When to use

Best for head-to-head comparison of two research directions with forced knowledge transfer. The adoption mechanism creates healthy competitive pressure.

### Tuning tips

- **High adoption_count (3+)**: Aggressive transfer — loser quickly converges toward winner's approach
- **Low adoption_count (1)**: Gentle transfer — branches maintain more independence
- **`worst_own` selection**: Loser replaces its worst-performing parameters, most targeted
- **`random` selection**: Introduces more randomness, can break out of local optima

## Meta-Agent (Layer 3)

Any mode can optionally enable the meta-agent — an LLM that reviews cross-branch results between rounds and tunes the game parameters.

```yaml
meta_agent:
  enabled: true
  interval_rounds: 5   # run every 5 rounds
```

The meta-agent can adjust:
- **Island**: `migration_rate`, `min_budget_ratio`
- **Bandit**: `exploration_constant`
- **Coopetition**: `adoption_count`

It uses the same Ollama model as the research agents. Enable it when you want fully autonomous long-running experiments where even the game rules evolve.

## Comparing Strategies

Run the same experiment under different strategies and compare leaderboards:

```bash
# Edit game_config.yaml: set tag: "island-test"
bash run-dgx-game.sh --mode island
# Let it run for several rounds, then Ctrl+C

# Edit game_config.yaml: set tag: "bandit-test"
bash run-dgx-game.sh --mode bandit
# Let it run, then compare results/ directories
```

Each run uses a different tag, so branches and results are isolated. Compare the `results/<branch>.tsv` files across runs.

## Configuration Reference

See `game_config.yaml` for the complete schema with all parameters and defaults. The file is self-documenting with inline comments.
