# AutoResearch for Reinforcement Learning

## Executive Summary

This project is worth building, but not as "an autonomous RL scientist" in the broadest sense on day one.

The strongest version is:

> An **RL experiment manager** that runs the outer research loop: it proposes experiments, executes training, reads curves and rollouts, diagnoses failure modes, and chooses targeted interventions across reward shaping, curriculum, environment difficulty, and algorithm class.

That framing is stronger than "AutoRL" because most prior AutoRL work focuses on hyperparameter optimization, algorithm selection, or reward search in isolation. It is also stronger than a generic "AI scientist for RL" claim because a solo-buildable version can target a specific and defensible gap: **RL-specific failure analysis and iterative experiment revision under a fixed experiment budget**.

My assessment is that this direction is **partially explored but not taken**. The nearest existing work automates:

- general ML research loops ([Karpathy's `autoresearch`](https://github.com/karpathy/autoresearch), [The AI Scientist](https://arxiv.org/abs/2408.06292), [MLR-Copilot](https://arxiv.org/abs/2408.14033), [Agent Laboratory](https://aclanthology.org/2025.findings-emnlp.320/), [AutoRA](https://joss.theoj.org/papers/10.21105/joss.06839))
- RL subproblems such as reward design or curriculum generation ([AutoRL reward search](https://research.google/pubs/evolving-rewards-to-automate-reinforcement-learning/), [Eureka](https://eureka-research.github.io/), [Text2Reward](https://text-to-reward.github.io/), [DrEureka](https://github.com/eureka-research/DrEureka), [Eurekaverse](https://eureka-research.github.io/eurekaverse/), [CurricuLLM](https://github.com/labicon/CurricuLLM))

What I did **not** find in primary sources is a well-established system that unifies:

- algorithm choice
- reward redesign
- curriculum / environment difficulty adjustment
- failure diagnosis from training traces and behavior
- memory across many RL experiment iterations

under one reproducible outer-loop benchmark.

That is the opening.

## A. What AutoResearch Is

### Primary reading

The clearest primary artifact is Karpathy's repo, not social media summaries:

- [`karpathy/autoresearch`](https://github.com/karpathy/autoresearch)
- [`program.md`](https://github.com/karpathy/autoresearch/blob/master/program.md)
- supplemental context: [No Priors interview with Karpathy, published March 20, 2026](https://www.youtube.com/watch?v=kwSVtQ7dziU)

### What Karpathy seems to mean by it

From the repo README and `program.md`, AutoResearch is a deliberately narrow autonomous experimentation setup:

- a small but real training system
- one main file the agent is allowed to edit
- a fixed evaluation harness and metric
- a fixed time budget per run
- an experiment log
- a keep-or-discard loop
- a human who edits the agent instructions rather than directly editing experiment code

The core repo design is intentionally minimal:

- `train.py` is the editable surface
- `prepare.py` and the evaluator are fixed
- `program.md` is the "research org code"
- each experiment is committed, run, evaluated, logged, and either kept or reverted

### Core loop

1. Define a constrained experimental arena.
2. Let the agent modify the experiment implementation.
3. Run a real training job under a fixed budget.
4. Read a fixed metric.
5. Keep improvements, discard regressions.
6. Repeat many times.

### Key design principles

- **Tight scope beats open-endedness.** One editable file, one metric, one hardware target.
- **Comparable experiments matter.** The fixed 5-minute budget is a core normalization device.
- **Agent autonomy is real only if it runs experiments, not just suggests ideas.**
- **The prompt is part of the system.** Karpathy treats `program.md` as the human-controlled research policy.
- **Logs and reversibility matter.** The branch advances only on wins.

### Central vs optional

Central:

- real execution
- iterative keep/discard loop
- constrained edit surface
- fixed evaluator
- experiment memory / logging

Optional or contingent:

- multi-agent orchestration
- paper writing
- literature review
- broad autonomy across many files / repos
- fully open-ended idea generation

### Important inference

Inference: Karpathy's main bet is not merely "LLMs can tune models." It is that **an autonomous outer loop can accumulate many small, compounding experimental improvements if the search surface and evaluation loop are sharply engineered**. That design philosophy transfers to RL, but RL will need a richer evaluator than a single scalar loss because behavioral failure is harder to see.

## B. Existing Work Related to This Idea

### 1. AutoResearch-like systems for ML research

- [Karpathy `autoresearch`](https://github.com/karpathy/autoresearch): the most relevant conceptual seed. It is not a full science pipeline; it is an autonomous experiment loop around a constrained training setup.
- [The AI Scientist](https://arxiv.org/abs/2408.06292) and [AI Scientist repo](https://github.com/SakanaAI/AI-Scientist): end-to-end idea generation, experimentation, paper writing, and automated review across ML domains.
- [The AI Scientist-v2](https://arxiv.org/abs/2504.08066): pushes toward broader, less template-bound automated ML research via agentic tree search.
- [MLR-Copilot](https://arxiv.org/abs/2408.14033) and [repo](https://github.com/du-nlp-lab/MLR-Copilot): literature-grounded idea generation plus experiment implementation and execution.
- [Agent Laboratory](https://aclanthology.org/2025.findings-emnlp.320/) and [repo](https://github.com/SamuelSchmidgall/AgentLaboratory): literature review, experimentation, and report writing with human-in-the-loop options.
- [AutoRA](https://joss.theoj.org/papers/10.21105/joss.06839) and [docs](https://autoresearch.github.io/autora/): older but important closed-loop automated empirical research framework; emphasizes theorist/experimentalist loops more than LLM agents.
- [MLE-bench](https://openai.com/index/mle-bench/) and [repo](https://github.com/openai/mle-bench): not an autonomous research system, but an important benchmark showing what ML agents can and cannot do on real ML engineering tasks.

Why this matters:

- This bucket shows that autonomous or semi-autonomous ML experimentation is already real.
- It also shows that most systems are still strongest in supervised / general ML engineering settings, not RL-specific outer loops.

### 2. Classical AutoRL and meta-level automation in RL

- [Automated Reinforcement Learning (AutoRL): A Survey and Open Problems](https://arxiv.org/abs/2201.03916)
- [Automated Reinforcement Learning: An Overview](https://arxiv.org/abs/2201.05000)
- [AutoRL.org](https://autorl.org/)
- [ARLO](https://arlo-lib.github.io/arlo-lib/docs-page.html)
- [ARLBench](https://github.com/automl/arlbench)

What this bucket covers:

- hyperparameter optimization
- algorithm selection
- architecture choices
- pipeline configuration
- benchmarking of AutoRL methods

Why this matters:

- Your idea is **not** novel if framed as generic RL automation or RL hyperparameter search.
- The literature already treats those as core AutoRL problems.

### 3. Reward design / reward shaping automation

- [Evolving Rewards to Automate Reinforcement Learning](https://research.google/pubs/evolving-rewards-to-automate-reinforcement-learning/): early AutoRL framing that explicitly treats reward tuning as an outer optimization problem.
- [Eureka](https://eureka-research.github.io/) and [repo](https://github.com/eureka-research/Eureka): LLM-generated reward code with iterative reward reflection; strong evidence that LLMs can be useful in an RL outer loop.
- [Text2Reward](https://text-to-reward.github.io/) and [repo](https://github.com/xlang-ai/text2reward): language-to-dense-reward generation, including iterative refinement with human feedback.
- [REvolve](https://rishihazra.github.io/REvolve/): LLM reward evolution using human feedback.

Why this matters:

- Reward design is one of the most mature and credible subproblems in "LLMs for RL experimentation."
- If your project is only "LLM edits reward functions," it will be derivative unless the evaluation or loop design is materially stronger.

### 4. Curriculum / environment generation

- [CurricuLLM](https://github.com/labicon/CurricuLLM): LLM-generated task curricula, including subtask generation, reward code, and goal distributions.
- [Eurekaverse](https://eureka-research.github.io/eurekaverse/) and [repo](https://github.com/eureka-research/eurekaverse): LLM-generated environment curricula for quadruped parkour, with sim-to-real transfer.

Why this matters:

- Curriculum generation is already a live research area.
- It is strong adjacent evidence that LLMs can modify more than scalar hyperparameters; they can alter the *training distribution*.

### 5. Sim-to-real / broader RL configuration automation

- [DrEureka](https://github.com/eureka-research/DrEureka): extends the reward-design idea into sim-to-real transfer by generating reward functions and domain-randomization settings.

Why this matters:

- This is one of the closest precedents to "outer-loop RL experiment design."
- But it is still scoped to a fairly specific robotics setting rather than a general RL research manager.

## C. Is This Idea Already Taken?

### Short answer

**No, but pieces of it are.**

### Already well explored

- generic AutoRL as HPO / algorithm selection / pipeline tuning
- LLM-based reward generation
- LLM-based curriculum or environment generation
- autonomous ML experimentation in non-RL settings

### Partially explored

- RL outer loops where an LLM proposes modifications and training evaluates them
- sim-to-real configuration search with language models
- human-in-the-loop refinement of reward functions

### Still open

The clearest open gap is:

> A reproducible RL experimentation agent that reasons about **why** training failed and chooses **which intervention type** to try next, instead of only searching within one intervention family.

That means choosing among:

- algorithm switch
- reward revision
- curriculum change
- environment simplification / difficulty scheduling
- evaluation redesign
- termination or reset rule changes

based on evidence from:

- learning curves
- seed variance
- rollout summaries
- reward decomposition
- success/failure cases
- possibly short videos or trajectory snippets

### Grounded assessment

My best assessment is:

- **Research-space novelty:** moderate, if you emphasize RL-specific failure analysis plus multi-axis intervention selection.
- **Portfolio novelty:** strong, if you build a clean, reproducible system instead of a vague autonomous agent demo.
- **Paper novelty:** possible for a workshop or empirical systems paper, but only if you benchmark against serious baselines under equal experiment budgets.

### Important uncertainty

I did not find a primary-source system that exactly matches your proposed unified RL outer loop. That is evidence of whitespace, not proof of absence. The nearest clusters are reward-design systems and general autonomous-ML systems.

## D. Best Project Angle

### Recommended framing

Use this framing:

> **AutoResearch for RL: a failure-analysis and intervention agent for reinforcement learning experiments**

More explicit subtitle:

> An agent that reads RL experiment traces and rollouts, diagnoses likely failure modes, and selects targeted experiment revisions across reward shaping, curriculum, environment difficulty, and algorithm class.

### Why this is the strongest angle

- It preserves the AutoResearch spirit: autonomous outer-loop experimentation.
- It avoids the weak framing of "Optuna for PPO."
- It is RL-specific in a defensible way because RL failure diagnosis is materially harder than loss minimization in supervised learning.
- It differentiates from reward-only systems like Eureka and Text2Reward.
- It is buildable by a solo engineer if the action space is constrained.

### Angles I would avoid

- "General AutoRL platform": too broad, already crowded, likely collapses into HPO.
- "Autonomous RL scientist": too hype-heavy for an MVP.
- "Reward shaping agent" alone: interesting, but closer to existing work.
- "Algorithm selection agent" alone: too close to classical AutoRL.

### Best claim to make

The defensible claim is not:

> "This discovers new RL algorithms."

It is:

> "This improves RL experimentation by running a structured outer loop that performs evidence-based failure diagnosis and targeted revisions under a fixed experiment budget."

## E. MVP Proposal

### MVP goal

Build a constrained system that can outperform naive baselines in **experiment-budgeted RL iteration**, not necessarily in absolute final return.

### Scope

Use:

- environment family: `Gymnasium` + `MiniGrid`
- training stack: `Stable-Baselines3`
- algorithms: `PPO`, `DQN`, `SAC`
- config surface: YAML / JSON experiment specs, not arbitrary code edits

### Suggested benchmark tasks

- `LunarLander-v2`
  - useful for algorithm selection, reward tweaks, exploration, termination choices
- `MountainCarContinuous-v0`
  - useful for sparse-progress diagnosis and shaping
- `MiniGrid-DoorKey-8x8-v0`
  - useful for sparse rewards, curriculum, and rollout inspection

If you want an even tighter first version:

- start with `LunarLander-v2` and `MiniGrid-DoorKey-8x8-v0`

### Allowed intervention types

The agent can choose one intervention per iteration:

- switch algorithm
- adjust core hyperparameters within bounded ranges
- enable or modify reward-shaping terms from a predefined library
- change curriculum stage or environment difficulty
- change evaluation protocol
- stop early on clearly bad runs

This is important: the MVP should **not** let the LLM freely rewrite the full training code.

### Loop design

1. Start from a baseline experiment spec.
2. Train for a fixed budget.
3. Collect:
   - train/eval curves
   - seed variance
   - episode return statistics
   - success rate
   - trajectory summaries
   - short rollout videos or GIFs
4. Run a summarizer that converts raw artifacts into a compact structured report.
5. Ask the agent to:
   - identify the likely failure mode
   - choose one intervention type
   - propose the next spec
   - explain the rationale
6. Run the proposed experiment.
7. Keep or discard based on budget-normalized evaluation.

### Failure taxonomy for the agent

Give the system a fixed vocabulary:

- under-exploration
- unstable learning
- reward misspecification
- sparse reward stall
- curriculum too hard
- algorithm / action-space mismatch
- overfitting to training seed
- deceptive reward / reward hacking

### Inputs and outputs

Inputs:

- environment metadata
- current experiment spec
- experiment history
- summarized metrics
- rollout summaries / media references

Outputs:

- next experiment spec
- predicted failure mode
- intervention type
- concise rationale
- expected risk

### What to log

- experiment ID and parent ID
- config diff
- agent rationale
- predicted failure mode
- training/eval metrics by seed
- area-under-learning-curve
- wall-clock cost
- rollout artifact paths
- keep / discard decision

### Evaluation criteria

Compare against equal-budget baselines:

- fixed human baseline configs
- random search over the same parameter space
- Optuna or another HPO baseline
- ablation: no-failure-analysis agent, only metric-based agent

Success should mean at least one of:

- higher final benchmark score under the same experiment budget
- faster time-to-threshold
- better sample efficiency
- more consistent improvement across seeds

### What success would look like

A real MVP result is:

- within 30 to 100 outer-loop experiments, your agent beats random search and a simple HPO baseline on at least 2 RL tasks under equal compute
- and does so by using **multiple intervention types**, not only LR tuning

That is a meaningful result.

## F. Stretch Roadmap

### v1

- 2 environments
- structured configs only
- curve-based and stats-based diagnosis
- no video understanding yet
- single-agent outer loop

### v2

- add rollout trajectory summaries
- add reward decomposition analysis
- add explicit curriculum decisions
- add experiment memory and retrieval over prior runs

### v3

- add video / frame-based behavior inspection
- detect metric-behavior mismatch
- detect reward hacking or degenerate policies

### v4

- expand to broader benchmark suite
- add held-out tasks
- test transfer of discovered strategies across environments
- multi-agent setup: proposer, critic, analyst, scheduler

### v5

- allow limited code synthesis for reward functions or wrappers
- support robotics simulators or more complex RL suites
- paper-oriented benchmark with budget-controlled comparisons

## G. Risks and Failure Modes

### Conceptual risks

- The project may collapse into HPO if the intervention space is not carefully designed.
- The "reasoning" may be shallow post-hoc narration rather than genuinely useful diagnosis.
- The outer loop may not add enough value beyond simple search baselines.
- "AutoResearch for RL" may sound more novel than the measured contribution really is.

### Technical risks

- RL variance can drown out signal, making keep/discard decisions unreliable.
- Reward changes can confound results and make causal attribution hard.
- Short training budgets may reward hacks that do not scale.
- Behavioral regressions may not show up in scalar metrics.
- Video analysis may add cost and complexity before it adds much value.
- Experiment costs can explode once you add seeds, rollouts, and media.

### Evaluation risks

- If baselines are weak, positive results will not be credible.
- If tasks are too easy, the system will look better than it is.
- If tasks are too hard, the loop may appear useless due to noise.
- If the action space is unconstrained, failures will be hard to debug and reproduce.

### Product / portfolio risks

- A broad "autonomous scientist" narrative may make the work look inflated.
- A narrow reward-only system may look derivative.

## H. Final Recommendation

### Is this worth building?

Yes, **if** you build the constrained, evidence-driven version.

### What is it best suited for?

Best fit:

- **research prototype** first
- **strong portfolio project** second

Not the best initial framing for:

- a top-tier paper attempt out of the gate

It could become a workshop paper or strong empirical systems paper if you benchmark it seriously.

### Exact angle to pursue

Pursue this:

> Build **AutoResearch for RL** as an **outer-loop RL experiment manager** centered on **failure diagnosis and targeted intervention selection**.

Do not pursue this:

> "A general autonomous RL scientist that discovers better agents by itself."

### Why this angle wins

- It is more novel than plain AutoRL.
- It is more defensible than reward-only generation.
- It directly addresses what makes RL hard: noisy signals, sparse rewards, unstable learning, and behavior-metric mismatch.
- It is realistic for a solo builder.

## Build Recommendation

If you start this project, I would scope the first public version as:

> "An agentic RL experimentation loop that reads curves and rollouts, tags failure modes, and revises the next experiment across reward shaping, curriculum, and algorithm choice."

That is the version most likely to be:

- actually buildable
- clearly differentiated
- benchmarkable
- credible to researchers and employers

## Source Notes

Primary sources used for this brief:

- [Karpathy `autoresearch`](https://github.com/karpathy/autoresearch)
- [`program.md`](https://github.com/karpathy/autoresearch/blob/master/program.md)
- [No Priors interview with Andrej Karpathy, March 20, 2026](https://www.youtube.com/watch?v=kwSVtQ7dziU)
- [The AI Scientist](https://arxiv.org/abs/2408.06292)
- [The AI Scientist-v2](https://arxiv.org/abs/2504.08066)
- [MLR-Copilot](https://arxiv.org/abs/2408.14033)
- [Agent Laboratory](https://aclanthology.org/2025.findings-emnlp.320/)
- [AutoRA](https://joss.theoj.org/papers/10.21105/joss.06839)
- [MLE-bench](https://openai.com/index/mle-bench/)
- [AutoRL survey](https://arxiv.org/abs/2201.03916)
- [Automated RL overview](https://arxiv.org/abs/2201.05000)
- [AutoRL.org](https://autorl.org/)
- [Evolving Rewards to Automate RL](https://research.google/pubs/evolving-rewards-to-automate-reinforcement-learning/)
- [Eureka](https://eureka-research.github.io/)
- [Text2Reward](https://text-to-reward.github.io/)
- [DrEureka](https://github.com/eureka-research/DrEureka)
- [CurricuLLM](https://github.com/labicon/CurricuLLM)
- [Eurekaverse](https://eureka-research.github.io/eurekaverse/)
- [REvolve](https://rishihazra.github.io/REvolve/)
