<h1 align="center">
Building a Pacman Game Agent with <em>AReaL</em>
</h1>

<p align="center"><em>From a vision-language model to a long-horizon game agent</em></p>

This post explores a practical question: how can we train a vision-language model (VLM)
to become an interactive agent that acts coherently over an entire game, rather than
answering a single static prompt? We study this question through Pacman.
[MaaPacman](https://codehub-dg-g.huawei.com/AgenticRLGaming/MaaPacman) provides the
controlled game environment and interaction adapters, while AReaL provides the rollout
and distributed reinforcement-learning infrastructure.

The model repeatedly observes the game, selects an action from the current state's
admissible-action set, and learns from consequences that unfold across a complete game
trajectory. Our goal is not simply to maximize a Pacman score, but to understand the
task definition, feedback design, and systems support needed to train long-horizon
visual agents reliably.

## 1. Motivation: Why Games, and Why Start with Pacman?

Games provide a natural playground for studying these agents. They retain the central
difficulty of long-horizon interaction—each action changes future observations and
outcomes—while providing explicit rules, measurable results, and resettable
environments. Games therefore occupy a useful middle ground: they are more interactive
than static benchmarks, but more controllable and repeatable than the physical world.

This role of games as an AI testbed is well established. The
[Arcade Learning Environment](https://arxiv.org/abs/1207.4708) introduced Atari games as
a common platform for evaluating general agents. More recent work has expanded the
setting: Google DeepMind's
[SIMA](https://deepmind.google/blog/sima-generalist-ai-agent-for-3d-virtual-environments/)
maps screen images and language instructions to keyboard and mouse actions across
multiple 3D games, while OpenAI's [Video PreTraining](https://openai.com/index/vpt/)
learns Minecraft behavior from gameplay videos and is fine-tuned for tasks requiring
long sequences of actions. Microsoft Research's
[Muse](https://www.microsoft.com/en-us/research/blog/introducing-muse-our-first-generative-ai-model-designed-for-gameplay-ideation/)
takes a different direction by learning game dynamics from visual frames and controller
actions for gameplay generation and ideation. These projects do not solve the same
problem, but together they show why games are a useful laboratory for models that
perceive, predict, and act.

Pacman is an intentionally compact starting point. Its visual scene and action mechanics
are far smaller than those of Minecraft or a modern 3D game, which makes rollout,
training, and diagnosis more affordable. Yet completing a maze can still require
hundreds of dependent decisions: each move changes the next observation and the
currently available choices, while the final outcome may depend on decisions made much
earlier.

Its compactness also lets us build an inspectable experimental environment. We can reset
episodes reproducibly, capture visual observations, validate state-dependent actions,
record structured events, replay failures, and create new maze variants. Pacman is
therefore not the final destination, but a controlled bridge from a static VLM to a
long-horizon visual agent.

## 2. Problem Definition

How can we train a vision-language model (VLM) to perceive, decide, and act in a
long-horizon game through a sequence of grounded decisions?

Unlike a static vision-language task, Pacman requires a closed interaction loop. Each
model output changes the environment and therefore determines the next observation, the
next admissible-action set, and the agent's eventual outcome. One game may contain
hundreds of such dependent decisions.

### 2.1 Two policy settings

We study two policy settings that differ in the action abstraction assigned to the
model: a **primitive-action policy** and a **harness-mediated option policy**. In our
experimental curriculum, Stage I pairs the primitive-action policy with a ghost-free
environment, while Stage II pairs the harness-mediated option policy with ghost-enabled
play. These stage labels describe the capability curriculum, not Pacman's numbered game
levels. The staged design first isolates visual maze grounding and local navigation,
then introduces dynamic hazards and higher-level objectives.

To keep the action terminology precise, we use **global action vocabulary** `V` for a
fixed set of action identifiers, each represented by one output token, and
**state-dependent admissible-action set** `A(s_t) ⊆ V` for the identifiers permitted at
decision `t` by the environment or deterministic option harness. Constrained decoding
restricts sampling to `A(s_t)`; we reserve **policy support** for the
probability-theoretic set of actions assigned nonzero probability after the decoding
rule is applied, rather than use it as another name for the admissible-action set.
Throughout this post, the action vocabulary is fixed; the admissible-action set is what
changes. For the primitive-action policy, `V` contains four direction identifiers. For
the harness-mediated option policy, it is a fixed, bounded vocabulary of high-level
option identifiers. In both cases, the policy must sample from the current `A(s_t)`
rather than the full action vocabulary.

| Setting                            | Model output                                             | Admissible-action set at one decision                       | Execution unit                                     | Stage in this study      |
| ---------------------------------- | -------------------------------------------------------- | ----------------------------------------------------------- | -------------------------------------------------- | ------------------------ |
| **Primitive-action policy**        | One primitive-action identifier                          | Identifiers for directions open in the current state        | One primitive environment step                     | Stage I · ghost-free     |
| **Harness-mediated option policy** | One identifier for a harness-generated high-level option | Identifiers for the options advertised in the current state | A bounded, revalidated sequence of primitive steps | Stage II · ghost-enabled |

In the primitive-action policy, one model decision directly controls one game step.
Constrained decoding removes directions that are blocked in the current state, but the
model remains responsible for selecting the next low-level move.

In the harness-mediated option policy, a deterministic option harness instantiates a
small set of harness-generated high-level options from predefined strategy families:
collect pellets, avoid danger, or pursue an edible ghost. Only the strategy families are
predefined; each option's concrete target, route, and availability are recomputed from
the current game state. We refer to each resulting choice as a **harness-generated
high-level option**. Only identifiers for the currently advertised options enter
`A(s_t)`.

After the model selects an advertised option, the option harness executes its primitive
moves and checks after every step whether the option has completed or become invalid.
This temporal action abstraction shifts the learned decision from **which direction
should Pacman move next?** to **which high-level objective should Pacman pursue next?**
Low-level route construction and safety validation remain deterministic parts of the
option harness.

Across both settings, the task requires:

1. **Visual grounding:** extract task-relevant information from the current game screen.
1. **Closed-loop decision-making:** adapt each decision to the state produced by
   previous actions.
1. **State-dependent action admissibility:** select only an identifier in `A(s_t)`. In
   the harness-mediated option policy, both this admissible-action set and each
   advertised option's grounded target may change between decisions.
1. **Long-horizon decision-making:** account for consequences that may appear many
   decisions later.
1. **Generalization:** transfer the learned policy to unseen game states and maze
   layouts rather than memorize one trajectory.

## 3. Challenges

### 3.1 Sparse reward cannot bootstrap missing perception and planning

Our earliest probes asked whether the base VLM could first recover the game state from
pixels. On three real live-demo frames, it identified Pacman's exact row and column in
`0/3` cases, recovered the complete `OPEN` and `BLOCKED` direction sets in `0/3`, and
produced a valid 25-by-21 ASCII maze in `0/3`. Its mean absolute error when counting the
remaining pellets was 48. The model could recognize the scene as Pacman while still
failing to ground the state needed for control.

This creates a cold-start problem for reinforcement learning. In an early strict
image-only, no-training pilot, none of 32 rollouts completed the maze. Every episode ran
to the step limit and produced the same negative episode return. With no within-group
return variation, group-relative normalization would assign zero advantage to every
sample and provide no policy-gradient signal. Increasing the rollout horizon cannot
repair missing visual grounding or planning; it may only make the same uninformative
trajectories longer and more expensive.

The seed-0 replay below gives a qualitative view of the resulting control failure. The
original, untrained Qwen3.5-9B policy makes little useful progress and eventually
terminates as `STUCK`. The replay illustrates the behavioral consequence of weak visual
grounding and planning; it does not by itself isolate which component caused each bad
decision.

<div align="center">
  <video src="https://github.com/user-attachments/assets/2745b846-65d1-47a2-be22-8b79ee217ca1" controls preload="metadata" playsinline width="72%" poster="../assets/demos/pacman-base-seed0-poster.png"></video>
  <br>
  <a href="../assets/demos/pacman-base-seed0.mp4">Download the versioned base-model MP4</a>
</div>

This cold-start barrier motivates a curriculum-learning-like progression: introduce
visual grounding, local action selection, and longer-horizon planning in stages rather
than demand full-game competence from the initial policy.

### 3.2 Long feedback cycles and GPU memory requirements

Consider an experimental run with 49 optimizer updates, each requiring 48 rollout
episodes. From the first rollout of Iter1 to the last rollout of Iter49, the end-to-end
process—including the optimizer updates performed between rollout batches—took 46.21
hours.

If the architecture or experimental design contains a subtle bug, the learning curve may
not reveal it for many hours, by which point substantial compute has already been spent.

The experiments also have substantial GPU memory requirements. The vLLM rollout engine,
FSDP actor, and optional reference model share the available compute and memory on the
same 8-GPU node. Long rollout episodes also increase trajectory-storage and
log-probability costs. Model offloading, memory-aware batching, distributed trajectory
processing, and checkpoint retention are therefore part of the training design, not
optional infrastructure polish.

### 3.3 Admissible actions are not enough for correct training

Constrained decoding keeps model output inside the current state-dependent
admissible-action set. But a rollout containing only admissible actions can still lead
to an incorrect PPO update if rollout and training use different admissible-action sets
or temperatures to score the chosen action.

PPO can then mistake a bookkeeping mismatch for a policy change. Valid samples may be
clipped or rejected, and training may update inadmissible actions—even though every
executed action was admissible and the loss remains finite. Section 4.3 gives the
precise formulation and implementation contract.

### 3.4 Credit assignment has two axes: timescale and episode weight

Long-horizon learning raises two related but distinct questions: what learning signal
should each decision receive, and how much total optimization weight should each episode
receive?

Episode-level feedback aligns learning with the whole-game outcome but gives every
decision in an episode the same coarse task signal. Step- or option-level feedback is
more precise in time, but may miss consequences that appear later. These feedback
timescales are complementary rather than interchangeable.

Separately, a flat mean over trained tokens can give a longer trajectory more influence
simply because it contains more model decisions. The objective must therefore preserve
the intended feedback timescale without letting episode length determine episode weight.
Section 4.5 specifies the implemented contracts.

## 4. Method

### 4.1 System architecture: MaaPacman as the RL environment adapter

The simplest way to read Figure 1 is from left to right. MaaPacman's environment layer
sits between the RL workflow and concrete Pacman game instances. Toward the workflow, it
exposes a Gym-style interface: `reset(seed)` starts an episode, `step(action)` returns
the next RGB observation, base reward, termination or truncation flags, and structured
game metadata, and `render()` exposes the current frame. Toward the game, each
environment object manages an isolated `pacman-python` game instance. The workflow
therefore sees a stable environment interface rather than game-process details.

![MaaPacman–AReaL runtime architecture](../assets/figures/maapacman_areal_architecture.png)

*Figure 1. Each MaaPacman environment object presents a Gym-style interface to the RL
workflow and delegates actions to an isolated Pacman game instance. The rollout policy
selects actions; the FSDP actor learns from grouped trajectories and never controls the
game directly.*

During rollout, the workflow sends the RGB observation and current state-dependent
admissible-action set to the policy. Under the harness-mediated option policy, it also
sends metadata describing each advertised option. The policy returns an action
identifier; the environment or option harness resolves it to the corresponding primitive
action or grounded option. The figure uses the legacy label “legal option support”; in
this post, that label denotes `A(s_t)`, not policy support. The environment returns game
events and terminal state; the `areal-pacman` workflow applies the experiment-specific
reward function and assembles the interaction records into grouped trajectories. AReaL
trains on those trajectories and publishes updated weights back to the rollout policy.

This conceptual split maps to the repositories as follows:

| Repository      | Responsibility                                                                                                                                    | Deliberate boundary                                       |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| `pacman-python` | Concrete game instances: Pygame transitions, RGB frames, structured state, and terminal events                                                    | Game semantics only; no RL interface                      |
| `MaaPacman`     | Gym-style headless adapter, isolated game workers, and deterministic option harness                                                               | Environment interaction and planning; no distributed RL   |
| `areal-pacman`  | Environment-to-AReaL workflow, prompts, reward function, configuration, evaluation, and experiment records                                        | Experiment semantics; no generic FSDP/vLLM implementation |
| `AReaL`         | Asynchronous rollout, PPO/GRPO, vLLM, FSDP, checkpointing, distributed data movement, and our training-side constrained-log-probability extension | No game-environment or option-generation logic            |

This layout lets us test game events, reward semantics, and distributed policy updates
independently. It also prevents a common architecture error: drawing a direct control
edge from the training actor to the game.

### 4.2 Harness-mediated option policy: let the model choose intent

Under the harness-mediated option policy, the model selects a high-level intent rather
than an individual movement direction. At each decision point, the deterministic option
harness instantiates a bounded set of state-valid choices from three predefined strategy
families:

- **Collect pellets:** move toward an approved pellet target;
- **Avoid danger:** move toward a safe escape anchor;
- **Pursue an edible ghost:** approach a vulnerable ghost.

Each instantiated option includes a target, first primitive action, bounded commitment,
and safety metadata. The VLM selects one option identifier from the current
admissible-action set; it does not emit a precomputed action list. The deterministic
harness uses breadth-first search (BFS) over the maze topology and handcrafted safety
checks to ground that intent into primitive moves. `COLLECT`, `AVOID`, and `ELIMINATE`
options are capped at `min(8, d)`, `min(3, d)`, and `min(6, d)` steps, respectively,
where `d` is the current shortest-path distance. After every environment step, the
harness recomputes and revalidates the selected strategy and target, stopping on
completion, invalidation, episode termination or truncation, or the commitment cap. The
result is a bounded, revalidated action chunk rather than blind replay of a fixed route.

Empirically, each model-selected option controlled **1.49 primitive environment steps**
on average, compared with one step per decision under primitive-action control. This
shows that the deterministic harness can reduce model-call frequency by expanding one
high-level choice into a bounded, revalidated action chunk. The realized saving was
modest because most `COLLECT` options targeted a nearby pellet and ended after one step.

This division of labor is intentional. The option harness owns deterministic legality
checks, route construction, and safety validation; the learned policy decides which
advertised harness-generated high-level option is most useful in the current visual
context. It reduces the number of model decisions without turning the policy into a
hand-coded Pacman solver.

### 4.3 Constraint alignment across rollout and training

At decision `t`, let `A_t = A(s_t) ⊆ V` be the recorded state-dependent
admissible-action set. Constrained decoding renormalizes the rollout policy over `A_t`
at temperature `T`. [AReaL's decoupled PPO objective](https://arxiv.org/abs/2505.24298)
distinguishes the vLLM rollout behavior policy `π_behav`, the pre-update proximal policy
`π_prox`, and the current actor policy `π_θ` before applying the
[clipped PPO update](https://arxiv.org/abs/1707.06347). The behavior-correction ratio
directly tests whether rollout and training score the recorded choice consistently:

```math
w_t = \frac{\pi_{\mathrm{prox}}(a_t \mid s_t,A_t;T)}
           {\pi_{\mathrm{behav}}(a_t \mid s_t,A_t;T)}.
```

When the behavior and proximal parameters match, consistent normalization gives
`w_t = 1`. If rollout and training instead use different admissible-action sets or
temperatures, the ratio can deviate from one even when the model parameters match. This
is a normalization error, not policy staleness: it creates spurious importance weights,
can reject or clip valid samples, and sends gradient into inadmissible logits.

[Huang and Ontañón](https://arxiv.org/abs/2006.14171) study the analogous case of
sampling from a masked distribution while computing policy-gradient updates from the
unmasked distribution, which they call **naive invalid action masking**. In their PPO
experiments, this inconsistency produced substantially higher KL divergence and more
variable convergence than consistent masking.

At rollout time, the workflow maps the identifiers in `A_t` to exact tokenizer IDs. A
small AReaL request adapter forwards them to vLLM's built-in `allowed_token_ids`, which
masks other tokens before sampling. The rollout stores the sampled token, temperature,
constrained behavior log-probability `log π_behav`, and exact `A_t` together so they
remain aligned through distributed batching.

Before an update, the FSDP actor recomputes `log π_prox`; during the PPO forward pass,
it evaluates `log π_θ`. Neither operation decodes. Both renormalize the actor's logits
over the recorded `A_t`. When KL regularization is enabled, the reference policy `π_ref`
uses the same set and temperature, although it is not a denominator of the PPO ratio.
These distributions need not be equal because their parameters or versions may differ;
their admissible-action set and normalization rule must match.

The key invariant is stronger than “apply an action mask”:

> Rollout behavior, proximal-actor, current-actor, and reference-policy
> log-probabilities—when enabled—must use the same recorded admissible-action set and
> temperature.

### 4.4 Reward design: learn from events, not raw score

The Stage I primitive-action configurations derived rewards from structured game events
emitted by the environment rather than the game's raw score. In the 256-step
configuration, a valid executed action at step $t$ received:

```math
\begin{aligned}
r_t ={}& n_t^{\text{pellet}}
+ n_t^{\text{power}}
+ 50\,\mathbb{1}[\text{level cleared}] \\
&- 0.05
- 0.5\,\mathbb{1}[\text{wall collision}]
+ r_t^{\text{progress}}.
\end{aligned}
```

Fruit remains observable but carries no direct training reward. To provide denser
feedback, we add a progress term based on the change in shortest-path distance to the
nearest normal pellet:

```math
r_t^{\text{progress}}
= 0.1\,c_t\,(d_{t}^{\text{before}}-d_{t}^{\text{after}}),
\qquad
c_t = 1-\rho_t,
\qquad
\rho_t=\frac{m_t}{m_0}.
```

Here, $m_t$ is the number of normal pellets remaining after step $t$, and $m_0$ is the
initial count. The progress term is omitted when the same step already collects a normal
pellet, avoiding double payment for one event. Scaling by $c_t$ makes the guidance
strongest late in the game, when only a few pellets remain. An invalid or unparsable
action response instead received `-50` for that model decision and ended the episode.

The Stage I-B primitive-action continuation kept the same event-level reward family and
fixed step cost, but used raw per-decision feedback without reward or advantage
normalization. Ghost, death, and safety-refusal terms belong to ghost-enabled training
configurations; they should not be read into the ghost-free primitive-action results.

This shaped reward is a training signal, not the success criterion. Strict level
completion and held-out success remain the primary evaluation metrics.

### 4.5 From environment rewards to the policy objective

The reward function defines what feedback the environment produces; the objective
contract defines how that feedback is assigned, normalized, and reduced into an update.
These are separate design choices with three independent dimensions.

**Feedback timescale.** A decision can receive the reward from its immediate environment
step, or, for a harness-generated high-level option, the sum over that option's
committed execution span:

```math
R^{\text{option}}_{e,j}
= \sum_{t\in\text{committed span}(e,j)} r_{e,t}.
```

Both choices provide temporally local feedback and do not propagate consequences beyond
the scored step or option. An option-span return is therefore a local score, not a
counterfactual or state-conditioned advantage. At the episode timescale, all shaped step
rewards are instead summed over one rollout episode:

```math
R_e=\sum_t r_{e,t}.
```

The resulting episode signal preserves the whole-episode objective, but assigns the same
coarse task signal to every model decision in that episode.

**Normalization.** A local or episode-level signal may be used raw or normalized. In the
episode-level group-relative contract, returns sampled from the same initial prompt are
normalized with the group sample standard deviation:

```math
G_e=\frac{R_e-\mu_{\text{prompt}}}
{\sigma_{\text{prompt}}+10^{-5}}.
```

Normalizing a local signal changes its scale or comparison group, not its feedback
timescale. Likewise, episode aggregation retains the sum of intermediate rewards but
discards where within the episode they occurred.

**Episode weighting.** Loss is averaged within each rollout episode before episodes are
averaged:

```math
L=\frac{1}{|E|}\sum_{e\in E}
\left(\frac{1}{N_e}\sum_{t\in e}L_{e,t}\right).
```

Here, $N_e$ is the number of valid trained action tokens in rollout episode $e$. This
reduction prevents length alone from increasing an episode's intended weight: a
200-decision rollout's token losses and a 50-decision rollout's token losses are each
averaged within their own episode. The data pipeline preserves episode boundaries and
encoded admissible-action sets while balancing token load across workers.

### 4.6 What AReaL provides, and what Pacman required

[AReaL](https://github.com/inclusionAI/AReaL) provides the distributed RL substrate:
asynchronous agent rollouts, PPO/GRPO optimization, inference and training workers,
checkpointing, generic multi-turn agent workflows, and multimodal rollout and training
support. Its reference examples expose the last two capabilities mostly separately: the
multi-turn example is a text-only GSM8K retry loop, whereas `VisionRLVRWorkflow`
performs a single image-conditioned generation per episode.

MaaPacman combines these capabilities in an interactive visual-control workflow. At each
fresh model decision, the environment renders the current game state and packages
exactly one fresh PNG with the prompt, while the native workflow preserves the
corresponding VLM processor outputs through rollout and training. Under the
harness-mediated option policy, one selected option may execute several primitive
environment steps before the next model call. The precise contract is therefore one
fresh image per model turn, not one image per environment step.

The integration also carries each state's admissible-action set from rollout into actor
and reference log-probability computation, keeping the constrained policy consistent
across generation and training. It preserves environment-episode identity through
distributed batching, enabling equal-episode loss weighting even when trajectories have
different lengths, and adds the operational support needed for long episodes,
variable-duration options, VLM memory pressure, checkpoint retention, and reproducible
experiment records.

Together, these changes make our AReaL-based stack capable of training a Pacman agent
with state-dependent admissible actions. The current interface remains Pacman-specific
rather than a generic constrained-agent API.

## 5. Experimental Results

### Experimental overview

We followed a staged curriculum rather than train the full ghost-aware task from
scratch. Stage I learned ghost-free primitive-action navigation in two related phases:
Stage I-A bootstrapped the policy with a 256-step cap and selected Iter16, while Stage
I-B continued from that checkpoint with a 512-step cap and evaluated maze
generalization. Stage II transferred the selected Stage I policy to a ghost-enabled,
harness-mediated option policy and validated the whole-episode training path.

#### Stage setup

| Phase                   | Environment   | Action interface        | Initialization                | Rollout cap |
| ----------------------- | ------------- | ----------------------- | ----------------------------- | ----------: |
| **Stage I-A**           | Ghost-free    | One primitive direction | Base Qwen3.5-9B               |         256 |
| **Stage I-B**           | Ghost-free    | One primitive direction | Continue from Iter16          |         512 |
| **Stage II probe**      | Ghost-enabled | Harness-mediated option | Base Qwen3.5-9B and Iter25    |         512 |
| **Stage II validation** | Ghost-enabled | Harness-mediated option | Iter25 actor and KL reference |         512 |

We keep the same decoding settings throughout: `temperature=0.7` for training rollouts
and greedy decoding for evaluation.

#### Objective and evidence

| Phase                   | Reward-to-objective contract                                   | Main evidence                      | Evidence scope              |
| ----------------------- | -------------------------------------------------------------- | ---------------------------------- | --------------------------- |
| **Stage I-A**           | Temporally local per-decision feedback (no episode-level GRPO) | Iter16 selected over Iter49        | Checkpoint selection        |
| **Stage I-B**           | Temporally local per-decision feedback (no episode-level GRPO) | Iter25/31; 44/50 held-out mazes    | Ghost-free maze transfer    |
| **Stage II probe**      | No training                                                    | Iter25 `8/12` vs. base `2/12`      | Preliminary transfer signal |
| **Stage II validation** | Episode-level group-relative objective (GRPO)                  | Three updates completed end to end | Training-path validation    |

Both Stage I phases used temporally local per-decision feedback rather than the
whole-episode group-relative objective used in the bounded Stage II validation. Here,
“local” describes the **credit-assignment timescale**, not the observation: the model
still receives the current maze image at every decision.

For reproducibility, Stage I-A normalized the local decision rewards, whereas Stage I-B
used them raw. This changes the scaling, not the feedback timescale; neither phase used
episode-level GRPO.

### 5.1 Stage I: Ghost-free primitive-action navigation

Both Stage I phases ask the model to perceive the maze from the current screenshot and
choose one admissible primitive direction at each decision. Stage I-B continues from the
checkpoint selected in Stage I-A and extends the rollout horizon. We report them as
related phases, not as one causal learning curve.

#### Stage I-A: 256-step bootstrapping and checkpoint selection

This phase used Qwen3.5-9B, a 256-step rollout cap, and single-token primitive-direction
actions constrained to the directions open in the current state. It used the temporally
local feedback described above and did not use the whole-episode, equal-episode contract
in Section 4.5.

Each update collected 48 rollout episodes from four prompt rows with 12 episodes per
prompt, sampled at temperature 0.7 on one 8-GPU node. Every rollout episode collected
before a completed update reached the 256-step cap, so terminal evaluation was necessary
to distinguish a useful checkpoint from one with merely high shaped reward. We completed
49 optimizer updates. Terminal evaluation selected Iter16 rather than the latest
checkpoint.

| Checkpoint      | Mean shaped reward in its rollout batch | Greedy wins on seeds 0, 1, 2 | Mean normal-pellet clear rate |
| --------------- | --------------------------------------: | ---------------------------: | ----------------------------: |
| Base Qwen3.5-9B |                                       — |                          0/3 |                          1.6% |
| Iter16          |                               **150.7** |                      **3/3** |                    **100.0%** |
| Iter49          |                                   120.3 |                          0/3 |                         92.2% |

These three deterministic replays on one maze are a checkpoint-selection diagnostic, not
a success-rate estimate. Iter49 retained 92.2% mean pellet clearance but recorded 0/3
terminal wins. The later checkpoint therefore preserved much of the local shaped proxy
without preserving terminal reliability; neither training reward nor “percent cleared”
should replace checkpoint evaluation on the actual terminal objective.

The 256-step cap applied only when collecting rollouts for training: each training
episode was truncated after 256 environment steps, limiting exposure to late-game
states. Evaluation did not use this 256-step cap, so the policy could continue acting
past step 256 until the episode ended. For example, one successful Iter16 evaluation
cleared the maze after 666 environment steps. The evidence therefore supports limited
late-state coverage during training, not a general inability to act beyond 256 steps.

The base-model failure is shown in Section 3.1. The following seed-matched checkpoint
replays make the Iter16-versus-Iter49 selection gap directly visible.

<table>
  <tr>
    <th width="50%">Iter16 · selected checkpoint</th>
    <th width="50%">Iter49 · later checkpoint</th>
  </tr>
  <tr>
    <td valign="top">
      <video src="https://github.com/user-attachments/assets/15a7c273-3133-4650-9cb1-4be81323fb00" controls preload="metadata" playsinline width="100%" poster="../assets/demos/pacman-iter16-seed0-poster.png"></video>
    </td>
    <td valign="top">
      <video src="https://github.com/user-attachments/assets/1160f967-c7bb-4426-b5e2-1834b3138018" controls preload="metadata" playsinline width="100%" poster="../assets/demos/pacman-iter49-seed0-poster.png"></video>
    </td>
  </tr>
  <tr>
    <td align="center"><em>Seed 0: clears the maze.</em></td>
    <td align="center"><em>Seed 0: clears most pellets but terminates STUCK.</em></td>
  </tr>
  <tr>
    <td align="center"><a href="../assets/demos/pacman-iter16-seed0.mp4">Download Iter16 MP4</a></td>
    <td align="center"><a href="../assets/demos/pacman-iter49-seed0.mp4">Download Iter49 MP4</a></td>
  </tr>
</table>

*On the original ghost-free benchmark maze, Iter16 clears the level while Iter49 gets
stuck. The contrast shows why proxy progress cannot replace terminal checkpoint
selection.*

#### Stage I-B: 512-step continuation and learning dynamics

This phase continued from the selected Iter16 checkpoint, raised the rollout cap to 512
steps, and used **raw per-decision feedback, with neither reward nor advantage
normalization**. It produced Iter25 and Iter31.

![Stage I learning dynamics across the selected 256-step and 512-step lineage](../assets/figures/maapacman_stage1_learning_dynamics.png)

*Figure 2. The selected lineage follows Stage I-A through Iter16 and then its Stage I-B
continuation; Stage I-A continued separately to Iter49, which is not fully plotted. Each
point aggregates 48 rollout episodes. Because rollout caps and reward scaling differ,
shaped-reward levels should be compared within rather than across phases. Clear rate
remains comparable across caps; shorter mean episode length matters only after wins
appear.*

#### Held-out-maze generalization

We evaluated Iter25, Iter31, and the base model on a shared set of 50 held-out maze
layouts with deterministic greedy decoding and a 2,000-step limit. Iter16 itself was not
run on this suite, so this is a separate generalization study rather than a single
checkpoint curve spanning both Stage I phases.

![Generalization across 50 held-out mazes](../assets/figures/maapacman_real50_generalization.png)

*Figure 3. Iter25 produced a large paired improvement over the base model. Continuing to
Iter31 did not improve the strict pass count.*

| Model           | Strict passes |   Wilson 95% CI | Mean normal pellets remaining |  Easy / medium / hard |
| --------------- | ------------: | --------------: | ----------------------------: | --------------------: |
| Base Qwen3.5-9B |          0/50 |       0.0%–7.1% |                        237.20 |    0/17 · 0/17 · 0/16 |
| Iter25          |     **44/50** | **76.2%–94.4%** |                          0.30 | 17/17 · 17/17 · 10/16 |
| Iter31          |     **44/50** | **76.2%–94.4%** |                      **0.24** | 17/17 · 17/17 · 10/16 |

Base→Iter25 is a paired gain of 88 percentage points (exact two-sided McNemar
`p=1.14e-13`). Iter25 and Iter31 have no discordant outcomes (McNemar `p=1.0`); both
fail the same six hardest maze variants.

The scope matters: this was a **geometry/navigation evaluation in safe mode, with ghosts
and fruit disabled**. It is strong evidence that the learned policy transfers across
maze layouts. It is not evidence of general ghost-aware Pacman play.

Three replay-audited examples make the Iter31 navigation behavior directly inspectable.
They use the same frozen checkpoint, seed 0, greedy decoding, open-action masking, and a
2,000-step limit.

<table>
  <tr>
    <th width="33%">Original maze</th>
    <th width="33%">Held-out maze A</th>
    <th width="33%">Held-out maze B</th>
  </tr>
  <tr>
    <td valign="top">
      <video src="https://github.com/user-attachments/assets/912ae066-187a-4e7f-ae40-e42a0d396255" controls preload="metadata" playsinline width="100%" poster="../assets/demos/pacman-iter31-level1-original-poster.png"></video>
    </td>
    <td valign="top">
      <video src="https://github.com/user-attachments/assets/f17cbe9e-599f-400a-a533-5945dba0fc61" controls preload="metadata" playsinline width="100%" poster="../assets/demos/pacman-iter31-level5-held-out-poster.png"></video>
    </td>
    <td valign="top">
      <video src="https://github.com/user-attachments/assets/9c2c173b-05c9-4b93-984c-fdea1c008f2c" controls preload="metadata" playsinline width="100%" poster="../assets/demos/pacman-iter31-held-out-maze-poster.png"></video>
    </td>
  </tr>
  <tr>
    <td align="center">413 steps · reward 1,907<br>score 2,320 · <strong>strict pass</strong></td>
    <td align="center">446 steps · reward 1,924<br>score 2,370 · <strong>strict pass</strong></td>
    <td align="center">841 steps · reward 2,799<br>score 3,640 · <strong>strict pass</strong></td>
  </tr>
  <tr>
    <td align="center"><a href="../assets/demos/pacman-iter31-level1-original.mp4">Download MP4</a></td>
    <td align="center"><a href="../assets/demos/pacman-iter31-level5-held-out.mp4">Download MP4</a></td>
    <td align="center"><a href="../assets/demos/pacman-iter31-held-out-maze.mp4">Download MP4</a></td>
  </tr>
</table>

*All three replays clear every normal pellet. As in the quantitative study, ghosts and
fruit are disabled, so these videos demonstrate geometry and pellet-collection behavior
rather than ghost-aware play.*

### 5.2 Stage II: Ghost-enabled, harness-mediated option policy

Stage II changes both the environment and the decision interface. Ghosts are enabled,
and instead of choosing every primitive direction, the model selects among
harness-generated high-level options for collecting pellets, avoiding danger, or
pursuing an edible ghost.

#### Rollout-only transfer probe

We first loaded Iter25 into this policy without performing another PPO update. In one
matched probe at seed 2 and temperature 0.7, each model ran 12 episodes with a 512-step
cap. Iter25 completed the normal-pellet objective in `8/12` episodes, compared with
`2/12` for the original Qwen3.5-9B model. Both produced `12/12` unique option paths in
this sample.

#### Bounded training-path validation

We then initialized both the actor and frozen KL reference from the immutable Iter25
checkpoint and created a fresh optimizer for a bounded three-update validation run. Each
update used four initial prompts and 12 rollout episodes per prompt, each capped at 512
steps, for 48 episodes per update. Full collected rollout-episode returns were
normalized only within each prompt group, the resulting episode signal was assigned to
that episode's option decisions, and each episode received equal intended loss weight.
All three optimizer updates completed and wrote regular model checkpoints.

#### Exploratory matched post-run diagnostic

We subsequently began a separate in-sample evaluation on the original Stage II training
prompts. The planned evaluation was stopped before all four policy versions reached 144
episodes, so we froze the available file lists and retained only the exact 60 episode
IDs present for the pre-update control and all three updated checkpoints.

| Policy version     | Normal-pellet completions | Change vs. control | Mean normal pellets remaining | Mean score | Steps per model turn | Exact McNemar `p` vs. control |
| ------------------ | ------------------------: | -----------------: | ----------------------------: | ---------: | -------------------: | ----------------------------: |
| Pre-update control |             21/60 (35.0%) |                  — |                          73.1 |      9,231 |                 1.47 |                             — |
| Update 1           |             22/60 (36.7%) |            +1.7 pp |                          63.6 |     10,046 |                 1.51 |                         1.000 |
| Update 2           |             26/60 (43.3%) |            +8.3 pp |                          46.4 |     11,169 |                 1.50 |                         0.359 |
| Update 3           |             23/60 (38.3%) |            +3.3 pp |                          49.1 |     11,114 |                 1.50 |                         0.839 |

Update 2 had the highest observed completion rate, but its paired difference from the
control was not statistically significant. The results are an exploratory snapshot on
training prompts, not a completed evaluation or a held-out generalization claim.

#### Matched qualitative replay

The following replay-audited videos use the same training-prompt specification and
sampling slot (seed 12, sample 00). The pre-update Iter25 control terminated with a
safety refusal after 54 environment steps, with 152 normal pellets remaining. The Update
2 checkpoint cleared all normal pellets in 352 steps.

<table>
  <tr>
    <th width="50%">Pre-update Iter25 control</th>
    <th width="50%">After optimizer update 2</th>
  </tr>
  <tr>
    <td valign="top">
      <video src="https://github.com/user-attachments/assets/0d061873-7fea-4fdf-850d-69ca9d795c8d" controls preload="metadata" playsinline width="100%" poster="../assets/demos/pacman-stage2-control-seed12-sample00-poster.png"></video>
    </td>
    <td valign="top">
      <video src="https://github.com/user-attachments/assets/81ef8afc-2a14-407c-92e4-b1e0f44743e2" controls preload="metadata" playsinline width="100%" poster="../assets/demos/pacman-stage2-update2-seed12-sample00-poster.png"></video>
    </td>
  </tr>
  <tr>
    <td align="center">54 environment steps · safety refusal<br>152 normal pellets remaining</td>
    <td align="center">352 environment steps<br><strong>All normal pellets cleared</strong></td>
  </tr>
  <tr>
    <td align="center"><a href="../assets/demos/pacman-stage2-control-seed12-sample00.mp4">Download control MP4</a></td>
    <td align="center"><a href="../assets/demos/pacman-stage2-update2-seed12-sample00.mp4">Download Update 2 MP4</a></td>
  </tr>
</table>

These videos visualize one matched in-sample example. They do not by themselves estimate
win-rate improvement, demonstrate held-out transfer, or establish reliable ghost
avoidance.

#### What this evidence establishes

The rollout probe shows that the Stage I checkpoint can operate with the ghost-enabled,
harness-mediated option policy without collapsing to one repeated option sequence. The
bounded three-update validation run shows that the whole-episode constrained-policy
training path executes end to end. Update 2 had a higher observed completion rate in the
exploratory paired snapshot, but its paired difference from the control was not
statistically significant. The matched videos make one behavioral contrast inspectable.
This is still not a long Stage II learning curve, a multi-seed improvement estimate, or
evidence of general ghost avoidance. In particular, normal-pellet completion should not
be silently equated with every possible full-game objective.

## 6. Insights

### 6.1 More updates do not guarantee a better checkpoint

Iter16 beat Iter49 on the terminal evaluation even though Iter49 still had a high shaped
reward and 92.2% mean pellet clearance. Long agent runs therefore need immutable
periodic checkpoints and a fixed selection suite. “Latest” and “best” are different
concepts. The result is consistent with later updates preserving a local shaped proxy
without preserving terminal reliability. Limited late-episode credit and state coverage
are plausible contributors, but this experiment does not isolate either mechanism or
establish convergence to a local optimum.

### 6.2 Reward processing defines the objective, not just its scale

How rewards are aggregated, normalized, and reduced determines the learning objective. A
normalized local reward remains local feedback; an episode return preserves the
whole-game objective but provides coarser temporal credit. Episode averaging separately
prevents longer trajectories from receiving more intended weight merely because they
contain more trained tokens.

The reported runs do not isolate these choices. Stage I-A and Stage I-B differ in
initialization, rollout horizon, and other settings, while Stage II also changes the
environment and action interface. The results therefore do not establish that raw or
normalized feedback—or local or episode-level feedback—is categorically better. That
comparison requires a matched ablation.

### 6.3 Terminal outcomes expose proxy-metric failures

Pellet-clear percentage, shaped reward, game score, and action diversity are useful
diagnostics, but none is the task itself. Iter49 is the clearest example: 92.2% average
clearance and 0/3 terminal wins. Terminal success and exact failure reasons must lead
the evaluation table.

### 6.4 Generalization claims must match the environment contract

The 44/50 result demonstrates transfer across maze geometry under safe-mode navigation.
Enabling ghosts changes the transition distribution, reward events, and failure modes;
under the harness-mediated option policy, it also changes the advertised options.
“Generalizes across mazes” and “general Pacman agent” are not interchangeable claims.

### 6.5 Traceability is part of reliable training

Long-horizon failures can arise from misaligned state-action metadata or incomplete
rollouts even when the visible symptom appears elsewhere. Reproducible experiment
records and explicit checkpoint-completion status are therefore necessary to interpret a
curve.

## 7. Limitations and Future Work

The current evidence supports a useful but bounded conclusion. These are limits of the
present implementation or evidence, rather than part of the task definition:

1. **Observation interface:** the harness-mediated option policy receives both the
   screenshot and authoritative metadata for the currently advertised options. It
   therefore evaluates multimodal high-level option selection rather than pixels-only
   control. Future work should test temporal visual input such as two-frame or
   map-plus-local observations.
1. **Ghost-aware generalization:** separate prototype runs show that the
   harness-mediated option policy can complete ghost-enabled games, but the reported
   50-maze suite disables ghosts and fruit to isolate geometry. Repeat that suite with
   controlled ghost dynamics and report deaths, edible-ghost decisions, and safety
   refusals.
1. **Handcrafted option harness:** the predefined strategy families, target-selection
   rules, route construction, and safety gates are currently handcrafted. A longer-term
   direction is to co-evolve the policy and option-generation mechanism while retaining
   deterministic legality and safety validation.
1. **Evaluation coverage and hard-maze diagnosis:** repeat training with multiple random
   seeds, evaluate Iter16, Iter25, and Iter31 on the same 50-maze suite, and investigate
   the six hardest mazes failed by both Iter25 and Iter31 before extending training.
1. **Controlled reward and credit study:** under matched initialization and budget,
   compare raw per-decision feedback, whole-episode normalization, raw option-span
   feedback, and an episode objective with a small bounded local auxiliary. The
   auxiliary remains an untrained proposal; begin with a fixed coefficient before
   considering adaptive gradient balancing.

## 8. Conclusion

Pacman gave us a compact way to expose the real difficulties of visual-agent RL:
state-dependent admissible-action sets, long episodes, delayed credit, distributed
constraint alignment, and expensive feedback. The deterministic MaaPacman environment
made scalable, reproducible training possible, while AReaL supplied the asynchronous
rollout and distributed RL foundation.

The main empirical result is that the trained policy transferred from the base maze
distribution to 44 of 50 held-out layouts under the stated safe-mode contract, compared
with 0 of 50 for the base model. The equally important systems result is that the latest
checkpoint was not the best one, and that reward-normalization choices change the
learning objective rather than merely the scale of a metric.

That combination—measurable improvement plus explicit boundaries on what the experiment
proves—is the standard we want for the next, harder game-agent experiments.
