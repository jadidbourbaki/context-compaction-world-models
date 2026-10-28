# AGENTS.md

Guidance for AI agents working on the context-compaction-world-models repo. The
CLAUDE.md symlink resolves to this file. Read it in full at the start of a
session.

## Project context

context-compaction-world-models holds the code for the paper Context
Compaction for World Models. It trains four world models of a generated tile
game and measures their context retrieval at two layers of the inference
stack. At the model layer, the four designs form a chain in which each step
changes one thing. The `per_layer` design, which the paper calls
Decoder-only, keeps a KV cache in every block. The `ced_full` design, CED,
computes the keys and values of its upper blocks from the lower output,
following the Causal Encoder-Decoder of DeepSeek-V4.1-Flash (2026). The `ced`
design, CED-SWA, switches the lower blocks to a sliding window. The
`shared_cache` design, DD, shares one KV cache across its upper blocks,
following [You Only Cache Once](https://arxiv.org/abs/2405.05254) (2024).
Per token, the four designs cache 16, 16, 8, and 2 vectors. At the agent
harness layer, `compaction.py` compacts a session into a walk through every
visited room, and `agent.py` runs each world model as a world agent with and
without dreaming. The paper reports the context retrieval gain against the
shuffled control, the same comparison at a fixed KV cache budget, the effect
of dreaming, and the agent latency of each policy.

The game is generated. Every episode builds a new eight by eight grid of walled rooms,
each 16 by 16 tiles and joined to its neighbours by doors, scatters
scenery inside them, and navigates a character between them. The character
stands in the centre of a room and one cardinal action moves it directly to
a neighbouring room. Eight tile-type
tokens each carry a 16 by 16 occupancy mask, and one action token
completes each step. The model sees the room with the character drawn over it. The preview
renders each logical tile with a 16 by 16 public-domain sprite.
Revisit accuracy is an exact count of wrong tiles inside the wall border
when a room is entered again.

A generated world lives only in its own episode, so a room of an
evaluation recording is one the model never trained on and its tiles can
reach a prediction only through the history. That is the memory test, and
a generated game is what makes it valid. A fixed map is memorizable, and a
model trained on one answers every question from its weights
([Chan et al.](https://arxiv.org/abs/2205.05055), 2022). An earlier
version of this project used The Legend of Zelda: Link's Awakening on an
emulator and measured exactly that failure.

The code is Python. Models are PyTorch modules around FlexAttention. GPU
runs go through SkyPilot on AWS.

## Repository layout

```
shared_cache_world_model/world.py      generate a world per episode and log room transitions
shared_cache_world_model/revisits.py   find every revisit in a recording, with its gap
shared_cache_world_model/baselines.py  measure the reference points the probe is read against
shared_cache_world_model/data.py       validated recordings and sampled training windows
shared_cache_world_model/compaction.py compact a transcript into a covering walk from its actions alone
shared_cache_world_model/model.py      the four architectures behind one interface
shared_cache_world_model/task_overfit.py prove every arm can retrieve through the main decoder
shared_cache_world_model/train.py      train one model and write results/<run>/summary.json
shared_cache_world_model/probe.py      score revisit accuracy against gap length
shared_cache_world_model/figure.py     draw the figure from results/
shared_cache_world_model/export_weights.py strip checkpoints to the weights a probe needs
shared_cache_world_model/agent.py      play the game with the model remembering revisited rooms
shared_cache_world_model/agent_batch.py run the closed-loop agent on every checkpoint
shared_cache_world_model/systems.py    measure cache bytes, decode latency, and compaction cost
shared_cache_world_model/play.py       play the game inside a model in the browser
tests/                                 CPU tests on generated worlds and tiny random models
space/                                 the Hugging Face Space that serves the play page
sky/train.yaml                         the SkyPilot task for one GPU on AWS
results/<run>/                         one summary.json per run, committed
docs/                                  the data description
```

Recordings stay out of git because of their size. Every one of them is
reproducible from the settings in its `meta.json`.

## Don't reinvent the wheel

Before writing any non-trivial logic, look for something already built.
Search the standard library first, then the dependencies already in
`pyproject.toml`, then PyPI. A mature package is almost always more
correct and better tested than a version written under deadline, and
every line not written is a line nobody has to review, test, or maintain.
Hand-rolling what a library already solves adds a design of our own that
we then own forever.

Judge a candidate dependency by its maintenance and adoption: recent
releases, responsive maintainers, wide use in serious projects, a focused
scope. A good dependency is welcome. Reinvent only when nothing fits or
the dependency would weigh far more than the problem it solves.

In this repo that means: PyTorch's FlexAttention for every attention call, `nn.Linear`,
`nn.LayerNorm`, and `nn.Embedding` around it, and pydantic for every
setting. The attention math itself is never written out. Attention is
causal over steps and unrestricted inside a step. That rule is a function of
the query and key positions, and FlexAttention's block mask evaluates it per
128 by 128 block and skips the blocks with nothing visible. A 128-step
history is 1,152 tokens. PyTorch's transformer
layers require that mask and are not used. Before writing code for a new
direction, search for prior papers, existing datasets, and library support,
and present the choice with sources.

## Quality gate

`just check` runs `ruff format --check`, `ruff check`, `ty check`, and
`pytest`. Work is complete when `just check` passes locally. The tests
run on the CPU against tiny generated worlds and two-layer random models,
so a change to the model or the probe is tested without a GPU.

## Writing prose

The following rules apply to all prose in the repo: README, docs, run
logs, commit messages, code comments, docstrings.

### Hard rules

- No em-dashes and no en-dashes. Split the sentence or use a comma.
- No semicolons in prose. Use a period and start a new sentence.
- No parenthetical asides. A parenthesis may hold an abbreviation on
  first use.
- No contrast constructions such as "X, not Y", "rather than Y", or
  "instead of Y". State what the code or the method does.
- No comma splices. Two independent clauses get two sentences.
- No ASCII diagrams. Describe relationships in prose.
- No emoji.
- No vague back-references. Do not open a sentence with "This",
  "That", "These", "Those", "Their", or "It" pointing at a noun from
  an earlier sentence. Name the noun again.
- No slogans, aphorisms, or stock metaphors. Every sentence names a
  concrete actor and a concrete object.

### Soft rules

- Write short declarative sentences in the active voice. Docs use
  "we": "We record", "We then measure".
- Put the subject first.
- Define a term in the sentence where it first appears.
- Cite papers as markdown links with the title and year, for example
  [Genie](https://arxiv.org/abs/2402.15391) (2024).

## Writing comments

- Docstrings are one line.
- Comment only where the reason is not obvious from the code. Keep
  the comment to one or two short lines in plain words.
- Where a line of code needs an explanation, a short comment sits
  above it. The comment says what the line does and why the code needs
  it here, in plain words, and gives numerical behavior as a numerical
  example: "a history of 128 steps is 1,152 tokens".
- Model and training code explains its PyTorch calls this way at their
  first use in a module. The maintainer reads these files to learn.
- Describe the present behavior and its reason. Do not describe the
  history of the code or pull requests.
- No comment repeats what a well-named identifier already says.

## Code readability

- One assignment per line. No tuple packing such as `a, b = 1, 2`.
- Descriptive names: `tiles_per_frame`, `history_steps`, `gap_steps`.
  A single letter is acceptable only as a comprehension index.
- Name intermediate results. Do not chain calls.
- Type every function signature, parameters and return.
- Tensors document their shape and dtype in a comment at the line that
  creates them.
- Delete code that no current result depends on. When an approach is
  abandoned or reverted, remove its functions, settings, tests, and
  comments in the same change.
- Fail hard. No `try/except: pass`. No silent fallbacks. A malformed
  recording or an unknown setting raises with a message that names the
  offending value.
- Validate inputs at the boundary with pydantic models and pass those
  models across function boundaries.
- Seed everything with 42.
- Pick the device with `torch.accelerator.current_accelerator()`.

## Experiment hygiene

- Every run writes `results/<run>/summary.json` with the scores, the
  full config, the git commit, the library versions, and the device.
  The file is committed. Recordings and checkpoints stay out of git.
- Compare the two architectures at the same depth, width, data, seed
  set, and training tokens. A run with a reduced step count is a smoke
  test and is reported as one.
- Keep the results of negative runs with the same detail as positive ones.
- A change to the recording format or the probe invalidates earlier
  runs. Rerun them.

## Writing tests

Add a test only when it is necessary. A test earns its place by being
able to fail for a real defect. Before
adding one, name the defect it would catch and the result that defect
would corrupt. Before keeping one, ask whether any other test already
fails on the same defect. Delete a test whose only failure mode is
already covered by a stronger test.

### What must be covered

Every claim the paper makes rests on a handful of properties, and each
one carries a test that fails loudly when the property breaks.

- No step sees a later step, and every token of a step sees its whole
  step. A leak here invents a result.
- The probe reads the step before the frame it scores. An earlier
  formulation let the model see the frame it was predicting and produced
  a number that measured nothing.
- Incremental decoding matches a full forward pass, because the cache
  cost we report only means something if the cache reproduces the model.
- The cache accounting matches the architecture, because the figure
  reports it as the price of the shared cache.
- A window never spans two episodes, and two recordings never share a
  room.
- A recording's arrays match the settings and the checksum that name it.

### How to write them

- Test the property, not the transcript of the implementation. An
  assertion that restates a one-line function in different syntax
  catches nothing.
- Where a result is generated, test it with `hypothesis` over many
  drawn inputs. A loop over a fixed seed is a property test that only
  ever tries one case.
- Check a result with a different method than the one that produced it.
  Verify a graph property with `networkx`, not with a second copy of the
  search under test.
- Never assert a cosmetic detail. Pixel offsets, colour values, and
  layout coordinates change without any result changing, so pin the
  behaviour that matters instead.
- Shared setup goes in `tests/conftest.py` as a fixture. Building the
  same world in four files is four places to update.

### Tools

Use what the ecosystem already provides, for the same reason the rest of
the repo does.

- `pytest` with plain `assert`, `@pytest.mark.parametrize` for several
  cases of one shape, `tmp_path` for files, and `pytest.raises` with
  `match=` so a test pins the message as well as the type.
- `hypothesis` for generated inputs and their invariants.
- `networkx` for graph and connectivity checks.
- `pytest.approx` for floats and `torch.testing.assert_close` for
  tensors. Never compare floats with `==`.
- Hand-written fakes for a model or a component whose real version would
  make the test slow or vague. A fake with one behaviour, such as a model
  that repeats its input, turns a fuzzy claim into an exact number.

The suite runs on the CPU on tiny worlds and two-layer models, so the
whole of it finishes in seconds and runs on every change.

## Commit messages

Conventional commits, one sentence each, no body.

```
feat: add the shared cache model
fix: score revisits on the background layer only
docs: log the first seed batch
chore: pin numpy 2.5.3
```

- Lowercase the type and the first word after the colon, unless it is
  a proper noun or acronym.
- No `Co-Authored-By` trailer and no "Generated with" line.
- Do not amend or rewrite published commits without explicit user
  consent. Force-push only with `--force-with-lease`, only on a
  feature branch, and only after confirming.

## Git practices

- Use the git identity the user has configured. Never pass
  `-c user.email` or `-c user.name`.
- Ask before every commit and every push, each time, even after a
  prior yes.
- Confirm with the user before any destructive operation.

## Python style

- uv for environments and dependencies: `uv add`, `uv lock`,
  `uv run`.
- ruff for linting and formatting.
- ty for type checking.
- just for task running.
- Runtime dependencies go in `[project].dependencies`. Dev tools go
  in `[dependency-groups].dev`.
- Pin every direct dependency to an exact version with `==` and commit
  `uv.lock`. The lockfile is never hand-edited.
- Put `from __future__ import annotations` at the top of every module.
- Use built-in generics such as `list[int]` and `dict[str, T]`, and
  `X | None`.
- Every import goes at the top of the module. An import inside a
  function needs a comment naming the circular import it avoids.
- `snake_case` for functions and variables, `PascalCase` for classes,
  `UPPER_SNAKE` for module constants. A leading underscore marks a
  name module-private.
- Signal failure by raising an exception. Catch narrowly. A bare
  `except:` is never correct. Use `raise ... from err` to preserve the
  cause.

## SkyPilot and AWS

- A launch spends money. Ask the user before every launch. The approved
  ceiling for this project is 300 dollars.
- Development runs and long pilots use one `p5.4xlarge` H100 when capacity
  permits. The first fallback is one RTX PRO 6000 Blackwell on a
  `g7e.2xlarge`. The final fallback is one L40S on a `g6e.xlarge`. Seed
  batches go on one `p5.48xlarge`. The task autostops after 30 idle minutes
  and tears the machine down.
- A run's name must not collide with an earlier run in the bucket, because
  a training script resumes from any checkpoint it finds under its name.
- Checkpoints and recordings live in S3, and a training script resumes
  from the latest checkpoint on start. Fetch `results/` before a
  machine goes down.
- No credential enters the repo.

## Working with the user

- Local, reversible actions such as editing a file or running a test
  need no preamble. Actions that spend money or are hard to reverse
  need explicit confirmation each time. Authorization for one action
  does not extend to similar actions.
- Lay out a plan before destructive multi-step work. Get a green
  light, then execute. After every destructive step, summarize the
  state.
- When you spot a side effect the user did not ask for, name it and
  ask before doing it.
- Default to terse. Lead with the result.
- End a turn with one or two sentences: what shipped and what is next.
- Do not restate the request.
- When a change introduces a model architecture or a training
  objective, walk the user through the code at the level of the
  tensors before it lands.
- Match the scope of a change to the request. A bug fix does not
  include a refactor.

## When in doubt

Re-read this document, then the most recent changes that touched the
same area. Match the existing patterns.
