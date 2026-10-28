set shell := ["bash", "-euo", "pipefail", "-c"]

# list recipes
default:
    @just --list

# create the environment
setup:
    uv sync

# generate a recording of fresh worlds
world *args:
    uv run python -m shared_cache_world_model.world {{args}}

# list the revisits in a recording
revisits *args:
    uv run python -m shared_cache_world_model.revisits {{args}}

# measure the probe's reference points
baselines *args:
    uv run python -m shared_cache_world_model.baselines {{args}}

# train one model
train *args:
    uv run python -m shared_cache_world_model.train {{args}}

# check every arm can retrieve a room
task-overfit *args:
    uv run python -m shared_cache_world_model.task_overfit {{args}}

# score revisit accuracy against gap
probe *args:
    uv run python -m shared_cache_world_model.probe {{args}}

# draw the figure from results/
figure *args:
    uv run python -m shared_cache_world_model.figure {{args}}

# upload a recording to the bucket
upload recording:
    aws s3 sync {{recording}} s3://hayder-shared-cache-world-model/{{recording}} --exclude "*.gif"

# train on one GPU on AWS
sky-train train_args *args:
    sky launch -c shared-cache sky/train.yaml --env TRAIN_ARGS={{quote(train_args)}} {{args}}

# run the same-context seed batch
sky-batch *args:
    sky launch -c shared-cache-batch sky/batch.yaml {{args}}

# run the six ced runs
sky-batch-ced *args:
    sky launch -c shared-cache-batch-ced sky/batch_ced.yaml {{args}}

# run the three ced_full runs
sky-batch-ced-full *args:
    sky launch -c shared-cache-batch-ced-full sky/batch_ced_full.yaml {{args}}

# queue a run on the running box
sky-exec train_args *args:
    sky exec shared-cache sky/train.yaml --env TRAIN_ARGS={{quote(train_args)}} {{args}}

# copy results back from the bucket
sky-fetch:
    aws s3 sync s3://hayder-shared-cache-world-model/results results --exclude "*.pt" --exclude "per_layer_h128_s42_20k/probe_*.json" --exclude "tiles_sharedkv_v1_per_layer_h128_s42/*"

# strip checkpoints to weights
export-weights *args:
    uv run python -m shared_cache_world_model.export_weights {{args}}

# download weights from Hugging Face
fetch-weights:
    hf download jadidbourbaki/context-compaction-world-models --local-dir weights

# run a model in closed loop
agent *args:
    uv run python -m shared_cache_world_model.agent {{args}}

# run the closed-loop agent on every same-context checkpoint
agent-batch *args:
    uv run python -m shared_cache_world_model.agent_batch {{args}}

# measure cache bytes, decode latency, and the cost of compaction for every design
systems *args:
    uv run python -m shared_cache_world_model.systems {{args}}

# play the game inside a model
play *args:
    uv run python -m shared_cache_world_model.play {{args}}

# publish the play page to its Space
deploy-space:
    hf upload jadidbourbaki/context-compaction-world-models space . --repo-type space
    hf upload jadidbourbaki/context-compaction-world-models shared_cache_world_model shared_cache_world_model --repo-type space --exclude "__pycache__/*"

# release the GPU
sky-down:
    sky down shared-cache -y

# format code in place
fmt:
    uv run ruff format shared_cache_world_model tests
    uv run ruff check --fix shared_cache_world_model tests

# lint read-only
lint:
    uv run ruff format --check shared_cache_world_model tests
    uv run ruff check shared_cache_world_model tests

# type check
typecheck:
    uv run ty check shared_cache_world_model

# run the tests on the CPU
test:
    uv run pytest -q

# read-only gate
check: lint typecheck test
