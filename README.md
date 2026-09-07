# LeLab GoalBlocks

An experimental, goal-conditioned SO-101 data collection and learning extension built on top of [Hugging Face LeLab](https://github.com/huggingface/LeLab) and [Hugging Face LeRobot](https://github.com/huggingface/lerobot).

This repository extends the LeLab recording interface with goal-image upload, optional Qwen-VL task-description generation, and persistent goal metadata. It also includes a compact GoalBlocks policy that consumes a goal image, natural-language task, two camera observations, and robot state to predict an SO-101 action chunk.

> **Unofficial derivative work:** this is an independent research prototype. It is not an official Hugging Face or LeRobot release and is not endorsed by Hugging Face.

## What Was Added

### Goal-aware LeLab recording

![Goal-aware recording configuration](docs/images/goal-recording-configuration.png)

The recording dialog accepts goal images and can generate an editable task description before a demonstration begins.

- Upload one to four goal images while configuring a recording.
- Generate an editable task description from the goal image with Qwen-VL through DashScope.
- Keep the API key in memory for one request only; it is not written to project files or dataset metadata.
- Store goal images under `goals/` in the resulting LeRobot dataset.
- Persist the goal-image paths, task text, source, and language in `meta/info.json`, including partially completed recordings.

### GoalBlocks baseline

`goalblocks_baseline/` provides:

- balanced loading from multiple independent LeRobot dataset repositories;
- top camera, wrist camera, robot state, task text, and goal-image inputs;
- a lightweight goal-conditioned action-chunk model;
- episode-level train/validation splits, AMP, warmup plus cosine scheduling, gradient clipping, checkpointing, JSONL metrics, and resume support;
- a small remote inference server and a safety-oriented WSL robot client.

The baseline has **738,432 parameters (0.74M)**. A comparable two-camera, six-state/six-action LeRobot ACT configuration has approximately **51.55M parameters**, about **69.8 times more**. This baseline is intended to validate the complete pipeline, not to replace ACT or represent the final planned architecture.

### Released Checkpoint

The early experimental checkpoint, training configuration, metrics, dataset index, and goal-aware recording screenshot are available on Hugging Face:

[bmnzyb/goalblocks-so101-baseline](https://huggingface.co/bmnzyb/goalblocks-so101-baseline)

This checkpoint was trained only to validate the end-to-end pipeline and must not be treated as a reliable or production-ready robot policy.

## Datasets

The initial experiment uses two separately hosted LeRobot datasets:

| Dataset | Episodes | Modalities | Notes |
| --- | ---: | --- | --- |
| [`bmnzyb/GoalBloakcs_1_20260906_155729`](https://huggingface.co/datasets/bmnzyb/GoalBloakcs_1_20260906_155729) | 3 | top RGB, wrist RGB, 6D state/action, task, goal image | The top camera contains corrupted/torn frames. |
| [`bmnzyb/GoalBloakcs_1_20260906_162409`](https://huggingface.co/datasets/bmnzyb/GoalBloakcs_1_20260906_162409) | 2 | top RGB, wrist RGB, 6D state/action, task, goal image | Very small validation dataset. |

The same information is available in [`datasets.json`](datasets.json). These five episodes are enough to test the software path, but not enough to claim reliable task performance.

We welcome collaborators who would like to collect more goal-conditioned SO-101 demonstrations: different block arrangements, camera placements, lighting conditions, objects, successes, and recoveries are all valuable. Please open an issue or pull request with a dataset link and a short description of its hardware and collection setup.

## Quick Start

Install LeLab from this checkout using its normal development instructions. The editable install ensures the `lelab` command loads this source tree:

```bash
uv tool install --editable .
lelab --no-open
```

Open `http://localhost:8000`, configure the robot and cameras, then use the Goal Images section in the recording dialog. The generated description remains editable before recording begins.

Formal training in the existing LeRobot environment:

```bash
CUDA_VISIBLE_DEVICES=2 python -m goalblocks_baseline.train \
  --device cuda:0 \
  --steps 10000 \
  --batch-size 16 \
  --output-dir /workspace/outputs/goalblocks_baseline_formal
```

Remote inference instructions, including the default no-action dry-run, are documented in [`goalblocks_baseline/README.md`](goalblocks_baseline/README.md).

## Security and Robot Safety

- Never commit Qwen, Hugging Face, GitHub, or SSH credentials.
- The remote robot client does not send actions unless `--execute` is explicitly supplied.
- A successful network or software test does not establish that a learned policy is safe.
- Clear the workspace, keep an operator near the emergency power control, validate camera and joint ordering, and run the dry-run before every hardware deployment.
- The included model was trained on only five episodes and must be treated as an experimental pipeline artifact, not a production robot controller.

## Attribution, Copyright, and License

This repository contains and modifies code from:

- [Hugging Face LeLab](https://github.com/huggingface/LeLab), the upstream application and the base Git history for this repository.
- [Hugging Face LeRobot](https://github.com/huggingface/lerobot), whose dataset, robot, camera, and policy interfaces are used by the extension; portions of the integration follow LeRobot APIs and conventions.

The upstream projects are distributed under the **Apache License 2.0**. This derivative repository retains the upstream `LICENSE`, copyright headers, and attribution. Modified files and newly added GoalBlocks components are identified by the Git history and summarized in [`NOTICE`](NOTICE).

Apache-2.0 permits use, modification, and redistribution subject to its terms. Attribution does not transfer ownership of upstream code, imply endorsement, or remove any license obligations. Downstream users must preserve applicable copyright, license, and notice text and should review the licenses of dependencies, model providers, and datasets separately.

Unless a file states otherwise, contributions made specifically to this derivative repository are offered under Apache-2.0 to remain compatible with the upstream project.

## Status

This is early research software. Goal-image persistence and the baseline pipeline have automated tests, but the first top camera produced corrupted JPEG frames, and the released training run was intentionally stopped early. See the model card for exact training status and limitations.




# SO-101 GoalBlocks Dataset Project

Last updated: September 2026

## Purpose

GoalBlocks is an open, goal-conditioned data collection project for the SO-101 single-arm robot. Given a target arrangement of colored blocks, the robot should select blocks from a workspace and reproduce the arrangement through teleoperation demonstrations.

Every task has two complementary conditions:

- a concise natural-language instruction;
- one or more goal images, such as a 45-degree view or top/front/side views.

A trajectory contains the SO-101 six-dimensional joint state, top and wrist camera observations, six-dimensional action targets, and task metadata. The long-term objective is one policy that can learn across target shapes, color combinations, object counts, operators, and scenes.

## Dataset Contract

Each raw contribution should remain a standard LeRobot dataset and should expose:

- `observation.state`: six-dimensional SO-101 joint state;
- `observation.images.top`: workspace camera;
- `observation.images.wrist`: wrist camera;
- `action`: six-dimensional future joint targets;
- `task` and `task_index`: the final human-reviewed task instruction;
- `goals/`: one or more goal images;
- `meta/info.json`: goal and collection metadata.

For the first release, keep one prompt and one goal image set per raw Hugging Face dataset repository. This makes collection, review, provenance, and later merging straightforward.

```text
contributor/goalblocks_goal_000001/
├── README.md
├── data/
├── videos/
├── meta/
│   ├── info.json
│   ├── stats.json
│   ├── tasks.parquet
│   └── episodes/
└── goals/
    ├── goal_45deg.jpg
    ├── goal_top.jpg
    ├── goal_front.jpg
    └── goal_side.jpg
```

Suggested additional metadata in `meta/info.json`:

```json
{
  "robot_type": "so-101",
  "contributor": {"contributor_id": "user_a"},
  "goal": {
    "goal_id": "goal_000001",
    "goal_images": ["goals/goal_45deg.jpg"],
    "task_prompt_source": "human_edited_or_goal_image_generated",
    "task_language": "en",
    "task": "Use three red blocks to form the number 1 shown in the goal image."
  }
}
```

If a future repository contains several tasks, store an episode-level mapping such as `meta/episode_goals.parquet` with `episode_index`, `goal_id`, `goal_images`, and `prompt_id`. A single global `goal` field is only appropriate for a single-task raw repository.

## Collection Workflow

The extended LeLab workflow implemented in this repository is:

```text
Connect SO-101 and cameras
        ↓
Upload one to four goal images
        ↓
Optionally generate a draft task with Qwen-VL
        ↓
Review and edit the final task text
        ↓
Record several teleoperation episodes
        ↓
Save goal images and metadata with the LeRobot dataset
        ↓
Validate replay and publish the raw dataset
```

The Qwen API key is used only for the generation request and is not written into dataset metadata or project files. The final instruction remains editable because human review is essential for unambiguous robot supervision.

## Multi-Task Organization

LeRobot v3 can represent multiple tasks inside one dataset through `meta/tasks.parquet`, per-frame `task_index`, and episode metadata. A practical early-stage organization is nevertheless two-layered:

```text
Raw layer
  One goal/prompt per independently reviewable contribution.

Release layer
  Periodically merge approved raw datasets into a versioned multi-task release.
```

For modest data volumes, the official merge workflow is appropriate:

```text
small LeRobot datasets → lerobot-edit-dataset merge → one release dataset → training
```

Merging is not remote streaming: source datasets are downloaded or cached and a new output dataset is created. Plan for approximately one source-cache copy plus one merged-output copy, in addition to temporary processing space.

For very large collections, this project proposes multi-repository streaming rather than repeated full merges. The included GoalBlocks baseline already provides a compact multi-repository loader for pipeline validation. It is deliberately separate from LeRobot's standard training command and should not be represented as built-in `lerobot-train` support.

## Goal-Conditioned Training

Goal images stored in metadata are not automatically consumed by a standard policy. A goal-conditioned training path needs a dataset wrapper or loader that returns the current observations, robot state, language task, goal image, and action target together.

```python
{
    "observation.images.top": ...,
    "observation.images.wrist": ...,
    "observation.state": ...,
    "task": "Use three red blocks to form the number 1.",
    "goal.images": [...],
    "action": ...,
}
```

This conditioning is necessary: identical current scenes can require different actions for different target arrangements. The initial GoalBlocks baseline accepts two observation cameras, robot state, task text, and a goal image. It is an experimental pipeline baseline, not a production policy.

## Contribution and Review

Contributors should publish raw data in their own Hugging Face dataset repositories and submit the repository identifier for review. Do not grant direct write access to a shared release dataset.

Review should verify:

- SO-101 robot type and six-dimensional state/action fields;
- expected camera fields, resolution, and frame rate;
- readable episodes and decodable videos;
- presence of `goals/` and valid goal metadata;
- clear task language;
- absence of obviously unsafe, failed, or corrupted demonstrations.

An `approved_sources.jsonl` manifest can later record which repositories are eligible for training:

```jsonl
{"source_repo_id":"user_a/goalblocks_goal_000001","approved":true}
{"source_repo_id":"user_b/goalblocks_goal_000002","approved":true}
{"source_repo_id":"user_c/goalblocks_goal_000003","approved":false}
```

## Roadmap

1. Validate the full pipeline with small, single-goal raw datasets.
2. Collect more reviewed demonstrations using goal-aware LeLab recording.
3. Release periodic merged LeRobot datasets when storage is practical.
4. Improve multi-repository streaming, validation, and balanced sampling for scale.
5. Compare language-only, goal-image-only, and joint language-plus-goal-image policies.

## Reference

The survey was informed by the public [Project-IRA SO-101 multi-task LeRobot dataset](https://huggingface.co/datasets/Project-IRA/TPSoSe2026_Dataset_Full_Merged_Final_LeRobot_SO101_V1), which demonstrates a versioned merged multi-task dataset assembled from smaller source recording sessions.

## License and Scope

This document describes an independent research and community-data proposal. The implementation in this repository is an unofficial derivative built on Hugging Face LeLab and LeRobot. It preserves upstream attribution and Apache-2.0 licensing; it is not an official Hugging Face release or endorsement.

