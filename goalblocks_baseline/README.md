# GoalBlocks Training and Remote Inference

This package trains a compact goal-conditioned SO-101 policy from multiple independent LeRobot datasets without changing LeRobot core code.

## Inputs and Output

The policy consumes a top-camera image, wrist-camera image, goal image, natural-language task encoded as UTF-8 bytes, and six-dimensional robot state. It predicts a configurable chunk of six-dimensional joint targets.

The model has 738,432 parameters (0.74M). A comparable two-camera LeRobot ACT configuration has approximately 51.55M parameters. This model is a pipeline baseline, not the final proposed GRAE-VLA architecture.

## Formal Training

```bash
cd /workspace
CUDA_VISIBLE_DEVICES=2 python -m goalblocks_baseline.train \
  --device cuda:0 \
  --steps 10000 \
  --batch-size 16 \
  --output-dir /workspace/outputs/goalblocks_baseline_formal
```

The defaults train on episodes 0 and 1 from the first repository and episode 0 from the second. Episode 2 from the first repository and episode 1 from the second are used for validation. Repositories remain separate and are balanced by the sampler.

Outputs include:

- `best_model.pt`: lowest validation-loss deployment checkpoint;
- `last_model.pt`: most recent periodic deployment checkpoint;
- `last_training.pt`: model, optimizer, scheduler, scaler, and step state for resume;
- `metrics.jsonl`: machine-readable training and validation metrics;
- `checkpoints/step_*.pt`: periodic model snapshots.

Running the same command again automatically resumes from `last_training.pt` when it exists.

## Remote GPU Server

The baseline is not a registered LeRobot `PreTrainedPolicy` and requires goal and task inputs that the standard policy server does not provide. The standalone server avoids modifications to LeRobot's registry or core inference code.

```bash
CUDA_VISIBLE_DEVICES=2 python -m goalblocks_baseline.remote_server \
  --checkpoint /workspace/outputs/goalblocks_baseline_formal/best_model.pt \
  --goal-image /path/to/goal.jpg \
  --task "Place the red block on top of the blue block." \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 8080
```

Do not expose port 8080 to the public internet. Use an SSH tunnel or a trusted private network.

## WSL Robot Client

Check device mappings before every run:

```bash
ls -l /dev/serial/by-id/
v4l2-ctl --list-devices
```

The default mode captures observations and requests predictions but does not send actions:

```bash
python -m goalblocks_baseline.remote_client \
  --server-host 127.0.0.1 \
  --robot-port /dev/serial/by-id/YOUR_FOLLOWER \
  --robot-id my_follower \
  --top-camera /dev/video0 \
  --wrist-camera /dev/video2
```

Only add `--execute` after validating the tunnel, latency, image streams, state order, calibration, and predictions. The client applies a five-degree relative target limit by default and stops on a two-second network timeout.

## Limitations

The initial training run used only five episodes and was stopped early. One top-camera stream contains known corrupt JPEG frames. The checkpoint is suitable for software integration experiments only and must not be treated as a reliable autonomous controller.
