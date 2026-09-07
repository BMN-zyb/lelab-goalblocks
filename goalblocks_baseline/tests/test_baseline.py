import socket
import unittest
from tempfile import TemporaryDirectory

import numpy as np
import torch
from PIL import Image

from goalblocks_baseline.dataset import encode_task, goalblocks_collate, prepare_rgb_image
from goalblocks_baseline.model import GoalConditionedActionChunkPolicy, GoalPolicyConfig
from goalblocks_baseline.remote_protocol import decode_arrays, encode_arrays, receive_message, send_message
from goalblocks_baseline.remote_server import InferenceEngine
from goalblocks_baseline.train import DEFAULT_REPOS, parse_episode_groups


class BaselineTest(unittest.TestCase):
    def test_task_encoding_is_fixed_length_and_deterministic(self):
        first = encode_task("stack the red block", max_bytes=32)
        second = encode_task("stack the red block", max_bytes=32)
        self.assertEqual(first.shape, (32,))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(first.ne(0).any())

    def test_prepare_rgb_image_accepts_hwc_uint8(self):
        image = torch.randint(0, 256, (48, 64, 3), dtype=torch.uint8)
        output = prepare_rgb_image(image, (32, 32))
        self.assertEqual(output.shape, (3, 32, 32))
        self.assertGreaterEqual(float(output.min()), 0.0)
        self.assertLessEqual(float(output.max()), 1.0)

    def test_model_forward_loss_and_prediction_shape(self):
        config = GoalPolicyConfig(action_horizon=8, hidden_dim=64, text_dim=32)
        model = GoalConditionedActionChunkPolicy(config)
        samples = []
        for _ in range(2):
            samples.append(
                {
                    "observation.images.top": torch.rand(3, 64, 64),
                    "observation.images.wrist": torch.rand(3, 64, 64),
                    "observation.state": torch.rand(6),
                    "action": torch.rand(8, 6),
                    "action_is_pad": torch.zeros(8, dtype=torch.bool),
                    "goal.image": torch.rand(3, 64, 64),
                    "task": "stack blocks",
                    "task_tokens": encode_task("stack blocks"),
                    "dataset_index": torch.tensor(0),
                    "episode_index": torch.tensor(0),
                    "repo_id": "test/repo",
                }
            )
        batch = goalblocks_collate(samples)
        output = model(batch)
        self.assertEqual(output["normalized_actions"].shape, (2, 8, 6))
        self.assertEqual(output["loss"].ndim, 0)
        self.assertEqual(model.predict_action_chunk(batch).shape, (2, 8, 6))

        with TemporaryDirectory() as directory:
            metadata = {"image_size": [64, 64], "action_names": ["joint"]}
            checkpoint = model.save_checkpoint(directory, metadata=metadata)
            restored = GoalConditionedActionChunkPolicy.load_checkpoint(checkpoint)
            self.assertEqual(restored.load_metadata(checkpoint), metadata)
            self.assertTrue(
                torch.allclose(model.predict_action_chunk(batch), restored.predict_action_chunk(batch))
            )

    def test_remote_protocol_round_trip(self):
        left, right = socket.socketpair()
        try:
            expected = {"state": np.arange(6, dtype=np.float32), "top": np.zeros((8, 8, 3), np.uint8)}
            send_message(left, encode_arrays(**expected))
            actual = decode_arrays(receive_message(right))
            for key in expected:
                np.testing.assert_array_equal(actual[key], expected[key])
        finally:
            left.close()
            right.close()

    def test_episode_group_parser(self):
        parsed = parse_episode_groups("0,1;0", DEFAULT_REPOS)
        self.assertEqual(parsed[DEFAULT_REPOS[0]], [0, 1])
        self.assertEqual(parsed[DEFAULT_REPOS[1]], [0])

    def test_inference_engine_accepts_wire_format(self):
        config = GoalPolicyConfig(action_horizon=4, hidden_dim=32, text_dim=16)
        model = GoalConditionedActionChunkPolicy(config)
        with TemporaryDirectory() as directory:
            checkpoint = model.save_checkpoint(directory, metadata={"image_size": [32, 32]})
            goal_path = f"{directory}/goal.jpg"
            Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(goal_path)
            engine = InferenceEngine(str(checkpoint), goal_path, "stack blocks", "cpu")
            actions = engine.predict(
                {
                    "top": np.zeros((48, 64, 3), dtype=np.uint8),
                    "wrist": np.zeros((48, 64, 3), dtype=np.uint8),
                    "state": np.zeros(6, dtype=np.float32),
                }
            )
            self.assertEqual(actions.shape, (4, 6))


if __name__ == "__main__":
    unittest.main()
