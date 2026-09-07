import base64
import json
from io import BytesIO


def test_save_goal_assets_writes_image_and_metadata(tmp_path) -> None:
    from lelab.goals import GoalImage, save_goal_assets

    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text('{"fps": 30}', encoding="utf-8")
    image = GoalImage(name="目标 图.png", mime_type="image/png", data=base64.b64encode(b"png").decode())

    paths = save_goal_assets(tmp_path, [image], "摆出红色数字 1。")

    assert paths == ["goals/01_goal_1.png"]
    assert (tmp_path / paths[0]).read_bytes() == b"png"
    info = json.loads((tmp_path / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["goal"]["task"] == "摆出红色数字 1。"
    assert "api" not in json.dumps(info).lower()


def test_generate_goal_description_sends_key_without_storing_it(monkeypatch) -> None:
    from lelab import goals

    captured = {}

    class Response(BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    def fake_urlopen(request, timeout):
        captured["authorization"] = request.headers["Authorization"]
        captured["body"] = json.loads(request.data)
        return Response(b'{"choices":[{"message":{"content":"pick the red block"}}]}')

    monkeypatch.setattr(goals.urllib.request, "urlopen", fake_urlopen)
    request = goals.GoalDescriptionRequest(
        api_key="secret-key",
        images=[goals.GoalImage(name="goal.png", mime_type="image/png", data=base64.b64encode(b"x").decode())],
    )

    assert goals.generate_goal_description(request) == "pick the red block"
    assert captured["authorization"] == "Bearer secret-key"
    assert "secret-key" not in json.dumps(captured["body"])


def test_write_goal_metadata_survives_final_info_contents(tmp_path) -> None:
    from lelab.goals import write_goal_metadata

    (tmp_path / "meta").mkdir()
    info_path = tmp_path / "meta" / "info.json"
    info_path.write_text('{"total_episodes": 3, "total_frames": 5488}', encoding="utf-8")

    write_goal_metadata(tmp_path, ["goals/01_goal.jpg"], "stack the blocks")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    assert info["total_episodes"] == 3
    assert info["goal"]["goal_images"] == ["goals/01_goal.jpg"]
    assert info["goal"]["task"] == "stack the blocks"
