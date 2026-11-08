"""Embedded guide retrieval and role selection; no GPU, weights or media required."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

exe = str(Path(sys.argv[1]).resolve())

with tempfile.TemporaryDirectory(prefix="slopfab-prompts-") as temporary:
    root = Path(temporary)
    config = root / "config.json"
    config.write_text(json.dumps({"schema_version": 1, "weight_paths": [],
                                 "models": {"MyVideo": {"family": "h3"}}}), encoding="utf-8")

    def run(*args, code=0):
        result = subprocess.run([exe, "--agent", "--config", str(config), *args], cwd=root,
                                capture_output=True, text=True, encoding="utf-8", timeout=30)
        assert result.returncode == code, (args, result.stdout, result.stderr)
        events = [json.loads(line) for line in result.stdout.splitlines()]
        assert len(events) == 1, events
        event = events[0]
        assert event["type"] == ("error" if code else "result"), event
        return event

    def select(model="h3", *args):
        return run("prompts", "select", "--model", model, *args)["data"]

    first = run("prompts", "list")
    assert "prompting guide" in first["agent_context"]["instructions"]
    index = first["data"]["guides"]
    assert len(index) == 19
    assert {g["model"] for g in index} == {"h3", "krea2", "moss", "fibo2", "yue2", "minimax_music3"}
    for entry in index:
        guide = run("prompts", "get", entry["model"], entry["use_case"])["data"]
        assert guide["prompt_template"] and guide["example"]["prompt"]
        assert guide["model_guidance"]["sources"] and guide["model_guidance"]["slopfab_support"]
    assert len(run("prompts", "list", "--model", "MyVideo")["data"]["guides"]) == 9
    assert run("prompts", "get", "h3", "FL2VA")["data"]["use_case"] == "fl2va"
    assert select()["use_case"] == "t2va"
    assert select("MyVideo")["use_case"] == "t2va"

    initial = select("h3", "--first-frame", "opening å.png")
    assert initial["use_case"] == "i2va" and "0.00 seconds" in initial["prompt_template"]
    assert initial["reference_bindings"][0]["path"] == "opening å.png"
    assert initial["reference_bindings"][0]["generation_option"] == "--reference-image"
    last = select("h3", "--last-frame", "ending.png", "--duration", "5.875", "--shots", "2")
    assert last["use_case"] == "l2va"
    assert "<Picture 1> (from [Shot 2])" in last["prompt_template"]
    assert "5.88-second" in last["prompt_template"] and "Picture 2" not in last["prompt_template"]
    assert last["reference_bindings"][0]["generation_option"] == "--end-anchor"
    both = select("h3", "--last-frame", "ending.png", "--first-frame", "opening.png")
    assert both["use_case"] == "fl2va" and "{{duration}}" in both["prompt_template"]
    assert [b["path"] for b in both["reference_bindings"]] == ["opening.png", "ending.png"]
    assert [b["label"] for b in both["reference_bindings"]] == ["<Picture 1>", "<Picture 2>"]
    assert "Ref2VA" in both["binding"]
    generic = select("h3", "--reference-image", "identity.png", "--reference-video", "motion.y4m",
                     "--reference-audio", "voice.wav")
    assert generic["use_case"] == "ref2va" and "0.00-second" not in generic["prompt_template"]
    swapped = select("h3", "--character-swap", "--reference-image", "character.png",
                     "--reference-video", "source.y4m")
    assert swapped["use_case"] == "character-swap"
    assert [b["label"] for b in swapped["reference_bindings"]] == ["<Picture 1>", "<Video 1>"]
    assert [b["role"] for b in swapped["reference_bindings"]] == ["replacement character identity", "source performance and scene"]
    assert "--lora" in swapped["slopfab_support"] and "Turbo" in swapped["slopfab_support"]
    for args in [[], ['--reference-image', 'a'], ['--reference-video', 'v'],
                 ['--reference-image', 'a', '--reference-video', 'v', '--first-frame', 'b'],
                 ['--reference-image', 'a', '--reference-video', 'v', '--shots', '2'],
                 ['--reference-image', 'a', '--reference-image', 'b', '--reference-video', 'v']]:
        run('prompts', 'select', '--model', 'h3', '--character-swap', *args, code=2)
    run('prompts', 'select', '--model', 'moss', '--character-swap', code=2)
    assert [b["label"] for b in generic["reference_bindings"]] == ["<Picture 1>", "<Video 1>", "<Audio 1>"]
    assert select("h3", "--continuation")["use_case"] == "continuation"
    assert select("h3", "--task", "image")["use_case"] == "image"
    assert select("h3", "--task", "image", "--edit-image", "source.png")["use_case"] == "edit"
    # Option-like path values remain opaque to both global --agent and help parsing.
    for path in ("--agent", "--help"):
        assert select("h3", "--first-frame", path)["reference_bindings"][0]["path"] == path

    assert select("krea2")["use_case"] == "image"
    layout = select("moss", "--layout")
    layout_prompt = json.loads(layout["example"]["prompt"])
    assert layout_prompt["bbox_format"] == "yxyx"
    assert layout_prompt["bbox_coord_space"] == "abs_centralized"
    for item in layout_prompt["items"]:
        assert all(-512 <= n <= 512 for n in item["bbox"])
    fibo = select("fibo2", "--reference-image", "scene.png", "--reference-image", "object.png")
    assert fibo["use_case"] == "multi-reference"
    assert [b["label"] for b in fibo["reference_bindings"]] == ["image_1", "image_2"]
    assert "structured_caption" in json.loads(fibo["example"]["prompt"])
    assert select("fibo2", "--reference-image", "source.png")["reference_bindings"][0]["label"] == "image"
    for model in ("yue2", "minimax_music3"):
        song = select(model)
        instrumental = select(model, "--instrumental")
        assert song["use_case"] == "song" and song["example"]["lyrics"]
        assert song["example"]["lyrics"] not in song["example"]["prompt"]
        assert instrumental["use_case"] == "instrumental" and instrumental["example"]["lyrics"] == ""
        assert "no vocals" in instrumental["prompt_template"]

    bad = [(), ("--model", "unknown"), ("--model", "h3", "--typo"),
           ("--model", "h3", "--duration", "nan"), ("--model", "h3", "--duration", "inf"),
           ("--model", "h3", "--duration", "-1"), ("--model", "h3", "--shots", "1.5"),
           ("--model", "h3", "--first-frame"), ("--model", "h3", "--layout"),
           ("--model", "h3", "--first-frame", "a", "--reference-image", "b"),
           ("--model", "h3", "--continuation", "--last-frame", "b"),
           ("--model", "h3", "--reference-audio", "audio.wav"),
           ("--model", "krea2", "--reference-image", "a"),
           ("--model", "moss", "--edit-image", "a"),
           ("--model", "fibo2", "--edit-image", "a", "--reference-image", "b"),
           ("--model", "fibo2", "--task", "video"),
           ("--model", "yue2", "--reference-audio", "a"),
           ("--model", "minimax_music3", "--first-frame", "a"),
           ("--model", "h3", "--model", "h3")]
    for args in bad:
        run("prompts", "select", *args, code=2)
    run("prompts", "list", "--model", "unknown", code=2)
    run("prompts", "get", "krea2", "fl2va", code=2)
    assert run("system", "capabilities")["data"]["prompting"]["offline"]
    # Help is available without any models or the original catalog file nearby.
    result = subprocess.run([exe, "--agent", "--config", str(config), "help", "prompts"],
                            cwd=root, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0 and "--first-frame" in result.stdout

print("prompting CLI passed")
