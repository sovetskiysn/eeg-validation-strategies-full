from pathlib import Path

print("Hello, World!")

results_dir = Path(__file__).resolve().parent.parent / "results"
results_dir.mkdir(exist_ok=True)
(results_dir / "justun_biber.txt").write_text("Hello, World!\n")
