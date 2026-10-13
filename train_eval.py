"""Generate a synthetic line, train the classifier, print test metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from dataset import build_split, save_samples  # noqa: E402
from model import load_model, save_model, timed_predict, train  # noqa: E402
from model import metrics as score  # noqa: E402

MODEL_PATH = ROOT / "models" / "defect_rf.joblib"
REPORT_PATH = ROOT / "models" / "metrics.json"
SAMPLE_DIR = ROOT / "data" / "samples"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    split = build_split()
    save_samples(SAMPLE_DIR, split["x_test"], split["y_test"])
    model = train(split["x_train"], split["y_train"])
    predicted, elapsed = timed_predict(model, split["x_test"])
    report = score(split["y_test"], predicted, elapsed)
    save_model(model, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"model: {MODEL_PATH}")
    print(f"precision:    {report['precision']:.3f}")
    print(f"recall:       {report['recall']:.3f}")
    print(f"false alarm:  {report['false_alarm_rate']:.3f}")
    print(f"accuracy:     {report['accuracy']:.3f}")
    print(f"latency:      {report['latency_ms']:.2f} ms/image")


if __name__ == "__main__":
    main()


def ready_model():
    if MODEL_PATH.exists():
        return load_model(MODEL_PATH)
    main()
    return load_model(MODEL_PATH)
