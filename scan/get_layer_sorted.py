"""SS-Score scan, step 3/3: rank layers and pick the top-k safety-critical ones.

Usage:
  python scan/get_layer_sorted.py scan/features_llama-8b/res.log
"""
import json


def run_main(log_path):
    records = []
    with open(log_path, "r") as f:
        for line in f:
            records.append(json.loads(line))
    # lowest validation loss first
    records = sorted(records, key=lambda x: x["val_loss"])
    for record in records:
        print(record)
    print("\ntop-3 safety-critical layers:",
          [r["layer_name"] for r in records[:3]])


if __name__ == "__main__":
    import fire
    fire.Fire(run_main)
