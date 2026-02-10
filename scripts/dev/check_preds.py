#!/usr/bin/env python
"""Check and compact predictions.json."""
import json

with open("web/predictions.json") as f:
    preds = json.load(f)

print(f"Number of predictions: {len(preds)}")

# Check first prediction structure
first_key = list(preds.keys())[0]
first_pred = preds[first_key]
print(f"Keys in prediction: {list(first_pred.keys())}")
print(f"Coords shape: {len(first_pred['coords'])} atoms x 3")

# Round coordinates to 2 decimal places to save space (0.01 Angstrom precision)
for sample_id, pred in preds.items():
    pred['coords'] = [[round(c, 2) for c in atom] for atom in pred['coords']]
    pred['rmsd'] = round(pred['rmsd'], 3)
    pred['inference_time'] = round(pred['inference_time'], 3)

# Save compactly (no indentation)
with open("web/predictions.json", "w") as f:
    json.dump(preds, f, separators=(',', ':'))

print("Saved compactly with rounded coords")
