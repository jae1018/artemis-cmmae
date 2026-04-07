"""
Quickstart example for artemis-cmmae.

Loads the bundled sample data (1 month of ARTEMIS observations),
runs the classifier, and prints a summary of predictions.
"""

from artemis_cmmae import PlasmaClassifier, load_sample_data

# Load sample data (bundled with the package)
df = load_sample_data()
print(f"Loaded {len(df)} observations from {df.index[0].date()} to {df.index[-1].date()}")

# Initialize classifier (alpha=1 is the default and the model reported in the paper)
clf = PlasmaClassifier(alpha=1)

# Run predictions
predictions, label_map = clf.predict(df)

# Print summary
print("\nPrediction summary:")
for label_id, name in sorted(label_map.items()):
    count = (predictions == label_id).sum()
    if count > 0:
        print(f"  {name:>15s} (id={label_id:2d}): {count:6d}  ({100 * count / len(predictions):.1f}%)")
