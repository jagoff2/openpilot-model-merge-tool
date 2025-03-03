# ONNX Model Weight Merger for Comma.ai Openpilot

This tool merges the weights of two comma.ai openpilot supercombo.onnx models while preserving the overall model structure. It's designed to combine the trained weights from different models into a single model that can leverage the strengths of both source models.

## Features

- Preserves the model structure while merging weights
- Supports fine-grained component-level merging
- Adjustable mixing ratio to control the influence of each source model
- Compatible with all comma.ai supercombo.onnx models that share the same architecture
- Detailed logging and optional weight exports for verification

## Requirements

- Python 3.7+
- ONNX (>=1.10.0)
- NumPy

## Installation

```bash
# Clone the repository
git clone https://github.com/jagoff2/openpilot-model-merge-tool.git
cd openpilot-model-merger

# Install dependencies
pip install onnx numpy
```

## Usage

### Basic Usage

To merge two models with default settings (50/50 weight mix):

```bash
python merge_supercombo_fixed.py model_a.onnx model_b.onnx --output merged_supercombo.onnx
```

### Advanced Usage

Control the mixing ratio (0.0 = pure model A, 1.0 = pure model B):

```bash
# 70% model B, 30% model A
python merge_supercombo_fixed.py model_a.onnx model_b.onnx --output merged.onnx --ratio 0.7
```

Target specific components for merging:

```bash
# Only merge the action_block component (lateral/longitudinal control)
python merge_supercombo_fixed.py model_a.onnx model_b.onnx --output merged.onnx --components action_block

# Merge just lane line detection components
python merge_supercombo_fixed.py model_a.onnx model_b.onnx --output merged.onnx --components lane_lines lane_lines_prob

# Merge only the lead vehicle detection and plan components
python merge_supercombo_fixed.py model_a.onnx model_b.onnx --output merged.onnx --components lead lead_prob plan
```

List all available components:

```bash
python merge_supercombo_fixed.py --list-components
```

Export weights for verification:

```bash
python merge_supercombo_fixed.py model_a.onnx model_b.onnx --output merged.onnx --save-weights
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `model_a` | First input model (provides base structure) | (required) |
| `model_b` | Second input model (weights to merge) | (required) |
| `--output` | Path for the merged output model | merged_supercombo.onnx |
| `--components` | Components to merge (space-separated) | no_bottleneck_policy temporal_hydra temporal_summarizer action_block |
| `--ratio` | Weight of model B's values (0.0 = pure A, 1.0 = pure B) | 0.5 |
| `--save-weights` | Export weights as NPZ files for inspection | (flag) |
| `--list-components` | List all available components and exit | (flag) |

## Available Components

### High-level components
- `vision` - Vision encoder (CNN)
- `no_bottleneck_policy` - Policy network
- `temporal_summarizer` - Temporal feature summarizer
- `temporal_hydra` - Temporal policy heads
- `action_block` - Control actions (lateral/longitudinal)

### Policy-specific components
- `plan` - Path planning components
- `lead` - Lead vehicle detection
- `lead_prob` - Lead probability estimation
- `lane_lines` - Lane line detection
- `lane_lines_prob` - Lane line probability
- `road_edges` - Road edge detection
- `desire_state` - Desired state prediction

### Control-specific components
- `action_block_out` - Final action outputs
- `action_block_in` - Action input processing
- `resblocks` - Decision blocks

### Feature processing components
- `transformer` - Attention mechanisms
- `summarizer_resblock` - Residual blocks in summarizer
- `temporal_resblock` - Residual blocks in temporal

## How It Works

The script identifies components in the models based on node name prefixes. For each targeted component, the script:

1. Identifies all weight tensors (initializers) used by the component
2. Verifies that both models have matching tensor shapes
3. Computes a weighted average of the corresponding tensors based on the specified ratio
4. Replaces the tensors in the output model with the merged weights

The resulting model maintains the structure of the first input model but with weights that combine information from both models.

## Typical Merging Scenarios

### Improving Lane Detection? Maybe?
```bash
python merge_supercombo_fixed.py model_a.onnx model_b.onnx --components lane_lines lane_lines_prob --ratio 0.7
```

### Enhancing Lateral Control? Worse lateral control? Who knows?
```bash
python merge_supercombo_fixed.py model_a.onnx model_b.onnx --components action_block_out --ratio 0.8
```

### Better Lead Vehicle Tracking? No lead vehicle tracking? Find out!
```bash
python merge_supercombo_fixed.py model_a.onnx model_b.onnx --components lead lead_prob --ratio 0.6
```

### Improved Path Planning? Sent directly into the ditch? Endless possibilties. 
```bash
python merge_supercombo_fixed.py model_a.onnx model_b.onnx --components plan --ratio 0.75
```

## Troubleshooting

### Model Structure Incompatibility

If you see errors about shape mismatches, verify that both input models have the same architecture. The merging only works for models with identical structure.

### Missing Components

If the script reports 0 nodes for a component, check that your model uses the expected naming conventions. You can inspect the model structure using tools like Netron.

### Weight Verification

Use the `--save-weights` flag to export the weights of all models for verification. You can load the NPZ files in Python and compare the values to ensure proper merging.

```python
import numpy as np

# Load weight files
weights_a = np.load('model_a_weights.npz')
weights_b = np.load('model_b_weights.npz')
weights_merged = np.load('merged_weights.npz')

# Check a specific weight, e.g., the first lateral control weight
name = 'temporal_policy.action_block.action_block_out.weight'
print("Weight A:", weights_a[name])
print("Weight B:", weights_b[name])
print("Merged:", weights_merged[name])
print("Expected:", 0.5 * weights_a[name] + 0.5 * weights_b[name])
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the architecture of comma.ai's openpilot supercombo.onnx models
- Inspired by weight averaging techniques from the deep learning community
