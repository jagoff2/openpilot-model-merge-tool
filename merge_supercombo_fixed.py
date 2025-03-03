#!/usr/bin/env python3
"""
Merge two comma.ai openpilot supercombo.onnx models by combining weights of specific
components after the shared CNN trunk. This preserves the model structure while merging
the weights of specific components (policy, action_block, temporal components).

The common trunk output is expected to be:
    /supercombo/vision/Flatten_output_0

Usage:
    python merge_supercombo_fixed.py model_a.onnx model_b.onnx --output merged_supercombo.onnx --ratio 0.5
"""

import onnx
import numpy as np
from onnx import numpy_helper
import argparse
import os
import re

def output_exists(model, target_output):
    """Return True if target_output appears among any node or graph outputs."""
    for node in model.graph.node:
        if target_output in node.output:
            return True
    for out in model.graph.output:
        if out.name == target_output:
            return True
    return False

def build_node_map(model):
    """
    Build a mapping of node names to nodes.
    """
    node_map = {}
    for i, node in enumerate(model.graph.node):
        name = node.name if node.name else f"unnamed_node_{i}"
        node_map[name] = node
    return node_map

def identify_components_by_prefix(model):
    """
    Identify model components based on node name prefixes.
    Returns a dictionary of component prefixes to sets of node indices.
    """
    components = {
        # High-level components
        "vision": set(),                     # Vision encoder (CNN)
        "no_bottleneck_policy": set(),       # Policy network 
        "temporal_summarizer": set(),        # Temporal feature summarizer
        "temporal_hydra": set(),             # Temporal policy heads
        "action_block": set(),               # Control actions (likely lateral/longitudinal)
        
        # Policy-specific components
        "plan": set(),                       # Path planning components
        "lead": set(),                       # Lead vehicle detection
        "lead_prob": set(),                  # Lead probability
        "lane_lines": set(),                 # Lane line detection
        "lane_lines_prob": set(),            # Lane line probability
        "road_edges": set(),                 # Road edge detection
        "desire_state": set(),               # Desired state prediction
        
        # Control-specific components
        "action_block_out": set(),           # Final action outputs
        "action_block_in": set(),            # Action input processing
        "resblocks": set(),                  # Decision blocks
        
        # Feature processing components
        "transformer": set(),                # Attention mechanisms
        "summarizer_resblock": set(),        # Residual blocks in summarizer
        "temporal_resblock": set(),          # Residual blocks in temporal
        
        # Model outputs
        "output_plan": set(),                # Plan output nodes
        "output_lane_lines": set(),          # Lane lines output nodes
        "output_lane_lines_prob": set(),     # Lane lines prob output nodes
        "output_road_edges": set(),          # Road edges output nodes
        "output_lead": set(),                # Lead output nodes
        "output_lead_prob": set(),           # Lead prob output nodes
        "output_desire_state": set(),        # Desire state output nodes
        "output_meta": set(),                # Meta output nodes
        "output_desire_pred": set(),         # Desire prediction output nodes
        "output_pose": set(),                # Pose output nodes
        "output_wide_from_device": set(),    # Wide from device output nodes
        "output_sim_pose": set(),            # Sim pose output nodes
        "output_road_transform": set(),      # Road transform output nodes
        "output_lateral": set(),             # Lateral action output nodes
        "output_longitudinal": set(),        # Longitudinal action output nodes
        
        # Vision subsystems
        "vision_encoder": set(),             # Vision encoder
        "vision_decoder": set(),             # Vision decoder
        "vision_features": set(),            # Vision features extraction
        
        # Specific module types
        "gemm": set(),                       # All GEMM operations
        "conv": set(),                       # All Conv operations
        "relu": set(),                       # All ReLU activations
    }
    
    # Map each node to its component based on name prefix
    for i, node in enumerate(model.graph.node):
        name = node.name
        if not name:
            continue
            
        # High-level components
        if "/supercombo/vision/" in name:
            components["vision"].add(i)
        elif "/supercombo/no_bottleneck_policy/" in name:
            components["no_bottleneck_policy"].add(i)
        elif "/temporal_summarizer/" in name:
            components["temporal_summarizer"].add(i)
        elif "/temporal_hydra/" in name:
            components["temporal_hydra"].add(i)
        elif "/action_block/" in name:
            components["action_block"].add(i)
            
        # Policy-specific components
        if "/plan/" in name or "/plan_1/" in name:
            components["plan"].add(i)
        if "/lead/" in name or "/lead_1/" in name:
            components["lead"].add(i)
        if "/lead_prob/" in name or "/lead_prob_1/" in name:
            components["lead_prob"].add(i)
        if "/lane_lines/" in name or "/lane_lines_1/" in name:
            components["lane_lines"].add(i)
        if "/lane_lines_prob/" in name or "/lane_lines_prob_1/" in name:
            components["lane_lines_prob"].add(i)
        if "/road_edges/" in name or "/road_edges_1/" in name:
            components["road_edges"].add(i)
        if "/desire_state/" in name or "/desire_state_1/" in name:
            components["desire_state"].add(i)
            
        # Control-specific components
        if "/action_block/action_block_out/" in name:
            components["action_block_out"].add(i)
        if "/action_block/action_block_in/" in name:
            components["action_block_in"].add(i)
        if "/action_block/resblocks" in name:
            components["resblocks"].add(i)
        
        # Feature processing components
        if "/transformer/" in name:
            components["transformer"].add(i)
        if "/summarizer/resblock/" in name:
            components["summarizer_resblock"].add(i)
        if "/temporal_hydra/resblock/" in name:
            components["temporal_resblock"].add(i)
        
        # Vision subsystems
        if "/vision/_en/" in name:
            components["vision_encoder"].add(i)
        if "/vision/_de/" in name:
            components["vision_decoder"].add(i)
        if "/vision/features/" in name:
            components["vision_features"].add(i)
        
        # Specific operation types
        if "/Gemm" in name or "Gemm_output" in name:
            components["gemm"].add(i)
        if "/Conv" in name or "Conv_output" in name:
            components["conv"].add(i)
        if "/Relu" in name or "Relu_output" in name:
            components["relu"].add(i)
        
        # Output nodes
        if "/temporal_hydra/plan_1/Gemm_output" in name:
            components["output_plan"].add(i)
        if "/temporal_hydra/lane_lines_1/Gemm_output" in name:
            components["output_lane_lines"].add(i)
        if "/temporal_hydra/lane_lines_prob_1/Gemm_output" in name:
            components["output_lane_lines_prob"].add(i)
        if "/temporal_hydra/road_edges_1/Gemm_output" in name:
            components["output_road_edges"].add(i)
        if "/temporal_hydra/lead_1/Gemm_output" in name:
            components["output_lead"].add(i)
        if "/temporal_hydra/lead_prob_1/Gemm_output" in name:
            components["output_lead_prob"].add(i)
        if "/temporal_hydra/desire_state_1/Gemm_output" in name:
            components["output_desire_state"].add(i)
        if "/supercombo/no_bottleneck_policy/hydra/meta_1/Gemm_output" in name:
            components["output_meta"].add(i)
        if "/supercombo/no_bottleneck_policy/hydra/desire_pred_1/Gemm_output" in name:
            components["output_desire_pred"].add(i)
        if "/supercombo/no_bottleneck_policy/hydra/pose_1/Gemm_output" in name:
            components["output_pose"].add(i)
        if "/supercombo/no_bottleneck_policy/hydra/wide_from_device_euler_1/Gemm_output" in name:
            components["output_wide_from_device"].add(i)
        if "/temporal_hydra/sim_pose_1/Gemm_output" in name:
            components["output_sim_pose"].add(i)
        if "/supercombo/no_bottleneck_policy/hydra/road_transform_1/Gemm_output" in name:
            components["output_road_transform"].add(i)
        if "/action_block/Mul_output" in name and "lateral" in node.output[0].lower():
            components["output_lateral"].add(i)
        if "/action_block/Mul_output" in name and "longitudinal" in node.output[0].lower():
            components["output_longitudinal"].add(i)
        # If we can't determine specific lateral/longitudinal outputs, use action_block_out for control
    
    return components

def get_initializer_dict(model):
    """Create a mapping of initializer names to initializers."""
    return {init.name: init for init in model.graph.initializer}

def get_initializer_values_dict(model):
    """Create a mapping of initializer names to numpy arrays."""
    return {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}

def get_initializers_for_nodes(model, node_indices):
    """
    Get all initializers used by the specified nodes.
    Returns a dictionary mapping initializer names to initializers.
    """
    # Build a set of all input names for the specified nodes
    node_inputs = set()
    for i in node_indices:
        node = model.graph.node[i]
        for inp in node.input:
            node_inputs.add(inp)
    
    # Find initializers matching these inputs
    init_dict = get_initializer_dict(model)
    component_initializers = {}
    
    for input_name in node_inputs:
        if input_name in init_dict:
            component_initializers[input_name] = init_dict[input_name]
    
    return component_initializers

def merge_weights(array_a, array_b, ratio=0.5):
    """
    Merge two numpy arrays using a weighted average.
    ratio: Weight of array_b (0 = pure A, 1 = pure B, 0.5 = equal mix)
    """
    return (1 - ratio) * array_a + ratio * array_b

def create_merged_initializer(init_a, array_merged):
    """Create a new initializer with the merged values but the same metadata as init_a."""
    merged_initializer = onnx.TensorProto()
    merged_initializer.CopyFrom(init_a)
    merged_initializer.raw_data = b''  # Clear existing data
    merged_initializer.raw_data = numpy_helper.from_array(array_merged).raw_data
    return merged_initializer

def merge_models_by_components(model_a, model_b, trunk_output, components_to_merge=None, ratio=0.5):
    """
    Merge models by combining weights of specific components from model_a and model_b.
    
    Args:
        model_a: The base model to keep the structure of
        model_b: The model to extract components from
        trunk_output: The name of the common trunk output
        components_to_merge: List of component names to merge 
        ratio: Weight of model_b's values (0 = pure A, 1 = pure B, 0.5 = equal mix)
    
    Returns:
        Merged model
    """
    if components_to_merge is None:
        components_to_merge = ["no_bottleneck_policy", "temporal_hydra", "temporal_summarizer", "action_block"]
    
    # Make a copy of model_a as our base
    merged_model = onnx.ModelProto()
    merged_model.CopyFrom(model_a)
    
    # Identify components in both models
    components_a = identify_components_by_prefix(model_a)
    components_b = identify_components_by_prefix(model_b)
    
    print("Components identified in model A:")
    for component, nodes in components_a.items():
        if len(nodes) > 0:
            print(f"  - {component}: {len(nodes)} nodes")
    
    print("Components identified in model B:")
    for component, nodes in components_b.items():
        if len(nodes) > 0:
            print(f"  - {component}: {len(nodes)} nodes")
    
    # For each component to merge, get its initializers from both models
    initializers_to_merge = {}
    for component in components_to_merge:
        if component in components_a and components_a[component]:
            if component in components_b and components_b[component]:
                initializers_a = get_initializers_for_nodes(model_a, components_a[component])
                initializers_b = get_initializers_for_nodes(model_b, components_b[component])
                
                # Find common initializers between models A and B
                common_initializers = set(initializers_a.keys()) & set(initializers_b.keys())
                initializers_to_merge[component] = common_initializers
                print(f"Found {len(common_initializers)} common initializers for {component}")
    
    # Convert initializers to numpy arrays for easier manipulation
    values_a = get_initializer_values_dict(model_a)
    values_b = get_initializer_values_dict(model_b)
    
    # Get initializers from both models
    init_dict_a = get_initializer_dict(merged_model)
    init_dict_b = get_initializer_dict(model_b)
    
    # Create list of all component initializers to merge
    all_initializers_to_merge = set()
    for component, initializers in initializers_to_merge.items():
        all_initializers_to_merge.update(initializers)
    
    # Merge weights and replace in the merged model
    merged_count = 0
    for name in all_initializers_to_merge:
        if name in values_a and name in values_b:
            # Check if shapes are compatible
            if values_a[name].shape == values_b[name].shape:
                # Create merged weight array
                merged_array = merge_weights(values_a[name], values_b[name], ratio)
                
                # Create new initializer with merged weights
                merged_initializer = create_merged_initializer(init_dict_a[name], merged_array)
                
                # Replace in merged model
                for i, init in enumerate(merged_model.graph.initializer):
                    if init.name == name:
                        merged_model.graph.initializer.remove(init)
                        merged_model.graph.initializer.append(merged_initializer)
                        merged_count += 1
                        break
            else:
                print(f"Warning: Shape mismatch for initializer {name}, skipping. A: {values_a[name].shape}, B: {values_b[name].shape}")
    
    print(f"Merged {merged_count} initializers using ratio {ratio:.2f} (B) / {1-ratio:.2f} (A)")
    
    return merged_model

def save_model_weights_as_npz(model, output_path):
    """Extract and save all weights from a model as a npz file for easier inspection."""
    weights = {}
    for initializer in model.graph.initializer:
        # Convert the initializer to a numpy array
        np_array = numpy_helper.to_array(initializer)
        weights[initializer.name] = np_array
    
    np.savez(output_path, **weights)
    print(f"Saved {len(weights)} weights to {output_path}")

def get_available_components():
    """Return a list of all available component names for merging."""
    return [
        # High-level components
        "vision",
        "no_bottleneck_policy", 
        "temporal_summarizer",
        "temporal_hydra",
        "action_block",
        
        # Policy-specific components
        "plan",
        "lead",
        "lead_prob",
        "lane_lines",
        "lane_lines_prob",
        "road_edges",
        "desire_state",
        
        # Control-specific components
        "action_block_out",
        "action_block_in",
        "resblocks",
        
        # Feature processing components
        "transformer",
        "summarizer_resblock",
        "temporal_resblock",
        
        # Model outputs
        "output_plan",
        "output_lane_lines",
        "output_lane_lines_prob",
        "output_road_edges",
        "output_lead",
        "output_lead_prob",
        "output_desire_state",
        "output_meta",
        "output_desire_pred",
        "output_pose",
        "output_wide_from_device",
        "output_sim_pose",
        "output_road_transform",
        "output_lateral",
        "output_longitudinal",
        
        # Vision subsystems
        "vision_encoder",
        "vision_decoder",
        "vision_features",
        
        # Specific module types
        "gemm",
        "conv",
        "relu",
    ]

def main():
    # Get all available components
    all_components = get_available_components()
    
    # Default components to merge
    default_components = [
        "no_bottleneck_policy", 
        "temporal_hydra", 
        "temporal_summarizer", 
        "action_block"
    ]
    
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Merge two comma.ai openpilot supercombo.onnx models by combining weights of specific components."
    )
    parser.add_argument("model_a", help="Path to first ONNX model (base structure)")
    parser.add_argument("model_b", help="Path to second ONNX model (components to extract)")
    parser.add_argument("--output", default="merged_supercombo.onnx", help="Filename for the merged ONNX model")
    parser.add_argument("--components", nargs="+", default=default_components,
                      help="Components to merge from model B into model A")
    parser.add_argument("--ratio", type=float, default=0.5, 
                      help="Weight of model B's values (0 = pure A, 1 = pure B, 0.5 = equal mix)")
    parser.add_argument("--save-weights", action="store_true", help="Save model weights as NPZ files for inspection")
    parser.add_argument("--list-components", action="store_true", help="List all available components and exit")
    args = parser.parse_args()
    
    # If requested, just list the available components and exit
    if args.list_components:
        print("Available components for merging:")
        
        print("\n# High-level components")
        print("vision                - Vision encoder (CNN)")
        print("no_bottleneck_policy  - Policy network")
        print("temporal_summarizer   - Temporal feature summarizer")
        print("temporal_hydra        - Temporal policy heads")
        print("action_block          - Control actions (lateral/longitudinal)")
        
        print("\n# Policy-specific components")
        print("plan                  - Path planning components")
        print("lead                  - Lead vehicle detection")
        print("lead_prob             - Lead probability estimation")
        print("lane_lines            - Lane line detection")
        print("lane_lines_prob       - Lane line probability")
        print("road_edges            - Road edge detection")
        print("desire_state          - Desired state prediction")
        
        print("\n# Control-specific components")
        print("action_block_out      - Final action outputs")
        print("action_block_in       - Action input processing")
        print("resblocks             - Decision blocks")
        
        print("\n# Feature processing components")
        print("transformer           - Attention mechanisms")
        print("summarizer_resblock   - Residual blocks in summarizer")
        print("temporal_resblock     - Residual blocks in temporal")
        
        print("\n# Model outputs")
        print("output_plan           - Plan output nodes")
        print("output_lane_lines     - Lane lines output nodes")
        print("output_lane_lines_prob - Lane lines probability output nodes")
        print("output_road_edges     - Road edges output nodes")
        print("output_lead           - Lead vehicle output nodes")
        print("output_lead_prob      - Lead probability output nodes")
        print("output_desire_state   - Desire state output nodes")
        print("output_meta           - Meta information output nodes")
        print("output_desire_pred    - Desire prediction output nodes")
        print("output_pose           - Pose output nodes")
        print("output_wide_from_device - Wide from device output nodes")
        print("output_sim_pose       - Sim pose output nodes")
        print("output_road_transform - Road transform output nodes")
        print("output_lateral        - Lateral control output nodes") 
        print("output_longitudinal   - Longitudinal control output nodes")
        
        print("\n# Vision subsystems")
        print("vision_encoder        - Vision encoder network")
        print("vision_decoder        - Vision decoder network") 
        print("vision_features       - Vision features extraction")
        
        print("\n# Specific module types")
        print("gemm                  - All GEMM operations")
        print("conv                  - All Conv operations")
        print("relu                  - All ReLU activations")
        
        return

    # Validate component names
    valid_components = all_components
    for component in args.components:
        if component not in valid_components:
            print(f"Warning: Unknown component '{component}'. Use --list-components to see available options.")
    
    # Load models
    model_a = onnx.load(args.model_a)
    model_b = onnx.load(args.model_b)
    
    # Expected common trunk output
    target_trunk_output = "/supercombo/vision/Flatten_output_0"
    
    # Verify both models have the trunk output
    if not output_exists(model_a, target_trunk_output):
        raise ValueError(f"Model A does not contain the trunk output '{target_trunk_output}'")
    if not output_exists(model_b, target_trunk_output):
        raise ValueError(f"Model B does not contain the trunk output '{target_trunk_output}'")
    
    print(f"Both models contain the common trunk output '{target_trunk_output}'")
    print(f"Merging components: {', '.join(args.components)}")
    print(f"Using mixing ratio: {args.ratio:.2f} (Model B) / {1-args.ratio:.2f} (Model A)")
    
    # Merge models by replacing specific components
    merged_model = merge_models_by_components(
        model_a, model_b, target_trunk_output, args.components, args.ratio
    )
    
    # Optionally save weights for inspection
    if args.save_weights:
        save_model_weights_as_npz(model_a, os.path.splitext(args.model_a)[0] + "_weights.npz")
        save_model_weights_as_npz(model_b, os.path.splitext(args.model_b)[0] + "_weights.npz")
        save_model_weights_as_npz(merged_model, os.path.splitext(args.output)[0] + "_weights.npz")
    
    # Save the merged model
    onnx.save(merged_model, args.output)
    print(f"Merged model saved to: {args.output}")

if __name__ == "__main__":
    main()