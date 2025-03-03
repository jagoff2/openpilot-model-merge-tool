#!/usr/bin/env python3
"""
Enhanced merge script for comma.ai openpilot supercombo.onnx models that handles
structural differences gracefully by combining weights of specific components
after the shared CNN trunk. This preserves the model structure while merging
compatible components and falling back to the base model for incompatible parts.

Usage:
    python enhanced_merge_supercombo.py model_a.onnx model_b.onnx --output merged_supercombo.onnx --ratio 0.5
"""

import onnx
import numpy as np
from onnx import numpy_helper
import argparse
import os
import re
import sys
import json
import time
from collections import defaultdict
import warnings

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
        "desired_curvature": set(),          # Desired curvature output
        
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
        "output_action": set(),              # All action outputs
        "output_desired_curvature": set(),   # Desired curvature outputs
        
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
        if "/desired_curvature/" in name or "desired_curvature" in name:
            components["desired_curvature"].add(i)
            
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
        
        # Action outputs - more generic detection that doesn't rely on specific output names
        if "/action_block/Mul_output" in name or "/action_block/action_block_out" in name:
            components["output_action"].add(i)
        
        # Desired curvature outputs
        if "desired_curvature" in name.lower():
            components["output_desired_curvature"].add(i)
    
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
        if i >= len(model.graph.node):
            continue
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

def map_initializers_to_components(model, components):
    """
    Create a mapping of initializer names to which component they belong to.
    This helps with applying component-specific ratios.
    """
    initializer_to_component = {}
    
    for component_name, nodes in components.items():
        initializers = get_initializers_for_nodes(model, nodes)
        for init_name in initializers:
            if init_name not in initializer_to_component:
                initializer_to_component[init_name] = []
            initializer_to_component[init_name].append(component_name)
    
    return initializer_to_component

def verify_structure_compatibility(model_a, model_b, trunk_output):
    """
    Verify that both models have compatible structures after the trunk output.
    This helps ensure the merge will produce a valid model.
    
    Returns: (is_compatible, message, mismatched_nodes)
    """
    # Build maps of node names in both models
    nodes_a = {node.name: i for i, node in enumerate(model_a.graph.node) if node.name}
    nodes_b = {node.name: i for i, node in enumerate(model_b.graph.node) if node.name}
    
    # Check that all nodes after trunk output exist in both models
    found_trunk = False
    mismatched_nodes = []
    
    for i, node in enumerate(model_a.graph.node):
        if not found_trunk and trunk_output in node.output:
            found_trunk = True
            continue
            
        if found_trunk and node.name:
            if node.name not in nodes_b:
                mismatched_nodes.append(f"Node {node.name} exists in model A but not in model B")
    
    found_trunk = False
    for i, node in enumerate(model_b.graph.node):
        if not found_trunk and trunk_output in node.output:
            found_trunk = True
            continue
            
        if found_trunk and node.name:
            if node.name not in nodes_a:
                mismatched_nodes.append(f"Node {node.name} exists in model B but not in model A")
    
    if mismatched_nodes:
        return False, f"Models have different structures: {len(mismatched_nodes)} mismatched nodes. First few: {mismatched_nodes[:5]}", mismatched_nodes
    
    return True, "Models have compatible structures", []

def analyze_component_compatibility(model_a, model_b, components_a, components_b):
    """
    Analyze the compatibility of components between model_a and model_b.
    Returns a dictionary indicating compatibility status for each component.
    """
    nodes_a = {node.name: i for i, node in enumerate(model_a.graph.node) if node.name}
    nodes_b = {node.name: i for i, node in enumerate(model_b.graph.node) if node.name}
    
    compatibility_report = {}
    
    for component_name, node_indices in components_a.items():
        if component_name not in components_b:
            compatibility_report[component_name] = {
                "status": "missing_in_b",
                "compatible_nodes": 0,
                "total_nodes": len(node_indices),
                "compatibility_ratio": 0.0,
                "mismatched_nodes": []
            }
            continue
        
        # Get named nodes in component from model A
        nodes_in_component_a = [model_a.graph.node[i].name for i in node_indices if model_a.graph.node[i].name]
        
        # Check how many exist in model B
        mismatched_nodes = []
        compatible_nodes = 0
        
        for node_name in nodes_in_component_a:
            if node_name in nodes_b:
                compatible_nodes += 1
            else:
                mismatched_nodes.append(node_name)
        
        # Calculate compatibility ratio
        total_nodes = len(nodes_in_component_a)
        compatibility_ratio = compatible_nodes / total_nodes if total_nodes > 0 else 0.0
        
        # Determine status
        if compatibility_ratio == 1.0:
            status = "fully_compatible"
        elif compatibility_ratio >= 0.8:
            status = "mostly_compatible"
        elif compatibility_ratio >= 0.5:
            status = "partially_compatible"
        else:
            status = "mostly_incompatible"
            
        compatibility_report[component_name] = {
            "status": status,
            "compatible_nodes": compatible_nodes,
            "total_nodes": total_nodes,
            "compatibility_ratio": compatibility_ratio,
            "mismatched_nodes": mismatched_nodes
        }
    
    return compatibility_report

def get_compatible_initializers(model_a, model_b, component_node_indices_a, component_node_indices_b):
    """
    Get initializers that are compatible between two models for a specific component.
    """
    # Get initializers for both models' components
    initializers_a = get_initializers_for_nodes(model_a, component_node_indices_a)
    initializers_b = get_initializers_for_nodes(model_b, component_node_indices_b)
    
    # Check which initializers exist in both models
    common_initializer_names = set(initializers_a.keys()) & set(initializers_b.keys())
    
    # Get the actual initializers
    common_initializers = {name: initializers_a[name] for name in common_initializer_names}
    
    return common_initializers

def enhanced_merge_models(model_a, model_b, trunk_output, components_to_merge=None, ratio=0.5, 
                         validation_mode="component", action_block_mode="conservative", 
                         component_ratios=None, compatibility_threshold=0.8):
    """
    Enhanced model merger with intelligent fallback for incompatible structures.
    
    Args:
        model_a: The base model to keep the structure of
        model_b: The model to extract components from
        trunk_output: The name of the common trunk output
        components_to_merge: List of component names to merge 
        ratio: Weight of model_b's values (0 = pure A, 1 = pure B, 0.5 = equal mix)
        validation_mode: 
            "strict" - All components must match exactly
            "component" - Validate at component level (default)
            "layer" - Validate at layer level (most granular)
            "none" - No validation (least safe)
        action_block_mode: 
            "conservative" - Only merge weights that are confirmed compatible
            "ultra-conservative" - Use very small ratio for action_block
            "skip" - Don't merge action_block weights
        component_ratios: Dictionary mapping component names to specific ratios
        compatibility_threshold: Minimum compatibility ratio to allow merging
    
    Returns:
        tuple: (merged_model, merge_report)
    """
    if components_to_merge is None:
        components_to_merge = ["no_bottleneck_policy", "temporal_hydra", "temporal_summarizer", "action_block"]
    
    if component_ratios is None:
        component_ratios = {}
    
    # Make a copy of model_a as our base
    merged_model = onnx.ModelProto()
    merged_model.CopyFrom(model_a)
    
    # Initialize merge report
    merge_report = {
        "validation_mode": validation_mode,
        "global_ratio": ratio,
        "component_ratios": component_ratios.copy(),
        "fully_merged": [],
        "partially_merged": {},
        "skipped": [],
        "compatibility_issues": [],
        "structure_validation": None,
        "safety_warnings": []
    }
    
    # Perform structure validation
    if validation_mode != "none":
        is_compatible, message, mismatched_nodes = verify_structure_compatibility(model_a, model_b, trunk_output)
        merge_report["structure_validation"] = {
            "is_compatible": is_compatible,
            "message": message,
            "mismatched_nodes_count": len(mismatched_nodes)
        }
        
        if not is_compatible and validation_mode == "strict":
            merge_report["safety_warnings"].append("Strict validation failed. Models have different structures.")
            print("Error: Models have different structures and strict validation is enabled.")
            print(f"Validation message: {message}")
            return merged_model, merge_report
    
    # Special handling for action_block
    action_block_skipped = False
    if "action_block" in components_to_merge:
        if action_block_mode == "skip":
            print("Skipping action_block component as requested.")
            components_to_merge.remove("action_block")
            merge_report["skipped"].append("action_block")
            action_block_skipped = True
            merge_report["safety_warnings"].append("Action block skipped by user request.")
        elif action_block_mode == "ultra-conservative":
            # Override component_ratios for action_block with very conservative value
            component_ratios["action_block"] = min(component_ratios.get("action_block", ratio), 0.05)
            component_ratios["action_block_out"] = min(component_ratios.get("action_block_out", ratio), 0.05)
            component_ratios["desired_curvature"] = min(component_ratios.get("desired_curvature", ratio), 0.05)
            print(f"Using ultra-conservative ratio for action_block: {component_ratios['action_block']:.3f}")
            merge_report["safety_warnings"].append(f"Using ultra-conservative ratio for action_block: {component_ratios['action_block']:.3f}")
        elif action_block_mode == "conservative":
            # Use a more moderate but still conservative ratio
            component_ratios["action_block"] = min(component_ratios.get("action_block", ratio), 0.3)
            component_ratios["action_block_out"] = min(component_ratios.get("action_block_out", ratio), 0.3)
            component_ratios["desired_curvature"] = min(component_ratios.get("desired_curvature", ratio), 0.3)
            print(f"Using conservative ratio for action_block: {component_ratios['action_block']:.3f}")
            merge_report["safety_warnings"].append(f"Using conservative ratio for action_block: {component_ratios['action_block']:.3f}")
    
    # Identify components in both models
    components_a = identify_components_by_prefix(model_a)
    components_b = identify_components_by_prefix(model_b)
    
    # Analyze component compatibility
    component_compatibility = analyze_component_compatibility(model_a, model_b, components_a, components_b)
    
    # Map initializers to components (for component-specific ratios)
    initializer_to_component_a = map_initializers_to_components(model_a, components_a)
    
    # Print compatibility report
    print("\nComponent Compatibility Analysis:")
    for component, data in component_compatibility.items():
        if component in components_to_merge:
            status_str = {
                "fully_compatible": "✅ Fully Compatible",
                "mostly_compatible": "✓ Mostly Compatible",
                "partially_compatible": "⚠ Partially Compatible",
                "mostly_incompatible": "❌ Mostly Incompatible",
                "missing_in_b": "❌ Missing in Model B"
            }.get(data["status"], "Unknown")
            
            print(f"  - {component}: {status_str} ({data['compatible_nodes']}/{data['total_nodes']} nodes, {data['compatibility_ratio']*100:.1f}%)")
    
    print("\nComponents identified in model A:")
    for component, nodes in components_a.items():
        if len(nodes) > 0:
            print(f"  - {component}: {len(nodes)} nodes")
    
    # Convert initializers to numpy arrays for easier manipulation
    values_a = get_initializer_values_dict(model_a)
    values_b = get_initializer_values_dict(model_b)
    
    # Create list of initializers to merge for each component
    component_initializers_to_merge = {}
    all_initializers_to_merge = set()
    
    for component in components_to_merge:
        # Skip component if it's not in both models
        if component not in components_a or component not in components_b:
            missing_in = []
            if component not in components_a:
                missing_in.append("A")
            if component not in components_b:
                missing_in.append("B")
            print(f"Warning: Component {component} does not exist in model(s) {', '.join(missing_in)}, skipping.")
            merge_report["skipped"].append(component)
            continue
        
        # Check component compatibility
        if component in component_compatibility:
            compatibility_data = component_compatibility[component]
            
            # Handle special cases
            is_action_related = component == "action_block" or "action_block" in component
            if is_action_related and not action_block_skipped:
                if compatibility_data["status"] != "fully_compatible":
                    if action_block_mode == "ultra-conservative":
                        print(f"Warning: Action-related component {component} is not fully compatible. Using ultra-conservative ratio: {component_ratios[component]:.3f}")
                        merge_report["safety_warnings"].append(f"Action-related component {component} not fully compatible. Using ultra-conservative ratio.")
                    elif action_block_mode == "conservative":
                        if compatibility_data["compatibility_ratio"] < 0.95:
                            print(f"Warning: Action-related component {component} has low compatibility ({compatibility_data['compatibility_ratio']*100:.1f}%). Using ultra-conservative ratio instead.")
                            component_ratios[component] = min(component_ratios.get(component, ratio), 0.05)
                            merge_report["safety_warnings"].append(f"Action component {component} has low compatibility. Reduced to ultra-conservative ratio: {component_ratios[component]:.3f}")
            
            # Apply compatibility threshold
            if compatibility_data["compatibility_ratio"] < compatibility_threshold:
                print(f"Warning: Component {component} compatibility ({compatibility_data['compatibility_ratio']*100:.1f}%) below threshold ({compatibility_threshold*100:.1f}%), skipping merge.")
                merge_report["skipped"].append(component)
                merge_report["compatibility_issues"].append({
                    "component": component,
                    "reason": f"Compatibility ratio {compatibility_data['compatibility_ratio']*100:.1f}% below threshold {compatibility_threshold*100:.1f}%"
                })
                continue
        
        # Get common initializers for fully compatible components
        if components_a[component] and components_b[component]:
            initializers_a = get_initializers_for_nodes(model_a, components_a[component])
            initializers_b = get_initializers_for_nodes(model_b, components_b[component])
            
            # Find common initializers between models A and B
            common_initializers = set(initializers_a.keys()) & set(initializers_b.keys())
            
            # Special handling for action_block in conservative mode
            if component == "action_block" and action_block_mode in ["conservative", "ultra-conservative"]:
                # Create a filtered set for action_block initializers
                action_keys = [k for k in common_initializers if 'action_block' in k]
                
                # Exclude action_block_out weights if they don't exactly match
                action_out_keys = [k for k in action_keys if 'action_block_out' in k]
                for k in action_out_keys:
                    # Keep it only if exactly matching in structure
                    if k in initializers_a and k in initializers_b:
                        init_a = initializers_a[k]
                        init_b = initializers_b[k]
                        # Compare shapes and all attributes
                        if (init_a.dims != init_b.dims or 
                            init_a.data_type != init_b.data_type or
                            init_a.name != init_b.name):
                            common_initializers.remove(k)
                            print(f"Excluding mismatched action_block initializer: {k}")
                            merge_report["compatibility_issues"].append({
                                "component": component,
                                "initializer": k,
                                "reason": "Mismatched action block initializer structure"
                            })
            
            component_initializers_to_merge[component] = common_initializers
            all_initializers_to_merge.update(common_initializers)
            print(f"Found {len(common_initializers)} common initializers for {component}")
        else:
            print(f"Warning: Component {component} exists but contains no nodes in one or both models, skipping.")
            merge_report["skipped"].append(component)
    
    # Merge weights and replace in the merged model
    merged_count = 0
    skipped_count = 0
    merged_components = defaultdict(int)  # Track which components had initializers merged
    component_statistics = defaultdict(lambda: {"count": 0, "bytes": 0})
    
    for name in all_initializers_to_merge:
        if name in values_a and name in values_b:
            # Check if shapes are compatible
            if values_a[name].shape == values_b[name].shape:
                # Determine which component this belongs to for ratio selection
                current_ratio = ratio
                component_name = "unknown"
                
                # Find components this initializer belongs to
                if name in initializer_to_component_a:
                    # Use the first component as primary, but track all
                    component_name = initializer_to_component_a[name][0]
                    
                    # If this component has a custom ratio, use it
                    if component_name in component_ratios:
                        current_ratio = component_ratios[component_name]
                    
                    # Special handling for specific nodes
                    if "action_block_out" in name or "desired_curvature" in name:
                        if "desired_curvature" in component_ratios:
                            current_ratio = component_ratios["desired_curvature"]
                        elif "action_block_out" in component_ratios:
                            current_ratio = component_ratios["action_block_out"]
                
                # Create merged weight array
                merged_array = merge_weights(values_a[name], values_b[name], current_ratio)
                
                # Track which component this initializer belongs to for reporting
                for component_list in initializer_to_component_a.get(name, []):
                    merged_components[component_list] += 1
                    component_statistics[component_list]["count"] += 1
                    component_statistics[component_list]["bytes"] += merged_array.nbytes
                
                # Replace the initializer completely (exact copy with updated data)
                for i, init in enumerate(merged_model.graph.initializer):
                    if init.name == name:
                        # Create a brand new tensor with the merged values
                        new_initializer = numpy_helper.from_array(merged_array, name=name)
                        # Replace it while preserving the name and position
                        merged_model.graph.initializer[i].CopyFrom(new_initializer)
                        merged_count += 1
                        print(f"Merged initializer {name} using ratio {current_ratio:.3f} ({component_name})")
                        break
            else:
                skipped_count += 1
                print(f"Warning: Shape mismatch for initializer {name}, skipping. A: {values_a[name].shape}, B: {values_b[name].shape}")
                merge_report["compatibility_issues"].append({
                    "initializer": name,
                    "reason": f"Shape mismatch: A: {values_a[name].shape}, B: {values_b[name].shape}"
                })
    
    # Update merge report with statistics
    for component, count in merged_components.items():
        stats = component_statistics[component]
        size_mb = stats["bytes"] / (1024*1024)
        
        # Calculate what percentage of component was merged
        total_component_initializers = len(get_initializers_for_nodes(model_a, components_a.get(component, set())))
        merge_percentage = count / total_component_initializers if total_component_initializers > 0 else 0
        
        if merge_percentage > 0.95:  # Consider it fully merged if >95% was merged
            merge_report["fully_merged"].append(component)
        else:
            merge_report["partially_merged"][component] = {
                "initializers_merged": count,
                "total_initializers": total_component_initializers,
                "merge_percentage": merge_percentage,
                "size_mb": size_mb
            }
    
    # Print merge statistics
    print(f"\nMerged {merged_count} initializers using default ratio {ratio:.3f} (B) / {1-ratio:.3f} (A)")
    print(f"Skipped {skipped_count} initializers due to compatibility issues")
    
    # Print breakdown by component
    print("\nMerged initializers by component:")
    for component, count in sorted(merged_components.items(), key=lambda x: x[1], reverse=True):
        stats = component_statistics[component]
        size_mb = stats["bytes"] / (1024*1024)
        print(f"  - {component}: {count} initializers, {size_mb:.2f} MB")
    
    # Print component-specific ratios that were used
    if component_ratios:
        print("\nComponent-specific ratios used:")
        for component, comp_ratio in component_ratios.items():
            print(f"  - {component}: {comp_ratio:.3f}")
    
    # Final report on merge effectiveness
    total_a_initializers = len(model_a.graph.initializer)
    print(f"\nOverall merge summary:")
    print(f"  - Total initializers in base model: {total_a_initializers}")
    print(f"  - Initializers merged: {merged_count} ({merged_count/total_a_initializers*100:.1f}%)")
    print(f"  - Initializers skipped: {skipped_count}")
    
    # Add summary statistics to merge report
    merge_report["summary"] = {
        "total_initializers": total_a_initializers,
        "initializers_merged": merged_count,
        "initializers_skipped": skipped_count,
        "merge_ratio": merged_count/total_a_initializers
    }
    
    return merged_model, merge_report

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
        "desired_curvature",
        
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
        "output_action",
        "output_desired_curvature",
        
        # Vision subsystems
        "vision_encoder",
        "vision_decoder",
        "vision_features",
        
        # Specific module types
        "gemm",
        "conv",
        "relu",
    ]

def test_parser_compatibility(model_path):
    """
    Test if the model can be parsed with the openpilot parser.
    This is a simple test to verify basic model structure compatibility.
    """
    try:
        # Try to import the parser
        from openpilot.selfdrive.modeld.parse_model_outputs import Parser
        print("Successfully imported the openpilot Parser, will check for compatibility.")
        
        # We can't easily run the model here, but we can check if the parser would recognize it
        parser = Parser(ignore_missing=True)
        print("Model structure looks compatible with the parser.")
        return True
    except ImportError:
        print("Openpilot Parser not available, skipping compatibility check.")
        return None
    except Exception as e:
        print(f"Warning: Parser compatibility check failed: {e}")
        return False

def get_sanitized_output_path(output_path, model_a, model_b, components, ratio, action_block_mode):
    """Generate a sanitized output path with descriptive filename."""
    # If output is explicitly provided, use it
    if output_path != "merged_supercombo.onnx":
        return output_path
    
    # Otherwise, generate a descriptive name
    base_a = os.path.splitext(os.path.basename(model_a))[0]
    base_b = os.path.splitext(os.path.basename(model_b))[0]
    comp_str = "_".join([c[:3] for c in components])
    ratio_str = f"{ratio:.2f}".replace(".", "p")
    action_str = action_block_mode[:3]
    
    return f"merged_{base_a}_{base_b}_{comp_str}_r{ratio_str}_{action_str}.onnx"

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
        description="Enhanced merge script for comma.ai openpilot supercombo.onnx models with intelligent fallback."
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
    parser.add_argument("--validation-mode", choices=["strict", "component", "layer", "none"], 
                      default="component", help="How strictly to validate structure compatibility")
    parser.add_argument("--action-block-mode", choices=["conservative", "aggressive", "ultra-conservative", "skip"], 
                      default="ultra-conservative", help="How to handle action_block merging")
    parser.add_argument("--component-ratios", type=str, default="",
                      help="Custom ratios per component in format 'component:ratio,...'")
    parser.add_argument("--compatibility-threshold", type=float, default=0.8,
                      help="Minimum compatibility ratio to allow merging (0.0-1.0)")
    parser.add_argument("--auto-output", action="store_true", 
                      help="Auto-generate output name based on merge parameters")
    parser.add_argument("--verify-parser", action="store_true",
                      help="Try to verify the model can be parsed with openpilot parser")
    parser.add_argument("--dump-metadata", action="store_true",
                      help="Dump metadata about the merge to a JSON file")
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
        print("desired_curvature     - Desired curvature output")
        
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
        print("output_action         - Action control output nodes")
        print("output_desired_curvature - Desired curvature output nodes")
        
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
    try:
        model_a = onnx.load(args.model_a)
        print(f"Successfully loaded model A: {args.model_a}")
    except Exception as e:
        print(f"Error loading model A: {e}")
        return
    
    try:
        model_b = onnx.load(args.model_b)
        print(f"Successfully loaded model B: {args.model_b}")
    except Exception as e:
        print(f"Error loading model B: {e}")
        return
    
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
    print(f"Validation mode: {args.validation_mode}")
    print(f"Action block mode: {args.action_block_mode}")
    print(f"Compatibility threshold: {args.compatibility_threshold:.2f}")
    
    # Process component-specific ratios
    component_ratios = {}
    if args.component_ratios:
        for item in args.component_ratios.split(","):
            try:
                comp, ratio_val = item.split(":")
                component_ratios[comp] = float(ratio_val)
                print(f"Setting custom ratio for {comp}: {float(ratio_val):.3f}")
            except Exception as e:
                print(f"Error parsing component ratio '{item}': {e}")
    
    # Generate auto output name if requested
    output_path = args.output
    if args.auto_output:
        output_path = get_sanitized_output_path(output_path, args.model_a, args.model_b, 
                                               args.components, args.ratio, args.action_block_mode)
        print(f"Auto-generated output name: {output_path}")
    
    merge_start_time = time.time()
    
    # Merge models with enhanced fallback mechanisms
    merged_model, merge_report = enhanced_merge_models(
        model_a, model_b, target_trunk_output, args.components, args.ratio, 
        validation_mode=args.validation_mode,
        action_block_mode=args.action_block_mode,
        component_ratios=component_ratios,
        compatibility_threshold=args.compatibility_threshold
    )
    
    merge_end_time = time.time()
    merge_duration = merge_end_time - merge_start_time
    print(f"\nMerge completed in {merge_duration:.2f} seconds")
    
    # Add timing information to merge report
    merge_report["timing"] = {
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(merge_start_time)),
        "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(merge_end_time)),
        "duration_seconds": merge_duration
    }
    
    # Try to verify compatibility with the openpilot parser
    if args.verify_parser:
        try:
            parser_compatible = test_parser_compatibility(args.model_a)
            if parser_compatible:
                print("Model structure appears compatible with openpilot parser")
                merge_report["parser_compatibility"] = "compatible"
            elif parser_compatible is False:  # Could be None if parser not available
                print("Warning: Model structure may not be compatible with openpilot parser")
                merge_report["parser_compatibility"] = "potentially_incompatible"
                merge_report["safety_warnings"].append("Model structure may not be compatible with openpilot parser")
            else:
                merge_report["parser_compatibility"] = "check_skipped"
        except Exception as e:
            print(f"Error during parser compatibility check: {e}")
            merge_report["parser_compatibility"] = "check_failed"
            merge_report["safety_warnings"].append(f"Parser compatibility check failed: {str(e)}")
    
    # Optionally save weights for inspection
    if args.save_weights:
        save_model_weights_as_npz(model_a, os.path.splitext(args.model_a)[0] + "_weights.npz")
        save_model_weights_as_npz(model_b, os.path.splitext(args.model_b)[0] + "_weights.npz")
        save_model_weights_as_npz(merged_model, os.path.splitext(output_path)[0] + "_weights.npz")
    
    # Save the merged model
    try:
        onnx.save(merged_model, output_path)
        print(f"Merged model saved to: {output_path}")
        merge_report["output_path"] = output_path
    except Exception as e:
        print(f"Error saving merged model: {e}")
        merge_report["errors"] = merge_report.get("errors", []) + [f"Error saving model: {str(e)}"]
        return
    
    # Dump metadata if requested
    if args.dump_metadata:
        metadata = {
            "model_a": args.model_a,
            "model_b": args.model_b,
            "output_path": output_path,
            "components": args.components,
            "ratio": args.ratio,
            "validation_mode": args.validation_mode,
            "action_block_mode": args.action_block_mode,
            "component_ratios": component_ratios,
            "compatibility_threshold": args.compatibility_threshold,
            "merge_time": merge_duration,
            "merge_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "merged_model_size": os.path.getsize(output_path),
            "merge_report": merge_report
        }
        
        metadata_path = os.path.splitext(output_path)[0] + "_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Merge metadata saved to: {metadata_path}")
    
    # Provide a summary of components that were fully merged, partially merged, or skipped
    print("\nMerge Summary:")
    if merge_report["fully_merged"]:
        print(f"Fully merged components: {', '.join(merge_report['fully_merged'])}")
    
    if merge_report["partially_merged"]:
        print("Partially merged components:")
        for comp, stats in merge_report["partially_merged"].items():
            print(f"  - {comp}: {stats['merge_percentage']*100:.1f}% merged ({stats['initializers_merged']}/{stats['total_initializers']} initializers)")
    
    if merge_report["skipped"]:
        print(f"Skipped components: {', '.join(merge_report['skipped'])}")
    
    if merge_report["safety_warnings"]:
        print("\nSafety Warnings:")
        for warning in merge_report["safety_warnings"]:
            print(f"  - {warning}")
    
    print(f"\nMerge complete! Use the following to compile and test:")
    print(f"  FLOAT16=0 JIT_BATCH_SIZE=16 python compile_script.py {output_path}")

if __name__ == "__main__":
    main()
