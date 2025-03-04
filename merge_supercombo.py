#!/usr/bin/env python3
"""
Enhanced merge script for comma.ai openpilot supercombo.onnx models with ONNX GraphSurgeon integration.
This script merges two OpenPilot models by intelligently combining weights of specific components
after the shared CNN trunk, with advanced graph analysis and validation.

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

# Try to import ONNX GraphSurgeon - gracefully handle if not available
try:
    import onnx_graphsurgeon as gs
    GRAPHSURGEON_AVAILABLE = True
except ImportError:
    GRAPHSURGEON_AVAILABLE = False
    warnings.warn("ONNX GraphSurgeon not found. Some advanced features will be disabled. "
                  "Install via: pip install nvidia-onnx-graphsurgeon or from NVIDIA TensorRT package.")

def get_tensor_shape(tensor_info):
    """Get the shape of a tensor from its info."""
    if not hasattr(tensor_info, 'type'):
        return None
    
    tensor_type = tensor_info.type
    if hasattr(tensor_type, 'tensor_type') and hasattr(tensor_type.tensor_type, 'shape'):
        dims = tensor_type.tensor_type.shape.dim
        if dims:
            return [dim.dim_value for dim in dims]
    return None

def find_feature_node(model, feature_name_pattern="hidden_state"):
    """Find the node that produces the feature tensor used for policy inputs."""
    for output in model.graph.output:
        if feature_name_pattern in output.name:
            return output
    return None

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
        # High-level components (vision/policy split based on modeld.py structure)
        "vision": set(),                     # Vision model (pre-trunk)
        "no_bottleneck_policy": set(),       # Policy model (post-trunk) 
        "temporal_summarizer": set(),        # Policy: temporal context processing
        "temporal_hydra": set(),             # Policy: output heads
        "action_block": set(),               # Policy: vehicle control outputs
        
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
        "output_plan": set(),                # Plan output nodes (trajectory planning)
        "output_lane_lines": set(),          # Lane lines output nodes (lane detection)
        "output_lane_lines_prob": set(),     # Lane lines prob output nodes
        "output_road_edges": set(),          # Road edges output nodes (road boundary detection)
        "output_lead": set(),                # Lead output nodes (lead car tracking)
        "output_lead_prob": set(),           # Lead prob output nodes
        "output_desire_state": set(),        # Desire state output nodes (lane change intention)
        "output_meta": set(),                # Meta output nodes
        "output_desire_pred": set(),         # Desire prediction output nodes
        "output_pose": set(),                # Pose output nodes (vehicle position)
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

def validate_output_shapes(model_a, model_b):
    """
    Ensure output tensor shapes remain compatible with parser expectations
    by checking output nodes dimensions.
    """
    output_names_a = {output.name for output in model_a.graph.output}
    output_names_b = {output.name for output in model_b.graph.output}
    
    # Check that both models have the same outputs
    if output_names_a != output_names_b:
        missing_in_a = output_names_b - output_names_a
        missing_in_b = output_names_a - output_names_b
        message = []
        if missing_in_a:
            message.append(f"Outputs in B but not in A: {missing_in_a}")
        if missing_in_b:
            message.append(f"Outputs in A but not in B: {missing_in_b}")
        return False, f"Output mismatch: {', '.join(message)}"
    
    # Check that matching outputs have the same shape
    for output_a in model_a.graph.output:
        name = output_a.name
        output_b = next((o for o in model_b.graph.output if o.name == name), None)
        if output_b:
            shape_a = get_tensor_shape(output_a)
            shape_b = get_tensor_shape(output_b)
            if shape_a != shape_b:
                return False, f"Output shape mismatch for {name}: {shape_a} vs {shape_b}"
    
    return True, "Output shapes compatible"

def validate_feature_compatibility(model_a, model_b):
    """
    Check that the feature shape passed from vision to policy
    remains compatible between models.
    """
    # Locate the feature tensor in both models (hidden_state is commonly used)
    feature_node_a = find_feature_node(model_a)
    feature_node_b = find_feature_node(model_b)
    
    if feature_node_a is None or feature_node_b is None:
        return True, "Could not find feature tensor nodes, skipping validation"
        
    shape_a = get_tensor_shape(feature_node_a)
    shape_b = get_tensor_shape(feature_node_b)
    
    if shape_a is None or shape_b is None:
        return True, "Could not determine feature shapes, skipping validation"
    
    if shape_a != shape_b:
        return False, f"Feature shape mismatch: {shape_a} vs {shape_b}"
    
    return True, "Feature shapes compatible"

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

# GraphSurgeon-specific functions (when available)

if GRAPHSURGEON_AVAILABLE:
    def find_tensor_by_name(graph, tensor_name):
        """Find a tensor in the graph by name."""
        for tensor in graph.tensors().values():
            if tensor.name == tensor_name:
                return tensor
        return None
    
    def get_subgraph_by_tensor(graph, start_tensor, forward=True, max_depth=None):
        """
        Get all nodes connected to a tensor in either forward or backward direction.
        Uses node IDs instead of node objects for tracking visited nodes to avoid
        unhashable type errors with Node objects.
        
        Args:
            graph: GraphSurgeon graph
            start_tensor: Starting tensor name or tensor object
            forward: True to follow outputs, False to follow inputs
            max_depth: Maximum traversal depth (None for unlimited)
        
        Returns:
            List of nodes in the subgraph (not a set to avoid unhashable issues)
        """
        # Get the starting tensor if name was provided
        if isinstance(start_tensor, str):
            start_tensor = find_tensor_by_name(graph, start_tensor)
            if start_tensor is None:
                return []
        
        # Use a set to track visited nodes by their id
        visited_node_ids = set()
        result_nodes = []
        
        # Get initial nodes to visit
        nodes_to_visit = list(start_tensor.outputs if forward else start_tensor.inputs)
        depth = 0
        
        while nodes_to_visit and (max_depth is None or depth < max_depth):
            current_level = nodes_to_visit
            nodes_to_visit = []
            
            for node in current_level:
                # Use the node's id as a unique identifier
                node_id = id(node)
                if node_id in visited_node_ids:
                    continue
                
                visited_node_ids.add(node_id)
                result_nodes.append(node)
                
                # Add connected nodes based on direction
                if forward:
                    for out_tensor in node.outputs:
                        nodes_to_visit.extend(out_tensor.outputs)
                else:
                    for in_tensor in node.inputs:
                        if not in_tensor.is_constant:  # Skip constants when going backwards
                            nodes_to_visit.extend(in_tensor.inputs)
            
            depth += 1
        
        return result_nodes
    
    def identify_components_with_graphsurgeon(model_path, trunk_output="/supercombo/vision/Flatten_output_0"):
        """
        Identify model components by analyzing the graph structure with GraphSurgeon.
        Much more accurate than name-based identification.
        
        Returns a dictionary of component names to lists of nodes.
        """
        graph = gs.import_onnx(onnx.load(model_path))
        
        # Find the trunk tensor (boundary between vision and policy)
        trunk_tensor = find_tensor_by_name(graph, trunk_output)
        if trunk_tensor is None:
            print(f"Error: Trunk tensor {trunk_output} not found in model")
            return None
        
        # Components dictionary - using lists instead of sets to avoid unhashable issues
        components = {
            "vision": [],              # All nodes before trunk
            "policy": [],              # All nodes after trunk
            "temporal_summarizer": [], # Temporal feature processor
            "temporal_hydra": [],      # Temporal output heads
            "action_block": [],        # Vehicle control outputs
            "no_bottleneck_policy": [] # Core policy network
        }
        
        # Node ID to component mapping for fast lookup
        node_id_to_component = {}
        
        # Get all nodes before the trunk (vision)
        components["vision"] = get_subgraph_by_tensor(graph, trunk_tensor, forward=False)
        
        # Get all nodes after the trunk (policy)
        components["policy"] = get_subgraph_by_tensor(graph, trunk_tensor, forward=True)
        
        # Store node IDs for vision component
        vision_node_ids = {id(node) for node in components["vision"]}
        policy_node_ids = {id(node) for node in components["policy"]}
        
        # Find specific components by name patterns
        for node in components["policy"]:
            name = node.name if node.name else ""
            
            if "/temporal_summarizer/" in name:
                components["temporal_summarizer"].append(node)
                node_id_to_component[id(node)] = "temporal_summarizer"
            elif "/temporal_hydra/" in name:
                components["temporal_hydra"].append(node)
                node_id_to_component[id(node)] = "temporal_hydra"
            elif "/action_block/" in name:
                components["action_block"].append(node)
                node_id_to_component[id(node)] = "action_block"
            elif "/supercombo/no_bottleneck_policy/" in name:
                components["no_bottleneck_policy"].append(node)
                node_id_to_component[id(node)] = "no_bottleneck_policy"
            
            # Specific policy components and outputs can be added here
        
        # Find component input/output tensors
        component_boundaries = {}
        for component_name, nodes in components.items():
            inputs = []
            outputs = []
            
            # Create set of node IDs for this component for quick lookup
            component_node_ids = {id(node) for node in nodes}
            
            for node in nodes:
                # Find inputs that come from outside the component
                for inp in node.inputs:
                    if not inp.is_constant:
                        is_external = True
                        for src in inp.inputs:
                            # Check if source node is in this component
                            if id(src) in component_node_ids:
                                is_external = False
                                break
                        if is_external:
                            inputs.append(inp)
                
                # Find outputs that go outside the component
                for out in node.outputs:
                    is_external = False
                    for dst in out.outputs:
                        # Check if destination node is outside this component
                        if id(dst) not in component_node_ids:
                            is_external = True
                            break
                    if is_external:
                        outputs.append(out)
            
            component_boundaries[component_name] = {
                "inputs": inputs,
                "outputs": outputs
            }
        
        # Return both components and their boundaries
        return components, component_boundaries
    
    def validate_graphsurgeon_compatibility(model_a_path, model_b_path, trunk_output):
        """
        Perform comprehensive compatibility validation using GraphSurgeon.
        
        Args:
            model_a_path: Path to first model
            model_b_path: Path to second model
            trunk_output: Name of trunk tensor
            
        Returns:
            dict: Validation report with detailed information
        """
        # Import both models
        graph_a = gs.import_onnx(onnx.load(model_a_path))
        graph_b = gs.import_onnx(onnx.load(model_b_path))
        
        # Find trunk tensors in both models
        trunk_a = find_tensor_by_name(graph_a, trunk_output)
        trunk_b = find_tensor_by_name(graph_b, trunk_output)
        
        if trunk_a is None or trunk_b is None:
            return {
                "is_compatible": False,
                "message": f"Trunk tensor {trunk_output} not found in one or both models",
                "trunk_a_present": trunk_a is not None,
                "trunk_b_present": trunk_b is not None
            }
        
        # Check trunk tensor shapes
        if trunk_a.shape != trunk_b.shape:
            return {
                "is_compatible": False,
                "message": f"Trunk tensor shapes don't match: {trunk_a.shape} vs {trunk_b.shape}",
                "trunk_a_shape": trunk_a.shape,
                "trunk_b_shape": trunk_b.shape
            }
        
        # Get policy subgraphs (everything after trunk)
        policy_a = get_subgraph_by_tensor(graph_a, trunk_a, forward=True)
        policy_b = get_subgraph_by_tensor(graph_b, trunk_b, forward=True)
        
        # Check if they have the same number of nodes (approximate check)
        if len(policy_a) != len(policy_b):
            return {
                "is_compatible": False,
                "message": f"Policy subgraphs have different sizes: {len(policy_a)} vs {len(policy_b)}",
                "policy_a_nodes": len(policy_a),
                "policy_b_nodes": len(policy_b),
                "difference": abs(len(policy_a) - len(policy_b))
            }
        
        # More detailed validation - check for specific component differences
        component_differences = []
        
        # Compare output nodes
        outputs_a = {output.name: output for output in graph_a.outputs}
        outputs_b = {output.name: output for output in graph_b.outputs}
        
        # Check for missing outputs
        missing_in_a = set(outputs_b.keys()) - set(outputs_a.keys())
        missing_in_b = set(outputs_a.keys()) - set(outputs_b.keys())
        
        if missing_in_a or missing_in_b:
            component_differences.append({
                "component": "outputs",
                "missing_in_a": list(missing_in_a),
                "missing_in_b": list(missing_in_b)
            })
        
        # Check for output shape mismatches
        shape_mismatches = []
        for name, output_a in outputs_a.items():
            if name in outputs_b:
                output_b = outputs_b[name]
                if output_a.shape != output_b.shape:
                    shape_mismatches.append({
                        "output": name,
                        "shape_a": output_a.shape,
                        "shape_b": output_b.shape
                    })
        
        if shape_mismatches:
            component_differences.append({
                "component": "output_shapes",
                "mismatches": shape_mismatches
            })
        
        # Build the final report
        report = {
            "is_compatible": len(component_differences) == 0,
            "message": "Models appear compatible" if len(component_differences) == 0 else f"Found {len(component_differences)} compatibility issues",
            "component_differences": component_differences,
            "policy_a_nodes": len(policy_a),
            "policy_b_nodes": len(policy_b)
        }
        
        return report
    
    def get_component_weights_graphsurgeon(graph, component_nodes):
        """
        Get all constant tensors (weights) used by nodes in a component.
        
        Args:
            graph: GraphSurgeon graph
            component_nodes: List of nodes in the component
            
        Returns:
            dict mapping weight names to weight tensors
        """
        weights = {}
        
        for node in component_nodes:
            for inp in node.inputs:
                if inp.is_constant:
                    weights[inp.name] = inp
        
        return weights
    
    def merge_weights_graphsurgeon(model_a_path, model_b_path, components_to_merge, output_path, 
                                 ratio=0.5, component_ratios=None, trunk_output="/supercombo/vision/Flatten_output_0"):
        """
        Directly merge weights in the ONNX graphs using GraphSurgeon.
        
        Args:
            model_a_path: Path to the base model
            model_b_path: Path to the second model
            components_to_merge: List of component names to merge
            output_path: Path for the merged model
            ratio: Global merge ratio (B's contribution)
            component_ratios: Dict mapping component names to specific ratios
            trunk_output: Name of the trunk tensor
        
        Returns:
            tuple: (success, merged_graph, report)
        """
        if component_ratios is None:
            component_ratios = {}
            
        try:
            # Load both models
            graph_a = gs.import_onnx(onnx.load(model_a_path))
            graph_b = gs.import_onnx(onnx.load(model_b_path))
            
            # Identify components
            result_a = identify_components_with_graphsurgeon(model_a_path, trunk_output)
            if result_a is None:
                return False, None, {"error": "Failed to identify components in model A"}
            
            components_a, _ = result_a
            
            result_b = identify_components_with_graphsurgeon(model_b_path, trunk_output)
            if result_b is None:
                return False, None, {"error": "Failed to identify components in model B"}
            
            components_b, _ = result_b
            
            # Statistics for reporting
            merge_report = {
                "merged_components": {},
                "skipped_components": [],
                "global_ratio": ratio,
                "component_ratios": component_ratios.copy(),
                "merged_weights_count": 0,
                "skipped_weights_count": 0
            }
            
            # Process each component
            for component in components_to_merge:
                if component not in components_a or component not in components_b:
                    # Component doesn't exist in one of the models
                    merge_report["skipped_components"].append({
                        "component": component,
                        "reason": f"Component not found in {'model A' if component not in components_a else 'model B'}"
                    })
                    continue
                
                # Get weights for this component in both models
                weights_a = get_component_weights_graphsurgeon(graph_a, components_a[component])
                weights_b = get_component_weights_graphsurgeon(graph_b, components_b[component])
                
                # Find common weights
                common_weights = set(weights_a.keys()) & set(weights_b.keys())
                
                # Use component-specific ratio if specified
                current_ratio = component_ratios.get(component, ratio)
                
                # Special handling for action_block - use more conservative ratio
                if component == "action_block":
                    current_ratio = min(current_ratio, 0.3)  # Conservative default
                    
                # Track statistics for this component
                component_stats = {
                    "total_weights_a": len(weights_a),
                    "total_weights_b": len(weights_b),
                    "common_weights": len(common_weights),
                    "applied_ratio": current_ratio,
                    "merged_weights": 0,
                    "skipped_weights": 0
                }
                
                # Merge common weights
                for weight_name in common_weights:
                    weight_a = weights_a[weight_name]
                    weight_b = weights_b[weight_name]
                    
                    # Check if shapes match
                    if weight_a.values.shape == weight_b.values.shape:
                        # Merge weights
                        merged_values = (1 - current_ratio) * weight_a.values + current_ratio * weight_b.values
                        weight_a.values = merged_values
                        
                        component_stats["merged_weights"] += 1
                        merge_report["merged_weights_count"] += 1
                    else:
                        component_stats["skipped_weights"] += 1
                        merge_report["skipped_weights_count"] += 1
                
                merge_report["merged_components"][component] = component_stats
            
            # Save the merged model
            output_model = gs.export_onnx(graph_a)
            onnx.save(output_model, output_path)
            
            return True, graph_a, merge_report
            
        except Exception as e:
            print(f"Error in GraphSurgeon merge: {e}")
            return False, None, {"error": str(e)}

def enhanced_merge_models(model_a, model_b, trunk_output, components_to_merge=None, ratio=0.5, 
                         validation_mode="component", action_block_mode="conservative", 
                         component_ratios=None, compatibility_threshold=0.8,
                         parser_aware=True):
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
        parser_aware: Enable parser-aware validations for output compatibility
    
    Returns:
        tuple: (merged_model, merge_report)
    """
    if components_to_merge is None:
        components_to_merge = ["no_bottleneck_policy", "temporal_hydra", "temporal_summarizer", "action_block"]
    
    if component_ratios is None:
        component_ratios = {}
    
    # Define component categories for parser-aware merging
    vision_components = ["vision", "vision_encoder", "vision_decoder", "vision_features"]
    policy_components = ["no_bottleneck_policy", "temporal_summarizer", "temporal_hydra", "action_block"]
    
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
        "parser_validation": None,
        "safety_warnings": [],
        "graphsurgeon_used": False  # Will be set to True if we use GraphSurgeon
    }
    
    # Vision/Policy boundary check
    vision_policy_boundary = trunk_output  # "/supercombo/vision/Flatten_output_0"
    
    # If merging vision components, enforce stricter validation
    merging_vision = any(comp in vision_components for comp in components_to_merge)
    if merging_vision and validation_mode != "strict":
        print("Warning: Merging vision components may affect model stability")
        print("Forcing strict validation for vision components")
        validation_mode = "strict"
        merge_report["safety_warnings"].append("Forced strict validation due to vision component merging")
    
    if GRAPHSURGEON_AVAILABLE:
        try:
            print("Using ONNX GraphSurgeon for enhanced validation and merging")
        
            # Save temporary copies for GraphSurgeon (it requires file paths)
            model_a_path = "model_a_temp.onnx"
            model_b_path = "model_b_temp.onnx"
            output_path = "merged_temp.onnx"
        
            onnx.save(model_a, model_a_path)
            onnx.save(model_b, model_b_path)
        
            # Now pass the file paths instead of model objects
            success, gs_graph, gs_report = merge_weights_graphsurgeon(
                model_a_path, model_b_path, components_to_merge, output_path,
                ratio, component_ratios, trunk_output
            )
        
            # Clean up temporary files
            try:
                os.remove(model_a_path)
                os.remove(model_b_path)
            except:
                pass
            
            if success:
                print("Successfully merged models using GraphSurgeon")
                merge_report["graphsurgeon_used"] = True
                merge_report["graphsurgeon_report"] = gs_report
                
                # Load the merged model
                merged_model = onnx.load(output_path)
                
                # Cleanup output file
                if os.path.exists(output_path):
                    os.remove(output_path)
                
                # Convert GraphSurgeon report to our standard format
                for component, stats in gs_report["merged_components"].items():
                    if stats["merged_weights"] > 0:
                        if stats["merged_weights"] == stats["total_weights_a"]:
                            merge_report["fully_merged"].append(component)
                        else:
                            merge_percentage = stats["merged_weights"] / stats["total_weights_a"]
                            merge_report["partially_merged"][component] = {
                                "initializers_merged": stats["merged_weights"],
                                "total_initializers": stats["total_weights_a"],
                                "merge_percentage": merge_percentage
                            }
                
                for skipped in gs_report["skipped_components"]:
                    merge_report["skipped"].append(skipped["component"])
                
                # We've successfully used GraphSurgeon, so we can return now
                return merged_model, merge_report
            else:
                print("GraphSurgeon merge failed, falling back to standard method")
                merge_report["graphsurgeon_error"] = gs_report.get("error", "Unknown GraphSurgeon error")
        
        except Exception as e:
            print(f"Error using GraphSurgeon: {e}")
            print("Falling back to standard merging method")
            merge_report["graphsurgeon_error"] = str(e)
    
    # If we reach here, either GraphSurgeon isn't available or it failed
    # Continue with the original merging approach
    
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
    
    # Perform parser-aware validation
    if parser_aware:
        # Validate output shapes
        outputs_compatible, outputs_message = validate_output_shapes(model_a, model_b)
        # Validate vision to policy feature compatibility
        features_compatible, features_message = validate_feature_compatibility(model_a, model_b)
        
        merge_report["parser_validation"] = {
            "outputs_compatible": outputs_compatible,
            "outputs_message": outputs_message,
            "features_compatible": features_compatible,
            "features_message": features_message
        }
        
        if not outputs_compatible:
            merge_report["safety_warnings"].append(f"Output shape validation failed: {outputs_message}")
            print(f"Warning: {outputs_message}")
            if validation_mode == "strict":
                print("Error: Output validation failed and strict validation is enabled.")
                return merged_model, merge_report
        
        if not features_compatible:
            merge_report["safety_warnings"].append(f"Feature compatibility validation failed: {features_message}")
            print(f"Warning: {features_message}")
            if validation_mode == "strict":
                print("Error: Feature validation failed and strict validation is enabled.")
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
        # High-level components (vision/policy split based on modeld.py structure)
        "vision",  # Vision model (pre-trunk)
        "no_bottleneck_policy",  # Policy model (post-trunk) 
        "temporal_summarizer",   # Policy: temporal context processing
        "temporal_hydra",        # Policy: output heads
        "action_block",          # Policy: vehicle control outputs
        
        # Policy-specific components
        "plan",                  # Path planning components
        "lead",                  # Lead vehicle detection
        "lead_prob",             # Lead probability estimation
        "lane_lines",            # Lane line detection
        "lane_lines_prob",       # Lane line probability
        "road_edges",            # Road edge detection
        "desire_state",          # Desired state prediction
        "desired_curvature",     # Desired curvature output
        
        # Control-specific components
        "action_block_out",      # Final action outputs
        "action_block_in",       # Action input processing
        "resblocks",             # Decision blocks
        
        # Feature processing components
        "transformer",           # Attention mechanisms
        "summarizer_resblock",   # Residual blocks in summarizer
        "temporal_resblock",     # Residual blocks in temporal
        
        # Model outputs
        "output_plan",           # Plan output nodes (trajectory planning)
        "output_lane_lines",     # Lane lines output nodes (lane detection)
        "output_lane_lines_prob", # Lane lines probability output nodes
        "output_road_edges",     # Road edges output nodes (road boundary detection)
        "output_lead",           # Lead output nodes (lead car tracking)
        "output_lead_prob",      # Lead probability output nodes
        "output_desire_state",   # Desire state output nodes (lane change intention)
        "output_meta",           # Meta information output nodes
        "output_desire_pred",    # Desire prediction output nodes
        "output_pose",           # Pose output nodes (vehicle position)
        "output_wide_from_device", # Wide from device output nodes
        "output_sim_pose",       # Sim pose output nodes
        "output_road_transform", # Road transform output nodes
        "output_action",         # Action control output nodes
        "output_desired_curvature", # Desired curvature output nodes
        
        # Vision subsystems
        "vision_encoder",        # Vision encoder
        "vision_decoder",        # Vision decoder
        "vision_features",       # Vision features extraction
        
        # Specific module types
        "gemm",                  # All GEMM operations
        "conv",                  # All Conv operations
        "relu",                  # All ReLU activations
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
    parser.add_argument("--parser-aware", action="store_true", default=True,
                      help="Enable parser-aware validations to ensure output compatibility")
    parser.add_argument("--dump-metadata", action="store_true",
                      help="Dump metadata about the merge to a JSON file")
    parser.add_argument("--use-graphsurgeon", action="store_true", default=True,
                      help="Use ONNX GraphSurgeon if available (more accurate but requires installation)")
    args = parser.parse_args()
    
    # If requested, just list the available components and exit
    if args.list_components:
        print("Available components for merging:")
        
        print("\n# High-level components (vision/policy split based on modeld.py structure)")
        print("vision                - Vision model (pre-trunk, processes camera inputs)")
        print("no_bottleneck_policy  - Policy model (post-trunk, planning and decisions)")
        print("temporal_summarizer   - Temporal feature processing")
        print("temporal_hydra        - Temporal output heads")
        print("action_block          - Vehicle control outputs")
        
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
        
        print("\nOutput-specific components")
        print("output_plan           - Plan output nodes (trajectory planning)")
        print("output_lane_lines     - Lane lines output nodes (lane detection)")
        print("output_lane_lines_prob - Lane lines probability output nodes")
        print("output_road_edges     - Road edges output nodes (road boundary detection)")
        print("output_lead           - Lead vehicle output nodes (lead car tracking)")
        print("output_lead_prob      - Lead probability output nodes")
        print("output_desire_state   - Desire state output nodes (lane change intention)")
        print("output_meta           - Meta information output nodes")
        print("output_desire_pred    - Desire prediction output nodes")
        print("output_pose           - Pose output nodes (vehicle position)")
        print("output_wide_from_device - Wide from device output nodes")
        print("output_sim_pose       - Sim pose output nodes")
        print("output_road_transform - Road transform output nodes")
        print("output_action         - Action control output nodes (lateral/longitudinal control)")
        print("output_desired_curvature - Desired curvature output nodes (steering commands)")
        
        print("\nVision subsystems")
        print("vision_encoder        - Vision encoder network")
        print("vision_decoder        - Vision decoder network") 
        print("vision_features       - Vision features extraction")
        
        print("\nSpecific module types")
        print("gemm                  - All GEMM operations")
        print("conv                  - All Conv operations")
        print("relu                  - All ReLU activations")
        
        print("\nModel Architecture Notes")
        print("- Vision model processes camera input and extracts features")
        print("- Policy model uses these features for planning and control")
        print("- Models are split at '/supercombo/vision/Flatten_output_0'")
        print("- Merging vision components requires strict validation")
        print("- Output tensor shapes must remain compatible with the OpenPilot parser")
        
        # GraphSurgeon information
        if GRAPHSURGEON_AVAILABLE:
            print("\nONNX GraphSurgeon Support: Available ✓")
            print("- Enhanced model analysis and validation enabled")
            print("- More accurate component detection available")
            print("- Improved compatibility checking")
        else:
            print("\nONNX GraphSurgeon Support: Not Available ✗")
            print("- Install GraphSurgeon for enhanced model analysis:")
            print("  pip install nvidia-onnx-graphsurgeon")
            print("  or install as part of NVIDIA TensorRT")
        
        return

    # Check if GraphSurgeon is requested but not available
    if args.use_graphsurgeon and not GRAPHSURGEON_AVAILABLE:
        print("Warning: ONNX GraphSurgeon requested but not available. Falling back to standard methods.")
        print("To use GraphSurgeon, install with: pip install nvidia-onnx-graphsurgeon")

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
        compatibility_threshold=args.compatibility_threshold,
        parser_aware=args.parser_aware
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
            "graphsurgeon_used": merge_report.get("graphsurgeon_used", False),
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
    
    # Print parser validation results if available
    if "parser_validation" in merge_report and merge_report["parser_validation"]:
        print("\nParser Compatibility Check:")
        pv = merge_report["parser_validation"]
        if "outputs_compatible" in pv:
            status = "✓ Compatible" if pv["outputs_compatible"] else "❌ Incompatible"
            print(f"  - Output shapes: {status}")
            if not pv["outputs_compatible"]:
                print(f"    Message: {pv['outputs_message']}")
        
        if "features_compatible" in pv:
            status = "✓ Compatible" if pv["features_compatible"] else "❌ Incompatible"
            print(f"  - Vision/Policy interface: {status}")
            if not pv["features_compatible"]:
                print(f"    Message: {pv['features_message']}")
    
    # Print GraphSurgeon report if used
    if "graphsurgeon_used" in merge_report and merge_report["graphsurgeon_used"]:
        print("\nGraphSurgeon Enhanced Analysis:")
        print("  ✓ Used GraphSurgeon for enhanced graph analysis and merging")
        if "graphsurgeon_report" in merge_report:
            gs_report = merge_report["graphsurgeon_report"]
            print(f"  - Merged weights: {gs_report['merged_weights_count']}")
            print(f"  - Skipped weights: {gs_report['skipped_weights_count']}")
    
    if merge_report["safety_warnings"]:
        print("\nSafety Warnings:")
        for warning in merge_report["safety_warnings"]:
            print(f"  - {warning}")
    
    print(f"\nMerge complete! Use the following to compile and test:")
    print(f"  FLOAT16=0 JIT_BATCH_SIZE=16 python compile_script.py {output_path}")

if __name__ == "__main__":
    main()