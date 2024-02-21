import numpy as np
import onnx
import math
import time

import numpy
import copy

from onnx import helper, numpy_helper
from onnx import shape_inference


class StrategiesModifier:

    def __init__(self, source_path, target_path, configuration={}):
        self.strategies = {
            "padding": self.replace_padding,
            "strides": self.replace_strides,
            "params": self.replace_params,
            "graph": self.replace_graph,
            "flatten": self.repair_flattened_layer,
            "hyperparams": self.check_and_replace_all,
            "symbolic_dimensions": self.repair_symbolic_dimensions,
            "transpose": self.neutralize_transpose,
            "adjust_weights_if_transpose": self.adjust_weights_if_transpose,
            "repair_input_dimension": self.repair_input_dimension
        }

        self.PARAMS_WEIGHTS = 1
        self.PARAMS_BIASES = 2
        self.default_hyperparameters = ["pads", "group", "kernel_shape", "dilations", "epsilon", "min", "max"] # "strides",  
        self.default_parameters = ["Conv", "DequantizeLinear" "Add", "Mul", "Clip", "GlobalAveragePool", "Flatten", "Gemm",\
             "AveragePool", "Relu", "MaxPool", "Squeeze", "Reshape", "BatchNormalization", "Flatten"]
        self.infer_shapes = False,
        self.source_path = source_path
        self.target_path = target_path
        self.configuration = configuration
        # Use this object to log modifications. Use layer as key, and append an array with changes.
        self.log_modifications = []
        self.num_log_modifications = 0

    def load(self, target_path=None, infer_shapes=False):
        self.source = onnx.load(self.source_path)
        # TODO: refactor
        self.original_source = onnx.load(self.source_path)
        self.original_target = onnx.load(self.target_path if target_path is None else target_path)
        self.target = onnx.load(self.target_path if target_path is None else target_path)
        self.previous_target = self.target
        self.infer_shapes = infer_shapes
        if (infer_shapes):
            self.source = shape_inference.infer_shapes(self.source)
            self.target = shape_inference.infer_shapes(self.target)

        self.source_initializer = self.source.graph.initializer
        self.target_initializer = self.target.graph.initializer
        self.source_params = {i.name: i for i in self.source_initializer} 
        self.target_params = {i.name: i for i in self.target_initializer} 
        return self

    def apply(self, strategy, custom_config=None, loaded=True):
        self.previous_target = copy.deepcopy(self.target)
        func_to_call = self.strategies[strategy]

        # Allowing implicit loading to allow multiple modification
        # on the same model.
        if(not loaded):
            self.load()

        config = self.configuration if custom_config is None else custom_config
        target_path = func_to_call(config)

        return target_path

    def load_onnx_model(self, model_path, data):
        # Specify input in data.
        return self.model_loader.load_onnx_model(model_path, data)
    def get_target():
        return self.target


    def get_num_log_modifications(self):
        return self.num_log_modifications

    def get_log_modifications(self):
        return self.log_modifications


    def check_and_replace_all(self, configuration):
        
        hyperparameters = configuration["types"] if "types" in configuration \
            else self.default_hyperparameters
        op_types = configuration["op_types"] if "op_types" in configuration else ["Conv"]
        param_indexes = configuration["param_indexes"] if "param_indexes" in configuration else None

        source_graph = self.source.graph
        target_graph = self.target.graph

        source_nodes = source_graph.node
        target_nodes = target_graph.node

        for op_type in op_types:
            conv_source_nodes = [node for node in source_nodes if node.op_type == op_type]
            conv_target_nodes = [self.find_matching_target_node(node) for node in source_nodes if node.op_type == op_type]
            print("Checking op_type: " + op_type)

            if (len(conv_source_nodes) != len(conv_target_nodes)):
                print("Op Type " + op_type + " has a layer number mismatch between source and target.\nConsider graph analysis.")
                continue

            range_to_check = param_indexes if param_indexes is not None else range(len(conv_source_nodes))
            cache = {}
            for index in range_to_check:
                for hyperparameter in hyperparameters:
                    self.check_and_replace_hyperparameter(conv_source_nodes, conv_target_nodes, configuration = {"layer_index": index, "hyperparameter_name": hyperparameter}, cache=cache)

        return self

    def check_and_replace_hyperparameter(self, source_nodes, target_nodes, configuration, cache=None):
        # set a number of arbitrary strategies to add padding nodes.

        layer_index = configuration["layer_index"]
        hyperparameter_name = configuration["hyperparameter_name"]
        
        source_node = source_nodes[layer_index]
        source_attrs = source_node.attribute
        # TODO: refactor
        target_node = self.find_matching_target_node(source_nodes[layer_index], cache=cache)
        if target_node is None:
            return

        target_attrs = target_node.attribute

        source_hp = [attr for attr in source_attrs if attr.name == hyperparameter_name]
        source_hp = source_hp[0] if len(source_hp) != 0 else None
        target_hp = [attr for attr in target_attrs if attr.name == hyperparameter_name]
        target_hp = target_hp[0] if len(target_hp) != 0 else None

        if (target_hp is None and source_hp is not None):
            self.log_modifications.append({
                "modification": "check_and_replace_hyperparameter",
                "note": "Hyperparameter (" + hyperparameter_name + ") missing in target." + \
                source_node.name + " vs " + target_node.name
            })
            self.num_log_modifications += 1
            new_target_hp = helper.make_attribute(hyperparameter_name, source_hp.ints)
            target_attrs.append(new_target_hp)
        elif(source_hp is None and target_hp is not None):
            target_attrs.remove(target_hp)
            self.log_modifications.append({
                "modification": "check_and_replace_hyperparameter",
                "note": "Hyperparameter (" + hyperparameter_name + ") missing in source.\n" + \
                source_node.name + " vs " + target_node.name
            })
            self.num_log_modifications += 1

        elif(source_hp is None and target_hp is None):
            #print("Hyperparameter (" + hyperparameter_name + ") not found, neither in source nor in target nodes. Skipping...")
            pass
        elif (source_hp.ints != target_hp.ints):
            target_attrs.remove(target_hp)
            new_target_hp = helper.make_attribute(hyperparameter_name, source_hp.ints)
            target_attrs.append(new_target_hp)

            self.log_modifications.append({
                "modification": "check_and_replace_hyperparameter",
                "note": "Hyperparameter (" + hyperparameter_name + ") shape mismatch.\n" + \
                source_node.name + " vs " + target_node.name + "\n" + \
                str(source_hp.ints) + " vs " + str(target_hp.ints)
            })
            self.num_log_modifications += 1

    def replace_padding(self, configuration):
        return self.check_and_replace_hyperparameter(self.source, self.target, \
            {"layer_index": configuration.layer_index, "hyperparameter_name": "pads"})

    def replace_strides(self, configuration):
        return self.check_and_replace_hyperparameter(self.source, self.target, \
            {"layer_index": configuration.layer_index, "hyperparameter_name": "strides"})

    def replace_group(self, configuration):
        return self.check_and_replace_hyperparameter(self.source, self.target, \
            {"layer_index": configuration.layer_index, "hyperparameter_name": "group"})

    def replace_dilations(self, configuration):
        return self.check_and_replace_hyperparameter(self.source, self.target, \
            {"layer_index": configuration.layer_index, "hyperparameter_name": "dilations"})

    def replace_kernel_shape(self, configuration):
        return self.check_and_replace_hyperparameter(self.source, self.target, \
            {"layer_index": configuration.layer_index, "hyperparameter_name": "kernel_shape"})


    def replace_params(self, configuration):

        #hyperparameters = configuration["types"] if "types" in configuration \
        #    else ["pads", "strides", "kernel_shape", "group", "dilations"]
        op_types = configuration["op_types"] if "op_types" in configuration else self.default_parameters

        input_param_indexes = configuration["input_param_indexes"] \
            if "input_param_indexes" in configuration else [self.PARAMS_WEIGHTS]

        source_graph = self.source.graph
        target_graph = self.target.graph
        source_nodes = source_graph.node
        target_nodes = target_graph.node

        param_indexes = configuration["param_indexes"] if "param_indexes" in configuration else range(len(source_nodes))
        log_overview = []

        for op_type in op_types:
            source_nodes = [node for node in source_nodes if node.op_type == op_type]
            target_nodes = [node for node in target_nodes if node.op_type == op_type]
            print("Checking op_type:" + op_type)

            if (len(source_nodes) != len(target_nodes)):
                print("Op Type " + op_type + " has a layer number mismatch between source and target.\n\
                    Consider graph analysis.")
                continue
            elif len(source_nodes) == 0:
                print("Length of source nodes is 0 for " + op_type + ". Skipping...")
                continue

            matching_target_cache = {}
        
            for i in param_indexes:
                # Only this needs to change.
                index = i

                if index == len(source_nodes):
                    print("Reducing node...")
                    index -= 1

                elif index > len(source_nodes):
                    print("Out of range - index " + str(index) + " For a list with range " +str(len(source_nodes)) + ". Skipping...")
                    continue

                # print(len(source_nodes))
                # print(index)
                source_node = source_nodes[index]
                target_node = self.find_matching_target_node(source_node, cache=matching_target_cache)
                
                if target_node is None:
                    break

                for param_index in input_param_indexes:
                    if(len(source_node.input) > param_index):
                        source_weight = source_node.input[param_index]
                        target_weight = target_node.input[param_index]

                        source_weight_tensor = numpy_helper.to_array(self.source_params[source_weight])
                        target_weight_tensor = numpy_helper.to_array(self.target_params[target_weight])
                        if (source_weight_tensor is None or target_weight_tensor is None):
                            print("One of the tensors is None. Skipping...")
                            break

                        shape_match = source_weight_tensor.shape == target_weight_tensor.shape
                        
                        if (not shape_match or not (source_weight_tensor == target_weight_tensor).all()):
                            print("Parameter mismatch found! Inputs (source): " + str(source_weight) + " vs (target): " + str(target_weight))

                            self.target_params[target_weight] = numpy_helper.from_array(source_weight_tensor, name=target_weight)

                            for idx, key in enumerate(self.target_params.keys()):
                                new_tensor = self.target_params[key]
                                
                                # TODO: Clean unused weights.
                                self.target.graph.initializer[idx].CopyFrom(new_tensor)

                            log_overview.append(
                                "Layer " + str(i) + ": Parameter mismatch - index " + str(param_index) + "."
                            )
                            self.num_log_modifications += 1

                    else:
                        print("Index provided does not exist in node. Skipping...")

        if len(log_overview) != 0:
            self.log_modifications.append({
                "modification": "replace_params",
                "op_types": op_types,
                "overview": log_overview
            })

        return self
                        

    def replace_graph(self, configuration):
        op_types = configuration["op_types"] if "op_types" in configuration else ["Conv"]
        param_indexes = configuration["param_indexes"] if "param_indexes" in configuration else None

        source_graph = self.source.graph
        target_graph = self.target.graph
        source_nodes = source_graph.node
        target_nodes = target_graph.node

        matching_target_cache = {}
        for op_type in op_types:
            op_source_nodes = [node for node in source_nodes if node.op_type == op_type]

            for param_index in (param_indexes if param_indexes is not None else range(len(op_source_nodes))):

                # If the first node is analyzed, consider the previous node
                # to be the very first node of the model.
                if param_index == 0:
                    print("You cannot analyze the very first node of the model. Skipping...")
                    continue
                else:
                    curr_source = op_source_nodes[param_index]
                    curr_target = self.find_matching_target_node(curr_source, cache=matching_target_cache)
                    # print("Curr_target: " + str(curr_target))
                    # TODO: Refactor.
                    prev_source_orig = self.find_previous_node(curr_source, self.original_source.graph.node, op_type)
                    
                    if (prev_source_orig is None):
                        continue
                    
                    prev_source = [node for node in source_nodes if node.name == prev_source_orig.name][0]

                    if (prev_source == curr_source):
                        continue
                    
                    prev_target = self.find_matching_target_node(prev_source, cache=matching_target_cache)

                # TODO: refactor.
                source_structure = self.extract_structure(prev_source, curr_source, self.original_source.graph.node, False)
                target_structure = self.extract_structure(prev_target, curr_target, target_nodes, False)

                if(source_structure != target_structure):
                    print("Found different structure in target from source. Updating structure from source...")
                    print("S: " + curr_source.name)
                    print("T: " + curr_target.name)
                    print("Structure:")  
                    print(source_structure)
                    print(target_structure)

                    self.log_modifications.append({
                        "modification": "replace_graph",
                        "op_type": op_type,
                        "param_index": param_index,
                        "update_structure": {
                            "source_name": curr_source.name,
                            "target_name": curr_target.name,
                            "source_structure": source_structure,
                            "target_structure": target_structure
                        }
                    })
                    self.num_log_modifications += 1

                    source_structure = self.extract_structure(prev_source, curr_source, source_nodes, True)
                    target_structure = self.extract_structure(prev_target, curr_target, target_nodes, True)

                    # print(source_structure == target_structure)
                    self.update_structure(source_data={
                        "structure": source_structure,
                        "prev_node": prev_source,
                        "curr_node": curr_source
                    },
                    target_data={
                        "structure": target_structure,
                        "prev_node": prev_target,
                        "curr_node": curr_target
                    })

        return self

    def find_previous_node(self, base_node, nodes, op_type, initial_node=None):

        if (initial_node is not None and base_node.op_type == op_type): 
            return base_node

        nodes_found = []
        for node in nodes:

            if base_node is not None and node.output[0] in base_node.input:
                nodes_found.append(node)
        if  len(nodes_found) == 1:
            return self.find_previous_node(nodes_found[0], nodes, op_type, nodes_found[0])
        else:
            return base_node

    def extract_structure(self, start_node, end_node, nodes, full_nodes=True):
        if start_node is None:
            return []
        
        queue = [start_node]
        visited = {}
        # print(start_node)
        visited[start_node.name] = True
        node_structure = []

        base_node_output = None

        while len(queue) != 0:
            node = queue.pop(0)
            
            base_node_output = node.output[0]

            if (base_node_output in end_node.input):
                continue

            for append_node in [node for node in nodes if base_node_output in node.input]:
                if append_node.name not in visited or visited[append_node.name] == False:
                    queue.append(append_node)
                    visited[append_node.name] = True
                    node_structure.append(append_node if full_nodes else append_node.op_type)
   
        return node_structure

    def repair_symbolic_dimensions(self, configuration):

        if (not self.infer_shapes):
            print("Symbolic dimension check cannot be applied with deactivated option 'infer-shapes'.")
            return self

        source_graph = self.source.graph
        target_graph = self.target.graph
        source_nodes = source_graph.node
        target_nodes = target_graph.node

        op_types = configuration["op_types"] if "op_types" in configuration else ["Conv"]
        param_indexes = configuration["param_indexes"] if "param_indexes" in configuration else None

        policy = configuration["policy"] if "policy" in configuration else "convert_target_to_actual"

        matching_target_cache = {}
        for op_type in op_types:
            op_source_nodes = [node for node in source_nodes if node.op_type == op_type]
            # op_target_nodes = [node for node in target_nodes if node.op_type == op_type]

            for index in (param_indexes if param_indexes is not None else range(len(op_source_nodes))):


                op_source = op_source_nodes[index]
                op_target = self.find_matching_target_node(op_source, cache=matching_target_cache)
                if op_target is None:
                    break

                # Using this as index() is not accessible for this class type.
                for main_index_s in range(len(source_nodes)):
                    if (source_nodes[main_index_s] == op_source):
                        break
                
                # Using this as index() is not accessible for this class type.
                for main_index_t in range(len(target_nodes)):
                    if (target_nodes[main_index_t] == op_target):
                        break

                source_dim = self.source.graph.value_info[main_index_s].type.tensor_type.shape.dim
                target_dim = self.target.graph.value_info[main_index_t].type.tensor_type.shape.dim

                if (policy == "convert_target_to_actual"):
                    i = 0
                    log_dims = []
                    while i < len(target_dim):
                        if (target_dim[i].dim_param != ""):
                            print ("Symbolic input detected - changed symbolic input dimension to numeric (1) for target. Layer: " + op_target_nodes[index].name)
                            target_dim[i].dim_value = 1
                            target_dim[i].dim_param = "1"
                            log_dims.append(i)
                        i += 1

                    self.log_modifications.append({
                        "modification": "convert_symbolic_dimension",
                        "type": "convert_target_to_actual",
                        "op_type": op_type,
                        "param_index": index,
                        "dimensions": log_dims
                    })
                    self.num_log_modifications += 1

                i = 0
                log_dims = []
                while i < len(source_dim):
                    if (source_dim[i].dim_param != target_dim[i].dim_param):
                        print ("Note: Source model contained a symbolic dimension in layer " + op_source_nodes[index].name + \
                            " - but not in target. If you want this to be applied to target - set policy to update_from_source in configuration")

                        if (policy == "update_from_source"):
                            target_dim[i].dim_value = target_dim[i].dim_value
                            target_dim[i].dim_param = target_dim[i].dim_param
                            log_dims.append(i)
                            print("Value updated from source to target.")
                    i += 1
                
                self.log_modifications.append({
                    "modification": "convert_symbolic_dimension",
                    "type": "update_target_from_source",
                    "op_type": op_type,
                    "param_index": index,
                    "dimensions": log_dims
                })
                self.num_log_modifications += 1

        return self

    def repair_input_dimension(self, configuration):
        print("Attempting to repair input dimension...")
        source_dim = self.source.graph.input[0].type.tensor_type.shape.dim
        target_dim = self.target.graph.input[0].type.tensor_type.shape.dim

        for i in range(len(source_dim)):
            if source_dim[i].dim_value == "":
                target_dim[i].dim_param = source_dim[i].dim_param
            else:
                target_dim[i].dim_value = source_dim[i].dim_value

        self.log_modifications.append({
            "modification": "repair_input_dimension"
        })
        self.num_log_modifications += 1
        
        return self

    def neutralize_transpose(self, configuration):

        print("Neutralizing transpose and adjusting input dimension...")

        source_nodes = self.source.graph.node
        target_nodes = self.target.graph.node
        op_target_nodes = [node for node in target_nodes if node.op_type == "Transpose"]
        param_indexes = configuration["param_indexes"] if "param_indexes" in configuration \
             else range(len(op_target_nodes))
        order = configuration["order"] if "order" in configuration else None


        for i in param_indexes:
            target_node = op_target_nodes[i]

            old_perm = [a for a in target_node.attribute if a.name == "perm"][0]
            new_perm = new_perm = order if order is not None else [i for i in range(len(old_perm.ints))]

            # Neutralize transpose by setting input tensor
            # same as output from the layer.
            new_target_perm = helper.make_attribute("perm", new_perm)
            target_node.attribute.remove(old_perm)
            target_node.attribute.append(new_target_perm)
            
        # Fix Dimensions.
        source_dim = self.source.graph.input[0].type.tensor_type.shape.dim
        target_dim = self.target.graph.input[0].type.tensor_type.shape.dim

        for i in range(len(source_dim)):
            # Replace symbolic value with 1.
            target_dim[i].dim_value = source_dim[i].dim_value
            if (source_dim[i].dim_param != ""):
                target_dim[i].dim_param = source_dim[i].dim_param
  
        
        source_concat = [node for node in source_nodes if node.op_type == "Concat"]
        if len(source_concat) == 0:
            return self
        first_concat_source = source_concat[0]
        concat_source_int = first_concat_source.attribute[0].i
        target_concat = [node for node in target_nodes if node.op_type == "Concat"]
        if len(target_concat) == 0:
            return self

        first_concat_target = target_concat[0]
        concat_target_int = first_concat_target.attribute[0].i

        self.log_modifications.append({
            "modification": "neutralize_transpose",
            "param_indexes": list(param_indexes)
        })
        # Double fix - dimension and transpose.
        self.num_log_modifications += 2
        
        # TODO: generalize solution.
        if concat_source_int != concat_target_int:
            new_target_ax = helper.make_attribute("axis", concat_source_int)
            first_concat_target.attribute.remove(first_concat_target.attribute[0])
            first_concat_target.attribute.append(new_target_ax)
            self.num_log_modifications += 1
   
            gather_target = [node for node in target_nodes if node.op_type == "Gather"]
            if (len(gather_target) == 0):
                return self
            
            # All Gather nodes should have the same input index, therefore we obtain the first.
            gather_source = [node for node in source_nodes if node.op_type == "Gather"]
            if len(gather_source) == 0:
                return self
            gather_source_0 = gather_source[0]
            for gather_t in gather_target:
                
                gather_t.attribute.remove(gather_t.attribute[0])
                gather_t.attribute.append(gather_source_0.attribute[0])
                self.num_log_modifications += 1



            unsqueeze_target = [node for node in target_nodes if node.op_type == "Unsqueeze"]
            if (len(unsqueeze_target) == 0):
                return self

            # All Unsqueeze nodes should have the same input index, therefore we obtain the first.
            source_unsqueeze = [node for node in source_nodes if node.op_type == "Unsqueeze"]
            if len(source_unsqueeze) == 0:
                return self
                
            unsqueeze_source_0 = source_unsqueeze[0]
            for unsqueeze_t in unsqueeze_target:
                unsqueeze_t.attribute.remove(unsqueeze_t.attribute[0])
                unsqueeze_t.attribute.append(unsqueeze_source_0.attribute[0])
                self.num_log_modifications += 1


        return self

    def adjust_weights_if_transpose(self, configuration):
        print("Adjusting weights if there is a transpose...")
        source_graph = self.source.graph
        target_graph = self.target.graph
        source_nodes = source_graph.node
        target_nodes = target_graph.node

        first_target_node = target_nodes[0]
        target_attrs = first_target_node.attribute
        first_target_node_perm = [attr for attr in target_attrs if attr.name == "perm"][0]

        if (first_target_node.op_type == "Transpose"):
            target_conv_nodes = [node for node in target_nodes if node.op_type == "Conv"]

            # Obtaining input.
            conv_node = target_conv_nodes[0]

            # Obtaining weights.
            conv_weight = conv_node.input[1]

            target_weight_tensor = numpy_helper.to_array(self.target_params[conv_weight])

            new_target_weight_tensor = np.transpose(target_weight_tensor, axes=first_target_node_perm.ints)
            if new_target_weight_tensor.shape == target_weight_tensor.shape:
                self.target_params[conv_weight] = numpy_helper.from_array(new_target_weight_tensor, name=conv_weight)

                for idx, key in enumerate(self.target_params.keys()):
                    new_tensor  = self.target_params[key]
                    
                    # TODO: Clean unused weights.
                    self.target.graph.initializer[idx].CopyFrom(new_tensor)

                self.log_modifications.append({
                    "modification": "adjust_weights_if_transpose_in_first_convolution",
                    "param_indexes": [0]
                })
                self.num_log_modifications += 1
        return self

    def repair_flattened_layer(self, configuration):

        source_graph = self.source.graph
        target_graph = self.target.graph
        source_nodes = source_graph.node
        target_nodes = target_graph.node

        param_indexes = configuration["param_indexes"] if "param_indexes" in configuration else None

        op_source_nodes = [node for node in source_nodes if node.op_type == "Flatten" or node.op_type == "Reshape"]
        op_target_nodes = [node for node in target_nodes if node.op_type == "Flatten" or node.op_type == "Reshape"]

        matching_target_cache = {}
        for index in (param_indexes if param_indexes is not None else range(len(op_source_nodes))):

            op_source = op_source_nodes[index]
            op_target = op_target_nodes[index]

            # Using this as index() is not accessible for this class type.
            for main_index_s in range(len(source_nodes)):
                if (source_nodes[main_index_s] == op_source):
                    break
            
            for main_index_t in range(len(target_nodes)):
                if (target_nodes[main_index_t] == op_target):
                    break

            prev_index_s = main_index_s - 1
            prev_index_t = main_index_t - 1

            prev_source = source_nodes[prev_index_s]
            prev_target = target_nodes[prev_index_t]
                
            if (prev_source.op_type == prev_target.op_type):
                inferred_perm = None
                if (self.infer_shapes and "inferred_perm" not in configuration):   

                    prev_source_dim = self.source.graph.value_info[prev_index_s].type.tensor_type.shape.dim
                    prev_target_dim = self.target.graph.value_info[prev_index_t].type.tensor_type.shape.dim
                    
                    if prev_source_dim != prev_target_dim:
                        if(prev_source_dim[0].dim_param != "" or prev_source_dim[1].dim_param == "1"):
                            print("Symbolic dimension detected in layer shape. Consider. Skipping flatten transposition.")
                        else:
                            print("Different dimension detected before flatten. Adding transposition layer...")
                            inferred_perm = self.infer_transpose(prev_source_dim, prev_target_dim)
                else:

                    print ("Warning: Shape inference disabled/overriden - utilizing manual transposition from configuration.")
                    inferred_perm = configuration["inferred_perm"] if "inferred_perm" in configuration else None

                if inferred_perm is not None:
                    print ("Adding Transpose for Flatten/Reshape...")
                    transpose_node = onnx.helper.make_node("Transpose", inputs=[], outputs=[], perm=inferred_perm)
                    prev_target_output = prev_target.output[0]
                    transpose_node.input.append(prev_target_output)
                    transpose_node.output.append(prev_target_output + "_transpose")
                    
                    op_target.input.remove(prev_target_output)
                    op_target.input.insert(0, prev_target_output + "_transpose")
                    
                    # TODO: Refactor to append to new model.
                    target_nodes.append(transpose_node)
                    self.log_modifications.append({
                        "modification": "repair_flattened_layer",
                        "index": str(index)
                    })
                    self.num_log_modifications += 1

            else:
                print("Unable to check previous nodes - divergence between source and target.")
              
        return self
    


    # ----------------- Helper Methods -----------------
    def get_target(self):
        return self.target

    def save_target(self, target, path):
        # onnx.checker.check_model(self.target)
        
        onnx.save(target, path)
        return self

    def save(self, path):
        # onnx.checker.check_model(self.target)
        onnx.save(self.target, path)
        return self

    def update_structure(self, source_data, target_data):
        # Whichever had source output (prev) as input, will now have target output (prev as input)
        # Whichever had output in source final input, will now have the 
        # Remove old nodes.

        source_structure = source_data["structure"]
        source_prev_node = source_data["prev_node"]
        source_curr_node = source_data["curr_node"]
        target_structure = target_data["structure"]
        target_prev_node = target_data["prev_node"]
        target_curr_node = target_data["curr_node"]

        source_nodes = self.source.graph.node
        target_nodes = self.target.graph.node

        # Detect and delete old target nodes in stucture.
        # target_out = None

        for node in target_structure:
            
            for input_name in node.input:
                if input_name in self.target_params:
                    del self.target_params[input_name]
                if input_name in self.target.graph.initializer:
                    self.target.graph.initializer.remove(input_name)
            self.target.graph.node.remove(node)

        insert_index = 0
        # Find the topological order of insertion.
        for target_node in self.target.graph.node:
            
            if target_node == target_curr_node:
                break
            insert_index += 1
            
        new_source_params = [] 
        for source_node in source_structure:
            
            # Connect with end.
            if source_node.output[0] in source_curr_node.input:
                # Replace source input with target input for end node.
                source_node.output.pop(0)
                source_node.output.insert(0, target_curr_node.input[0])

            
            # Connect with start
            if source_prev_node.output[0] in source_node.input:
                # Replace source input with target input for start node.
                source_node.input.pop(0)
                source_node.input.insert(0, target_prev_node.output[0])

            
            self.target.graph.node.insert(insert_index, source_node)
            insert_index += 1

            len_of_weights = len(source_node.input)
            for param_index in range(len_of_weights):
                # Skip input
                if param_index == 0:
                    continue

                source_weight_name = source_node.input[param_index]

                source_weight_tensor = numpy_helper.to_array(self.source_params[source_weight_name])
                if source_weight_name not in self.target_params:
                    new_source_params.append(source_weight_name)
                self.target_params[source_weight_name] = numpy_helper.from_array(source_weight_tensor, name=source_weight_name)

        # TODO: Update initializer.
        for idx, key in enumerate(self.target_params.keys()):
            new_tensor = self.target_params[key]
            if new_tensor.name in new_source_params:
                self.target.graph.initializer.insert(idx, new_tensor)
            else:
                self.target.graph.initializer[idx].CopyFrom(new_tensor)

        return self

    # Algorithm to detect dims.
    def infer_transpose(self, source_dim, target_dim):
        dims = []
        td = 0
        while td < len(target_dim):
            if source_dim[td] == target_dim[td]:
                dims.append(td)
            else:
                sd = 0
                while sd < len(source_dim):
                    # TODO: Refactor
                    if (target_dim[td] == source_dim[sd] and sd not in dims):
                        dims.append(sd)
                        break
                    sd += 1
            td += 1

        return dims

    def get_index_by_layer_name(self, nodes, layer_name):

        for i in range(len(nodes)):
            if nodes[i].name == layer_name:
                return i

    def find_matching_target_node(self, source_node, cache=None):

        # Obtain layers that have the same op_type with source
        all_target_nodes = self.target.graph.node
        candidate_target_nodes = [node for node in all_target_nodes if node.op_type == source_node.op_type and \
            len(node.input) == len(source_node.input) and len(node.output) == len(source_node.output)]

        if cache is not None and source_node.name in cache:
            return cache[source_node.name]

        for candidate_target_node in candidate_target_nodes:
            if (self.check_candidate_node(source_node, candidate_target_node)):
                if cache is not None:
                    cache[source_node.name] = candidate_target_node
                return candidate_target_node


    def check_candidate_node(self, source_node, candidate_target_node):

        if source_node.name == candidate_target_node.name:
            return True

        for i in range(1, len(source_node.input)):

            source_input = source_node.input[i]
            target_input = candidate_target_node.input[i]
            

            if (source_input not in self.source_params or target_input not in self.target_params):
                return False

            source_input_tensor = numpy_helper.to_array(self.source_params[source_input])
            target_input_tensor = numpy_helper.to_array(self.target_params[target_input])

            # Note: threshold might vary based on the model. Consider refactoring.
            if source_input_tensor.shape != target_input_tensor.shape or not np.allclose(source_input_tensor, target_input_tensor, atol=0.2):
                return False
        
        return True

    def update_dimensions(self):
        # Determine dimensions.
        source_graph = self.source.graph
        target_graph = self.target.graph

        source_nodes = source_graph.node
        target_nodes = target_graph.node

        for i in range(len(source_nodes)):
            source_node = source_nodes[i]
            target_node = self.find_matching_target_node(source_node)
            if (target_node is None):
                continue
            
            for main_index_s in range(len(source_nodes)):
                if (source_nodes[main_index_s] == source_node):
                    break
            for main_index_t in range(len(target_nodes)):
                if (target_nodes[main_index_t] == target_node):
                    break

            source_dim = self.source.graph.value_info[main_index_s].type.tensor_type.shape.dim
            target_dim = self.target.graph.value_info[main_index_t].type.tensor_type.shape.dim

            i = 0
            while i < len(source_dim):
                target_dim[i].dim_value = source_dim[i].dim_value
                i += 1
            print(target_dim)
        return self
