from pathlib import Path
import tvm
import json
import re
import os
from os import path, listdir, remove
from os.path import isfile, isdir, join
import copy
import scipy
import ddks
import torch
import time
import numpy as np

from joblib import Parallel, delayed

class Analyzer:

    def __init__(self, script_dir, n_jobs=-1):
        self.script_dir = script_dir
        self.n_jobs = n_jobs
        pass

    def get_target_nodes_length(self, config):

        debug_path2 = config["layer_analysis_data"]["debug_path2"]
        graph_dump2 = debug_path2 + "/_tvmdbg_graph_dump.json"
        with open(graph_dump2, "r") as g:
            graph2 = json.loads(g.read())

        return len([node for node in graph2["nodes"] if "conv" in node["name"]])

    def perform_fault_localization(self, config, similar_images, dissimilar_images, remove_similar_upon_completion=False):
        # Load params and everything.
        # Perform fault localization.
        # Return problematic layers for dissimilar image.

        # Built Model Paths.
        debug_path1 = config["layer_analysis_data"]["debug_path1"]
        debug_path2 = config["layer_analysis_data"]["debug_path2"]

        # Graph Dumps
        graph_dump1 = debug_path1 + "/_tvmdbg_graph_dump.json"
        graph_dump2 = debug_path2 + "/_tvmdbg_graph_dump.json"

        # Model Parameters used for static analysis (Paths).
        orig_lib_path1 = config["layer_analysis_data"]["lib1_graph_params"]
        orig_lib_path2 = config["layer_analysis_data"]["lib2_graph_params"]

        params1 = None
        params2 = None

        graph1 = None
        graph2 = None

        with open(orig_lib_path1, "rb") as p:
            mybytearray = bytearray(p.read())
            params1 = tvm.relay.load_param_dict(mybytearray)
            # print((params1))

        with open(orig_lib_path2, "rb") as p:        
            mybytearray = bytearray(p.read())
            params2 = tvm.relay.load_param_dict(mybytearray)

        with open(graph_dump1, "r") as g:
            graph1 = json.loads(g.read())

        with open(graph_dump2, "r") as g:
            graph2 = json.loads(g.read())
    
        p_keys_to_consider1 = []
        node_to_param_map1 = {}

        p_keys_to_consider2 = []
        node_to_param_map2 = {}

        map1_nodes_arr = []
        map1_to_map2 = {}

        map2_nodes_hash = {}

        p_ordered = []

        erroneous_param_nodes = []
        erroneous_activations = []

        #-------------------- Plot analysis (differential testing.) --------------------
        for node in graph1["nodes"]:
            # TODO: Extend it so that it works with other crucial layers as well.
            if "conv" not in node["name"]:
                continue
            
            for input_val in node["inputs"]:
                if (re.search("^[p][0-9]+$", input_val)):
                    p_keys_to_consider1.append(input_val)
                    
                    node_to_param_map1[node["name"] + "____0"] = input_val
                    map1_nodes_arr.append(node["name"] + "____0")


        count = 0
        for node in graph2["nodes"]:
            node_hash = tvm.ir.structural_hash(node)
            map2_nodes_hash[node_hash] = node
            if "conv" not in node["name"]:
                continue

            for input_val in node["inputs"]:
                if (re.search("^[p][0-9]+$", input_val)):
                    p_keys_to_consider2.append(input_val)
                    node_to_param_map2[node["name"] + "____0"] = input_val
                    if count >= len(map1_nodes_arr):
                        break
                    map1_tensor_name = map1_nodes_arr[count]
                    map1_to_map2[map1_tensor_name] = node["name"] + "____0"
                    
                    count += 1

        graph_p_abs_diffs = {}
        param_count = 0

        for node1 in graph1["nodes"]:
            node1_hash = tvm.ir.structural_hash(node1)
            inputs_node1 = [i for i in node1["inputs"] if re.search("^[p][0-9]+$", i)]

            if node1_hash in map2_nodes_hash:
                node2 = map2_nodes_hash[node1_hash]
                inputs_node2 = [i for i in node2["inputs"] if re.search("^[p][0-9]+$", i)]

                for i in range(len(inputs_node1)):
                    p1 = inputs_node1[i]
                    param1 = params1[p1]
                    p2 = inputs_node2[i]
                    param2 = params2[p2]

                    graph_p_abs_diffs[p1] = np.abs(np.subtract(param2.numpy().reshape(-1), param1.numpy().reshape(-1)))
                    param_count += 1
            else:
                # Attempt loading using key from param 1.
                for i in range(len(inputs_node1)):
                    p1 = inputs_node1[i]
                    param1 = params1[p1]
                    param2 = params2[p1]

                    print(param1.numpy().shape)
                    print(param2.numpy().shape)

                    reshape_param1 = param1.numpy().reshape(-1)
                    reshape_param2 = param2.numpy().reshape(-1)

                    if reshape_param1.shape == reshape_param2.shape:
                        graph_p_abs_diffs[p1] = np.abs(np.subtract(reshape_param1, reshape_param2))
                    else:
                        print("Warning: dimension mismatch on property: " + str(p1) + ", index: " + str(param_count))
                        erroneous_param_nodes.append({
                            "index": param_count,
                            "param": p1,
                        })
                        
                    param_count += 1
                
        keys_keras = None
        keys_torch = None

        param_index_under_test = 0

        similar_plot_mean_data = []
        dissimilar_plot_mean_data = None
        similar_images_stem = [Path(i).stem for i in similar_images]
        dissimilar_images_stem = [Path(i).stem for i in dissimilar_images]
        
        total_images_stem = [*similar_images_stem, *dissimilar_images_stem]
        total_images_params = ["output_tensors_" + i + ".params" for i in total_images_stem]

        # Note: Using python-3-8 - preserves key order by insertion order.
        similar_params_data = {}
        # layers_params_means = {}

        dissimilar_params_data = {}
        # TODO: change to set.
        problematic_layers = []

        # For each file...
        for param_file in total_images_params:
            with open(join(debug_path1, param_file), "rb") as f:
                intermediate_tensors1 = dict(tvm.relay.load_param_dict(f.read()))

            with open(join(debug_path2, param_file), "rb") as f2:
                intermediate_tensors2 = dict(tvm.relay.load_param_dict(f2.read()))

            # For each crucial layer...
            layer_index = 0

            for tensor1_name in node_to_param_map1.keys():     
                tensor1 = intermediate_tensors1[tensor1_name]
                tensor2_name = map1_to_map2[tensor1_name]
                tensor2 = intermediate_tensors2[tensor2_name]

                A = tensor1.asnumpy()
                B = tensor2.asnumpy()

                res_A = np.reshape(A, -1)
                res_B = np.reshape(B, -1)
                if res_A.shape != res_B.shape:       
                    if layer_index not in problematic_layers:
                        print("Warning: different shape across tensors " + tensor1_name + ", " + tensor2_name)
                        erroneous_activations.append({
                            "model1_layer_name": tensor1_name,
                            "model2_layer_name": tensor2_name,
                            "layer_index": layer_index,
                        })
                        problematic_layers.append(layer_index)
                    
                    layer_index += 1
                    continue

                # Element-wise difference of flattened arrays.
                # Note: We do not use abs to preserve which
                # version S/T had which activation value.
                dist_A_B = np.subtract(res_B, res_A)
                image_stem = str(Path(join(debug_path1, param_file)).stem).replace("output_tensors_", "")
                if image_stem in dissimilar_images_stem:
                    # Keep differences for dissimilar images.
                    #print ("Dissimilar Image: " + param_file)
                    if tensor1_name not in dissimilar_params_data:
                        dissimilar_params_data[tensor1_name] = []

                    dissimilar_params_data[tensor1_name].append(dist_A_B)
                else:
                    # Keep differences for all else images
                    #print ("Similar Image: " + param_file)
                    if tensor1_name not in similar_params_data:
                        similar_params_data[tensor1_name] = []
                    similar_params_data[tensor1_name].append(dist_A_B)

                layer_index += 1
     
        layer_index = 0
        layer_info = []

        #layers_count = [i for i in range(len(node_to_param_map1.keys())) if i not in problematic_layers]
        layer_index = 0
        # Layer-wise iteration.
        overall_start_time = time.time()
        for tensor1_name in node_to_param_map1.keys():
            # If there are no "similar images" data for the tensor, skip layer,
            # it has problem.
            if tensor1_name not in similar_params_data:
                print ("Skipping layer" + str(layer_index))
                layer_index += 1
                continue
            
            print("Performing analysis for layer " + str(layer_index))

            similar_params_diffed = similar_params_data[tensor1_name]
            similar_params_len = len(similar_params_diffed)

            dissimilar_params_diffed = dissimilar_params_data[tensor1_name]
            dissimilar_params_len = len(dissimilar_params_diffed)

            layer_flattened_size = len(similar_params_diffed[0])
            # start_time = time.time()
            similar_params_elem_wise = np.transpose(similar_params_diffed, [1, 0])
            dissimilar_params_elem_wise = np.transpose(dissimilar_params_diffed, [1, 0])

            # Analyze distribution element wise (in-parallel).
            count_elems_out_of_dev = 0
            #if self.n_gpu != -1:
            start_time = time.time()
            #, n_workers_per_gpu=self.n_workers_per_gpu
            results = Parallel(n_jobs=self.n_jobs)(\
                delayed(self.perform_KW_element_wise)(similar_params_elem_wise[ind], \
                dissimilar_params_elem_wise[ind]) \
                for ind in range(len(similar_params_elem_wise)))

            print("Elapsed Time: " + str(time.time() - start_time))

            count_elems_out_of_dev = np.sum(results)
            percentage_points_dissimilar = (count_elems_out_of_dev/layer_flattened_size)*100
            print("Percentage Dissimilar:" + str(percentage_points_dissimilar) + "%")

            tensor2_name = map1_to_map2[tensor1_name]
            param_values = graph_p_abs_diffs[node_to_param_map1[tensor1_name]]

            # Calculate percentage.
            # TODO: Extend for rest of graph.
            layer_info.append({
                "model1_layer_name": tensor1_name,
                "model2_layer_name": tensor2_name,
                "layer_index": layer_index,
                "elems_percentage": float(percentage_points_dissimilar),
                "params_mean_diff": np.mean(param_values),
                "erroneous_param_nodes": erroneous_param_nodes,
                "erroneous_activations": erroneous_activations
            })

            layer_index += 1
        print("Overall Elapsed Time: " + str(time.time() - start_time))

        if remove_similar_upon_completion:
            print ("Removing used similar images.")
            similar_images_params = ["output_tensors_" + i + ".params" for i in similar_images_stem]
            for image_param in similar_images_params:
                # Remove similar images both from source and target path.
                remove(join(debug_path1, image_param))
                remove(join(debug_path2, image_param))

        return layer_info

    def perform_KW_element_wise(self, similar_elem_wise, dissimilar_elem_wise, **kwargs):
        # If empty list in similar,
        # check if at least one element is different on dissimilar.
        count_elems_out_of_dev = 0

        if len(similar_elem_wise) == 0 or not np.any(similar_elem_wise):
            if np.any(dissimilar_elem_wise):
                count_elems_out_of_dev += 1
        elif np.all(similar_elem_wise == similar_elem_wise[0]):
            if np.any(dissimilar_elem_wise != similar_elem_wise[0]):
                count_elems_out_of_dev += 1
        else:
            kruskal_test = scipy.stats.kruskal(similar_elem_wise, dissimilar_elem_wise)
            if kruskal_test.pvalue < 0.05:
                count_elems_out_of_dev += 1
        return count_elems_out_of_dev