
from pathlib import Path
import tvm
import json
import re
import numpy as np
import os
from os import path, listdir, remove
from os.path import isfile, isdir, join
import copy
import scipy
import ddks
import torch
from scipy.stats import chisquare
import time
# import genai_evaluation as ge
# from statsmodels.distributions.empirical_distribution import ECDF

class Analyzer:

    def __init__(self, script_dir):
        self.script_dir = script_dir
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

        p_ordered = []

        #-------------------- Plot analysis (differential testing.) --------------------
        for node in graph1["nodes"]:
            if "conv" not in node["name"]:
                continue
            for input_val in node["inputs"]:
                if (re.search("^[p][0-9]+$", input_val)):
                    p_keys_to_consider1.append(input_val)
                    node_to_param_map1[node["name"] + "____0"] = input_val
                    map1_nodes_arr.append(node["name"] + "____0")


        count = 0
        for node in graph2["nodes"]:
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

        graph_p_diffs = {}
        for input_val in p_keys_to_consider1:
            if (re.search("^[p][0-9]+$", input_val)):
                graph_p_diffs[input_val] = np.subtract(params2[input_val].numpy().reshape(-1), params1[input_val].numpy().reshape(-1))

        keys_keras = None
        keys_torch = None

        plot_mean_data = {}
        plot_std_data = {}
        plot_max_data = {}
        plot_avg_data = {}

        param_index_under_test = 0
        param_count = 0

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

        # For each file...
        for param_file in total_images_params:
            with open(join(debug_path1, param_file), "rb") as f:
                intermediate_tensors1 = dict(tvm.relay.load_param_dict(f.read()))

            with open(join(debug_path2, param_file), "rb") as f2:
                intermediate_tensors2 = dict(tvm.relay.load_param_dict(f2.read()))

            # For each crucial layer...
            for tensor1_name in node_to_param_map1.keys():
            
            
                tensor1 = intermediate_tensors1[tensor1_name]
                tensor2_name = map1_to_map2[tensor1_name]
                tensor2 = intermediate_tensors2[tensor2_name]

                A = tensor1.asnumpy()
                B = tensor2.asnumpy()

                res_A = np.reshape(A, -1)
                res_B = np.reshape(B, -1)
                if res_A.shape != res_B.shape:
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
     
        layer_index = 0
        layer_info = []
        
        # Layer-wise iteration.
        for tensor1_name in node_to_param_map1.keys():

            similar_params_elem_wise = []
            dissimilar_params_elem_wise = []

            similar_params_diffed = similar_params_data[tensor1_name]
            similar_params_len = len(similar_params_diffed)

            dissimilar_params_diffed = dissimilar_params_data[tensor1_name]
            dissimilar_params_len = len(dissimilar_params_diffed)

            layer_flattened_size = len(similar_params_diffed[0])
            start_time = time.time()
            similar_params_elem_wise = np.transpose(similar_params_diffed, [1, 0])
            dissimilar_params_elem_wise = np.transpose(dissimilar_params_diffed, [1, 0])
            print(str(time.time() - start_time))

            # Analyze distribution element wise.
            count_elems_out_of_dev = 0
            for ind in range(len(similar_params_elem_wise)):

                similar_elem_wise = similar_params_elem_wise[ind]
                dissimilar_elem_wise = dissimilar_params_elem_wise[ind]

                # If empty list in similar,
                # check if at least one element is different on dissimilar.
                if len(similar_elem_wise) == 0:
                    if any(e != 0.0 for e in dissimilar_elem_wise):
                        count_elems_out_of_dev += 1
                        

                elif not np.any(similar_elem_wise):
                    if np.any(dissimilar_elem_wise):
                        count_elems_out_of_dev += 1

                # If same elements in base but at least one different from dissimilar,
                # consider it - no all-identical values allowed in kruskal.
                # elif np.all(np.isclose(similar_elem_wise, similar_elem_wise[0])):
                #     if any(e != similar_elem_wise[0] for e in dissimilar_elem_wise):
                #         count_elems_out_of_dev += 1
                                
                else:
                    kruskal_test = scipy.stats.kruskal(similar_elem_wise, dissimilar_elem_wise)
                    # print(similar_elem_wise)
                    # print(dissimilar_elem_wise)
                    # print(kruskal_test)
                    # TODO: Check iqr layers in lower quartile
                    if kruskal_test.pvalue < 0.05:
                        count_elems_out_of_dev += 1
           
            percentage_points_dissimilar = (count_elems_out_of_dev/layer_flattened_size)*100
            print(percentage_points_dissimilar)

            tensor2_name = map1_to_map2[tensor1_name]
            param_values = graph_p_diffs[node_to_param_map1[tensor1_name]]

            # Calculate percentage
            layer_info.append({
                "model1_layer_name": tensor1_name,
                "model2_layer_name": tensor2_name,
                "layer_index": layer_index,
                "elems_percentage": percentage_points_dissimilar
            })

            layer_index += 1

        if remove_similar_upon_completion:
            print ("Removing used similar images.")
            similar_images_params = ["output_tensors_" + i + ".params" for i in similar_images_stem]
            for image_param in similar_images_params:
                remove(join(debug_path1, image_param))
                remove(join(debug_path2, image_param))

        return layer_info