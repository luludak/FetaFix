import onnx
import onnxruntime as ort
from os import listdir, remove, path, makedirs
from os.path import isfile, isdir, join, exists, normpath, basename
import tvm
import tvm.relay as relay
import numpy as np
import time
import json
from operator import attrgetter
import gc
import traceback
import copy

import random
from PIL import Image

from pathlib import Path
from modifiers.strategies import StrategiesModifier

from processors.model_preprocessor import ModelPreprocessor
from executors.objects import ObjectsExecutor
from evaluators.evalgenerator import EvaluationGenerator
from repairers.analyzer import Analyzer
from mutators.generators import model as model_generator

from scipy.special import softmax

class Repairer:

    def __init__(self, config):
        self.remote = config["remote"]
        self.build = config["build"]
        self.script_dir = config["script_dir"]
        self.images_folder = config["images_folder"]

        # Default settings.
        self.enable_transpose = True
        self.enable_dim_fix = True
        self.clear_multiple_outputs = True
        self.align_dimension = True
        self.enable_model_repair = True
        self.continue_repair_when_image_fixed = False
        self.enable_layer_analysis = False

        # Load options from config and override,
        # else use default values, if not present.
        if "settings" in config:
            settings = config["settings"]
            self.enable_transpose = settings["enable_transpose"] if "enable_transpose" in settings \
                else self.enable_transpose
            self.enable_dim_fix = settings["enable_dim_fix"] if "enable_dim_fix" in settings \
                else self.enable_dim_fix
            self.enable_transpose = settings["clear_multiple_outputs"] if "clear_multiple_outputs" in settings \
                else self.clear_multiple_outputs
            self.align_dimension = settings["align_dimension"] if "align_dimension" in settings \
                else self.align_dimension
            self.enable_model_repair = settings["enable_model_repair"] if "enable_model_repair" in settings \
                else self.enable_model_repair
            self.continue_repair_when_image_fixed = settings["continue_repair_when_image_fixed"] \
                if "continue_repair_when_image_fixed" in settings else self.continue_repair_when_image_fixed
            self.enable_layer_analysis = settings["enable_layer_analysis"] if "enable_layer_analysis" in settings \
                else self.enable_layer_analysis

        
        if not self.enable_layer_analysis:
            self.similar_sample = 1
            self.dissimilar_sample = 1
        else:
            # TODO: set values here.
            self.similar_sample = 15
            self.dissimilar_sample = 15

        self.timer_limit = 7200
        self.no_of_layer_iterations = 3
        self.n_jobs = -1
        self.evaluation_generator = EvaluationGenerator()
        self.modification_log = []
        # Note: this needs manual setup for now.
        # The system will attempt a fix on Transpose in case of problematic input dimensions.
        # If set false, the system just tries to fix the dimension without searching for a Transpose node.

        # The order of strategies is determined by potential effect in the layers.
        # First, we ho with the one related to inputs - symbolic dimensions.
        # Then we update the params and hyperparams, to go layer-wise.
        # Then we focus on graph differences across key layers.
        # Then we check flattening, with higher probability to happen around the end of graph.
        # Adjust weights if transpose is deprecated.
        self.strategies = ["params", "params[conv]", "graph[add]", "graph", "hyperparams", "flatten"]
        self.adjust_weights = False,
        self.default_strategies_config = {
            "graph[conv]" : {"op_types": ["Conv"] },
            "graph[add]" : {"op_types": ["Add"] },
            "hyperparams[conv]" : {"op_types": ["Conv"] },
            "params": {
                "dynamic_input_param_indexes": {
                    "Conv": [1, 2],
                    "Mul": [1],
                    "BatchNormalization": [1, 2, 3, 4],
                    "default": [1]
                }                    

            },
            "params[mul]" : {"op_types": ["Mul"], "input_param_indexes": [1]},
            "params[conv]" : {"op_types": ["Conv"], "input_param_indexes": [1, 2]},
            "params[batch]" : {"op_types": ["BatchNormalization"], "input_param_indexes": [1, 2, 3, 4]},
            "params[dequantize]": {"op_types": ["DequantizeLinear"], "input_param_indexes": [1, 2]}
        }

        self.ranked_layers_strategy = ["params", "graph", "hyperparams[conv]"]    

    def evaluate(self, source_run_obj, target_run_obj):
        return self.evaluation_generator.generate_objects_comparison(source_run_obj, target_run_obj)

    def execute_and_evaluate_single_model(self, onnx_model, run_obj, image_path, config, include_certainties=False):
        image_obj = self.execute_onnx_model(onnx_model, [image_path], config, print_percentage=False, include_certainties=include_certainties)
        image_name = list(image_obj.keys())[0]
        return self.evaluate(run_obj, image_obj)["images"][image_name]


    def execute_and_evaluate_single(self, onnx_path, run_obj, image_path, config, include_certainties=False):
        image_obj = self.execute_onnx_path(onnx_path, [image_path], config, print_percentage=False, include_certainties=include_certainties)
        image_name = list(image_obj.keys())[0]
        return self.evaluate(run_obj, image_obj)["images"][image_name]

    def evaluate_single(self, run_obj, image_obj):
        # Return single evaluation.
        image_name = list(image_obj.keys())[0]
        return self.evaluate(run_obj, image_obj)["images"][image_name]        
 
    def execute_single(self, onnx_path, image_path, config, include_certainties=False):
        # Execute and return for single image.
        return self.execute_onnx_path(onnx_path, [image_path], config, include_certainties)

    def execute_onnx_path(self, onnx_path, images_paths, config, image_names=None, print_percentage=True, include_certainties=False): 
        onnx_model = onnx.load(onnx_path)
        return self.execute_onnx_model(onnx_model, images_paths, config, image_names, print_percentage, include_certainties)

    def execute_onnx_model(self, onnx_model, images_paths, config, image_names=None, print_percentage=True, include_certainties=False):
        # Execute and return all data.
        self.preprocessing_data = {
            "input": config["input_shape"],
            "image_dimension": config["image_dimension"],
            "library": None
        }
        # Set library to None, so that the name is utilized for preprocessing selection.
        model_preprocessor = ModelPreprocessor(self.preprocessing_data)

        model = onnx_model
        ort_sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])

        output_data = {}
        count = 0
        images_length = len(images_paths)
        step = (images_length // 4) if images_length > 4 else images_length

        for img_path in images_paths:
            
            if(print_percentage and count % step == 0):
                print("Complete: " + str((count // step) * 25) + "%")

            count += 1
            img_name = Path(img_path).name
            img = Image.open(img_path)

            # Convert monochrome to RGB.
            if(img.mode == "L" or img.mode == "BGR"):
                img = img.convert("RGB")

            img = img.resize(config["image_dimension"])
            img = model_preprocessor.preprocess(config["models_data"]["model_name"], img, True)

            input_name = model.graph.input[0].name
            output = ort_sess.run(None, {input_name : img.astype(np.float32)})  

            # Note: Enable to check for multiple node outputs.
            #if len(output) > 1:
                #print("Problem: model has more than one outputs! Selecting the first one...")
            #    scores = softmax(output[0])
            #else:
            scores = softmax(output)
            if(len(scores) > 2):
                squeeze(scores, [0, 2])
            # elif(len(scores) == 2 and (scores[0] == scores[1]).all()):
            #     scores = scores[0]

            scores = np.squeeze(scores)
            ranks = np.argsort(scores)[::-1]
            # In case of a double output.
            if(len(ranks) == 2):
                ranks = ranks[0]
            extracted_ranks = ranks[0:5]

            # We do not consider probabilities for now
            if include_certainties:
                output_data[img_name] = [(rank, str(scores[rank])) for rank in extracted_ranks.tolist()]
            else:
                output_data[img_name] = extracted_ranks.tolist()
        return output_data

    def build_tvm(self, onnx_path, output_model_path, input_shape):
        onnx_model = onnx.load(onnx_path)
        input_name = onnx_model.graph.input[0].name
        # Input shape is preserved across models.
        shape_dict = {input_name: input_shape}
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
        target = tvm.target.Target(self.build["target"], host=self.build["host"])
        file_name = Path(onnx_path).stem
        
        file_path = model_generator.generate_original_model(mod, target, params, output_model_path, file_name=file_name, opt_level=3)
        return file_path

    # This is TVM model dependent.
    def execute_tvm_with_debug(self, models_data, images_data, specific_images):
        objectsExecutor = ObjectsExecutor(models_data, images_data, self.build)
        return objectsExecutor.execute(self.remote, specific_images)

    def get_layers_of_type(self, nodes, op_types):
        return [i for i in range(len([node for node in nodes if node.op_type in op_types]))]

    def repair_full_model(self, source_path, target_path, layers_to_repair, output_path):

        modifier = StrategiesModifier(source_path, target_path)
        modifier.load().\
            apply("graph", {"param_indexes": layers_to_repair}).\
            apply("hyperparams").\
            apply("params", {"op_types": ["Conv"], "input_param_indexes": [1, 2], "param_indexes": layers_to_repair}).\
            apply("params", {"op_types": ["BatchNormalization"], "input_param_indexes": [1, 2, 3, 4]}).\
            apply("symbolic_dimensions", {"param_indexes": layers_to_repair}).\
            save(output_path)
            # apply("flatten").\

    def repair(self, source_onnx_path, target_onnx_path, configuration):

        initial_target_onnx_path = target_onnx_path  
        folder_paths = [join(self.images_folder, f) for f in listdir(self.images_folder) \
            if isdir(join(self.images_folder, f))]

        images_paths_dir = {}

        images_paths = []
        if len(folder_paths) != 0:
            
            for folder in folder_paths:
                for f in listdir(folder):
                    images_paths.append(join(folder, f))
                    images_paths_dir[f] = folder
                
        else:
            images_paths = [join(self.images_folder, f) for f in listdir(self.images_folder) \
                if isfile(join(self.images_folder, f))]

        # Note: Use the target path, to avoid wrong caching for tests with the same base.
        target_stem = basename(target_onnx_path).split('.')[0]
        source_dir = self.script_dir + "/" + configuration["models_out_relative"] + "/source"
        makedirs(source_dir, exist_ok=True)
        repair_file_path = join(source_dir, target_stem + "_source_run.json")
        full_repair_file_path = join(source_dir, target_stem + "_repaired.json")
        full_repair_log_file_path = join(source_dir, target_stem + "_repaired_log.json")
        print("Checking for cached source run: " + repair_file_path)
        if not isfile(repair_file_path):
            print("Running dataset with source ONNX model...")
            source_object = self.execute_onnx_path(source_onnx_path, images_paths, configuration["source_onnx"])
            print("Caching run to: " + repair_file_path)
            # Serializing json...
            json_object = json.dumps(source_object, indent=2)
 
            # Writing json...
            with open(repair_file_path, "w") as outfile:
                outfile.write(json_object)

        else:
            print("Loading cached source ONNX model run...")
            with open(repair_file_path, "r") as f:
                source_object = json.loads(f.read())
            print("Run loaded successfully.")

        repair_log = {}
        dissimilarity_percentage = None
        dissimilar_explored = []

        target_version = 0

        start_timer = time.time()

        # TODO: Cache this, no need to rebuild each time.
        # Build TVM model for source.
        source_tvm_path = ""
        if self.enable_layer_analysis:
            try:
                source_tvm_path = self.build_tvm(source_onnx_path, configuration["models_out_relative"] + "/source/", \
                    configuration["source_onnx"]["input_shape"])
                
            except Exception as e:
                print("WARNING: TVM build failed for source. Disabling layer analysis...")
                self.enable_layer_analysis = False
        
        # Execute Source
        source_models_data = configuration["source_onnx"]["models_data"]
        source_models_data["input_model_folder"] = source_tvm_path
        source_models_data["output_model_folder"] = str(Path(source_tvm_path).parent.absolute())
        source_images_data = {
            "input_images_folders" : [self.images_folder],
            "output_images_base_folder": source_models_data["output_model_folder"] + "/images"
        }

        sub_debug_path = "onnx/debug/_tvmdbg_device_OPENCL_0"

        source_image_params_path = ""
        target_image_params_path = ""
        
        dissimilarity_value_check = 100
        same_dissimilarity_count = 0
        align_dimension = self.align_dimension
        adjust_weights = self.adjust_weights

        all_percentages = []
        repaired_modifications = 0
        total_modifications = 0

        while (self.continue_repair_when_image_fixed == True or dissimilarity_percentage != 0):      
            
            print ("Running target ONNX model...")
            if (len(source_object) != len(images_paths)):
                raise Exception("Size mismatch between source run (" + str(len(source_object)) + ") " +\
                "and target image paths (" + str(len(images_paths)) + ").\nCrashing to avoid having you wait without purpose :) - " +\
                "Check cached and current dataset.")

            target_object = self.execute_onnx_path(target_onnx_path, images_paths, configuration["target_onnx"])
            full_evaluation = self.evaluate(source_object, target_object)
            dissimilarity_percentage = full_evaluation["percentage_dissimilar"]

            all_percentages.append(dissimilarity_percentage)

            if (dissimilarity_value_check == dissimilarity_percentage):
                same_dissimilarity_count += 1
            else:
                same_dissimilarity_count = 1
            
            dissimilarity_value_check = dissimilarity_percentage

            curr_timer = time.time() - start_timer
            if  (dissimilarity_percentage == 0 and self.continue_repair_when_image_fixed == False) or \
                (dissimilarity_percentage == 0 and self.continue_repair_when_image_fixed == True and same_dissimilarity_count == 5) or \
                (same_dissimilarity_count == 6) or \
                (curr_timer >= self.timer_limit):     
                
                print("Model repair complete! Path: " + target_onnx_path)
                print("Total repaired modifications: " + str(repaired_modifications))
                print("Total modifications: " + str(total_modifications))
                print(all_percentages)
                print("---- Final dissimilarity percentage: " + str(dissimilarity_value_check) + "% ----")
                print("Performing full inference for source:")
                source_object = self.execute_onnx_path(source_onnx_path, images_paths, configuration["source_onnx"], include_certainties=True)
                print("Performing full inference for target:")
                target_object = self.execute_onnx_path(target_onnx_path, images_paths, configuration["target_onnx"], include_certainties=True)
                repaired_object = {}
                for key in source_object.keys():
                    repaired_object[key] = {
                        "source": source_object[key],
                        "target": target_object[key]
                    }
                
                json_object = json.dumps(repaired_object, indent=2)
                self.modification_log.append({
                    "repaired_modifications": str(repaired_modifications),
                    "execution_time": str(curr_timer),
                    "percentages": all_percentages
                })
                json_log_object = json.dumps(self.modification_log, indent=2)
    
                # Writing json
                with open(full_repair_file_path, "w") as outfile:
                    outfile.write(json_object)
                    print("Saved repaired output at " + full_repair_file_path) 

                with open(full_repair_log_file_path, "w") as outfile:
                    outfile.write(json_log_object)
                    print ("Saved repaired output modification log at " + full_repair_log_file_path)    
                break

            print("---- Current Dissimilarity Percentage -----: " + str(dissimilarity_percentage) + "%")
            #return
            # Note: We do not reconsider already analyzed images.
            # Also, we explicitly exclude similar images.
            dissimilar_images_full = [key for key in full_evaluation["dissimilar"] if key not in dissimilar_explored]
  
            # Sorting images will allow us get the value with lowest tau first (ascending order).
            dissimilar_images_full = sorted(dissimilar_images_full, key=lambda k : full_evaluation["images"][k]["tau"])
            
            if self.dissimilar_sample == -1:
                dissimilar_images_under_test = dissimilar_images_full
            else:
                if (len(dissimilar_images_full) < self.dissimilar_sample):
                    print ("Warning: dissimilar images length less than specified sample. Layer analysis will be inaccurate.")
                dissimilar_images_under_test = dissimilar_images_full[:self.dissimilar_sample]
            
            dissimilar_image_paths = []
            for d in dissimilar_images_under_test:
                folder_path = images_paths_dir[d] if d in images_paths_dir else self.images_folder
                dissimilar_image_paths.append(join(folder_path, d))

            # Consider top-K images the one as base for repair.
            dissimilar_image = dissimilar_images_under_test[0] if len(dissimilar_images_under_test) > 0 else full_evaluation["similar"][0]
            dissimilar_image_path = dissimilar_image_paths[0] if len(dissimilar_image_paths) > 0 else ""
            if dissimilar_image_path == "":
                print("No dissimilar images left. Skipping...")
                continue
            dissimilar_explored.append(dissimilar_image)

            target_tvm_path = ""
            if self.enable_layer_analysis:
                try:
                    target_tvm_path = self.build_tvm(target_onnx_path, configuration["models_out_relative"] + "/target/", \
                        configuration["target_onnx"]["input_shape"])
                except Exception as e:
                    print("WARNING: TVM build failed for target. Disabling layer analysis...")
                    self.enable_layer_analysis = False

            # For target, run all.
            target_models_data = configuration["target_onnx"]["models_data"]
            target_models_data["input_model_folder"] = target_tvm_path
            target_models_data["output_model_folder"] = str(Path(target_tvm_path).parent.absolute())
            target_images_data = {
                "input_images_folders" : [self.images_folder],
                "output_images_base_folder": target_models_data["output_model_folder"] + "/images"
            }

            # Caching...
            if target_image_params_path == "":
                target_image_params_path = join(target_images_data["output_images_base_folder"], sub_debug_path)
            if exists(target_image_params_path):
                for image_path in [join(target_image_params_path, i) for i in listdir(target_image_params_path)]:
                    #if image_path.endswith(".params"):
                    remove(image_path)

            if source_image_params_path == "":
                source_image_params_path = join(source_images_data["output_images_base_folder"], sub_debug_path)
            
            if exists(source_image_params_path):
                for image_params in [join(source_image_params_path, i) for i in listdir(source_image_params_path)]:
                    remove(image_params)

            # ------ Strategies without layer analysis --------

            # Transpose on-first-layer strategy
            if adjust_weights == True:
                adjust_weights = False
                # Repairs without requiring layer analysis.
                modifier = StrategiesModifier(source_onnx_path, target_onnx_path).load()
                tmp_target_path = target_onnx_path.replace(".onnx", "_adj_weights.onnx")
                modifier.apply("adjust_weights_if_transpose", custom_config={}).save(tmp_target_path)
                target_onnx_path = tmp_target_path
                dissimilar_evaluation = self.execute_and_evaluate_single(target_onnx_path, source_object, \
                        dissimilar_image_path, configuration["target_onnx"])
                total_modifications += modifier.get_overall_num_log_modifications()
                
                if dissimilar_evaluation["tau"] > 0.98:
                    print ("Weights adjustment fixed image!")

                    modifications_size = modifier.get_num_log_modifications()
                    repaired_modifications += modifications_size
                    self.modification_log.append(modifier.get_log_modifications())
                    continue
                
            if self.enable_model_repair and align_dimension and configuration["target_onnx"]["input_shape"] != configuration["source_onnx"]["input_shape"]:
                print("Warning: the dimensions of source model differ from the target. This might lead to different results.")
                configuration["target_onnx"]["input_shape"] = configuration["source_onnx"]["input_shape"] 
                configuration["target_onnx"]["image_dimension"] = configuration["source_onnx"]["image_dimension"]

                # Disable if run once.
                align_dimension = False
                # TODO: Refactor so that it works for any shape.
                # Repairs without requiring layer analysis.
                modifier = StrategiesModifier(source_onnx_path, target_onnx_path).load()
                tmp_target_path = target_onnx_path
                if self.enable_transpose == True:
                    print("Note: Repair will attempt transposition neutralization and fixing input dimension.")
                    strategy = "transpose"
                    tmp_target_path = tmp_target_path.replace(".onnx", "_transpose.onnx")
                    if "transpose_order" in configuration["target_onnx"]:
                        custom_config={"order": configuration["target_onnx"]["transpose_order"]}
                    else:
                        custom_config={}
                    
                        
                    modifier.apply(strategy, custom_config=custom_config)
                

                if self.enable_dim_fix == True:
                    print("Note: Repair will attempt repairing input dimension.")
                    strategy = "repair_input_dimension"
                    tmp_target_path = tmp_target_path.replace(".onnx", "_repair_input_dimension.onnx")
                    custom_config={"param_indexes": [0]}
                    modifier.apply(strategy, custom_config=custom_config)


                # When converted from source to target and dimension has changed,
                # a transposition layer is inserted to the model.
                # We neutralize it in that case.
                modifier.apply(strategy, custom_config=custom_config).save(tmp_target_path)
                total_modifications += modifier.get_overall_num_log_modifications()
                target_onnx_path = tmp_target_path
                dissimilar_evaluation = self.execute_and_evaluate_single(target_onnx_path, source_object, \
                        dissimilar_image_path, configuration["target_onnx"])

                print("Dissimilar Image Tau: " + str(dissimilar_evaluation["tau"]))
                # print(target_onnx_path)
                if dissimilar_evaluation["tau"] > 0.98:
                    print ("Dimension repair fixed image!")

                    modifications_size = modifier.get_num_log_modifications()
                    repaired_modifications += modifications_size
                    self.modification_log.append(modifier.get_log_modifications())
                    continue

            # If not enabled, this will run for just 1 similar/dissimilar image.
            if self.enable_layer_analysis:
                self.execute_tvm_with_debug(source_models_data, source_images_data, dissimilar_images_under_test)
                self.execute_tvm_with_debug(target_models_data, target_images_data, dissimilar_images_under_test)

            layer_config = {
                "layer_analysis_data": {
                    "debug_path1": str(Path(source_tvm_path).parent.absolute()) + "/images/onnx/debug/_tvmdbg_device_OPENCL_0/",
                    "debug_path2": str(Path(target_tvm_path).parent.absolute()) + "/images/onnx/debug/_tvmdbg_device_OPENCL_0/",
                    "lib1_graph_params": source_tvm_path.replace(".tar", "_lib_params.params"),
                    "lib2_graph_params": target_tvm_path.replace(".tar", "_lib_params.params"),
                }
            }

            total_layer_data = {}
            
            # Analysis Phase.
            similar_images = full_evaluation["similar"].copy()

            layers_iterations = []
            source_onnx_model_nodes = onnx.load(source_onnx_path).graph.node
            
            # self.default_strategies_config["params[mul]"]["param_indexes"] = all_layers
            # self.default_strategies_config["params[batch]"]["param_indexes"] = all_layers
            # self.default_strategies_config["params[dequantize]"]["param_indexes"] = all_layers          

            if self.enable_layer_analysis:

                for it in range(self.no_of_layer_iterations):
                    
                    random.shuffle(similar_images)
                    if self.similar_sample == -1 or self.similar_sample >= len(similar_images):
                        similar_images_under_test = similar_images
                    else:
                        similar_images_under_test = similar_images[:self.similar_sample]

                    # Remove selected images from subsequent selections.
                    similar_images = [image for image in similar_images if image not in similar_images_under_test]

                    self.execute_tvm_with_debug(source_models_data, source_images_data, similar_images_under_test)
                    self.execute_tvm_with_debug(target_models_data, target_images_data, similar_images_under_test)
                    
                    print("---- Current Dissimilarity Percentage -----: " + str(dissimilarity_percentage) + "%")
                    # Perform analysis to detect responsible layers.
                    analyzer = Analyzer(self.script_dir, n_jobs=self.n_jobs)
                    layers_no = analyzer.get_target_nodes_length(layer_config)
                    layers_range_list = [i for i in range(layers_no)]

                    if len(dissimilar_images_full) >= self.dissimilar_sample:
                        # Note: the analyzer concerns crucial layers (Conv2D) for the analysis.
                        layers_iteration = analyzer.perform_fault_localization(layer_config, similar_images_under_test, dissimilar_images_under_test, \
                        remove_similar_upon_completion=True)
                        # Use to store data in original order.
                        layers_iterations.append(layers_iteration)
                        layers_iteration.sort(key=lambda l: abs(l["elems_percentage"]), reverse=True)
                        layers_iteration = [e["layer_index"] for e in layers_iteration]
                        print ("Partial Layers Order: " + str(layers_iteration))
                        
                        for order in range(len(layers_iteration)):
                            layer = layers_iteration[order]
                            if layer not in total_layer_data:
                                total_layer_data[layer] = 0
                            total_layer_data[layer] += order
                    else:
                        # If no dissimilar, apply fix to all layers, and do not repeat check.
                        print("Not enough dissimilar images, assuming default layer order...")
                        layers = layers_range_list
                        break
            else:
                # Utilize ONNX in order to infer layer number.
                print("Layer analysis disabled. Using original layer order.")
                layers = self.get_layers_of_type(source_onnx_model_nodes, "Conv")

            if self.enable_layer_analysis and len(dissimilar_images_full) >= self.dissimilar_sample:
                layers = sorted(total_layer_data, key=total_layer_data.get)

                # Export total analysis outcome.
                json_object = json.dumps(str(layers_iterations), indent=2)
                print("Saving analysis to file...")
                with open(target_onnx_path.replace(".onnx", "_layer_analysis_" + str(time.time()) + ".json"), "w") as outfile:
                    outfile.write(json_object)

            print("Final Layer Order:" + str(layers))

            dissimilar_evaluation = {
                "tau": -1
            }
            
            has_fixed_hyperparams = False
            best_target_path = target_onnx_path
               
            if not self.enable_model_repair:

                print("Model repair is disabled. Exiting...")
                break

            modifier = None
            
            if self.clear_multiple_outputs:
                modifier = StrategiesModifier(source_onnx_path, target_onnx_path).load()
                tmp_target_path = target_onnx_path.replace(".onnx", "_clear_multiple_outputs.onnx")
                modifier.apply("clear_multiple_outputs", custom_config={}).save(tmp_target_path)
                target_onnx_path = tmp_target_path
                dissimilar_evaluation = self.execute_and_evaluate_single(target_onnx_path, source_object, \
                        dissimilar_image_path, configuration["target_onnx"])
                total_modifications += modifier.get_overall_num_log_modifications()
                
                if dissimilar_evaluation["tau"] > 0.98:
                    print ("Output clearance fixed image!")

                    modifications_size = modifier.get_num_log_modifications()
                    repaired_modifications += modifications_size
                    self.modification_log.append(modifier.get_log_modifications())
                    continue


            for strategy in self.strategies:
                
                print("Attempting strategy: " + strategy)

                # Used to allow specifying crucial layer type.
                strategy_type = strategy.split("[")[0]

                # TODO: Refactor.
                # Deepcopy, as you later edit strategy object.
                strategy_config = copy.deepcopy(self.default_strategies_config[strategy] if strategy in self.default_strategies_config else {})
                
                if strategy not in self.default_strategies_config:
                    layers_iter = layers
                elif "param_indexes" not in self.default_strategies_config[strategy]:
                    if strategy in self.ranked_layers_strategy:
                        layers_iter = layers
                    else:
                        layers_iter = self.get_layers_of_type(source_onnx_model_nodes, self.default_strategies_config[strategy]["op_types"])  
                else:
                    layers_iter = self.default_strategies_config[strategy]["param_indexes"]
                    print(layers_iter)

                if modifier is not None:
                    total_modifications += modifier.get_overall_num_log_modifications()
                    repaired_modifications += modifier.get_num_log_modifications()
                    self.modification_log.append(modifier.get_log_modifications())

                modifier = StrategiesModifier(source_onnx_path, best_target_path).load()
                
                for layer in layers_iter:
                    tmp_target_path = best_target_path.split("[")[0] + "[" + str(time.time()) + "].onnx"
                
                    # Heavy memory handling and cleanup happens here, so we should call GC.
                    gc.collect()
                    
                    # Note for params:
                    # Since eventually all layers will be accessed,
                    # we keep params running per-layer, to have a better image of the
                    # repair process and the effect of each layer.
                    # Therefore, we use them as a normal strategy. 
                    strategy_config["param_indexes"] = [layer]
                    onnx_model_backup = copy.deepcopy(modifier.get_target())
                    prev_num_log_mods = modifier.get_num_log_modifications()
                    
                    try:
                        result = modifier.apply(strategy_type, custom_config=strategy_config)
                        # modifier.save(tmp_target_path)
                        # break
                        if result == -1:
                            break

                        # elif result == -2:
                        #     continue

                        # There are many cases, where a graph change might be associated with a hyperaparameter
                        # to achieve proper dimension. For that reason, attempt a check on the node under test
                        # for necessary hyperparameter adjustments.
                        if strategy_type == "graph":
                            modifier.apply("hyperparams", custom_config=strategy_config)
                            
                        
                        onnx_model = modifier.get_target()
                        # save(tmp_target_path)
                        
                        # TODO: Restore later.
                        # modifications_size = modifier.get_num_log_modifications()
                        # if (modifications_size != 0):
                        #     repaired_modifications += modifications_size
                        #     self.modification_log.append(modifier.get_log_modifications())
                        dissimilar_evaluation = self.execute_and_evaluate_single_model(onnx_model, source_object, \
                            dissimilar_image_path, configuration["target_onnx"])

                    except Exception as e:
                        print("Conversion failed. Rolling back...")
                        print(e)
                        print(traceback.format_exc())
                        
                        if strategy_type == "graph":
                            modifier.revert(onnx_model_backup, prev_num_log_mods)
                        else:
                            modifier.revert_last()

                        if exists(tmp_target_path):
                            remove(tmp_target_path)
                        continue

                    # Name for next best path.
                    #tmp_target_path = best_target_path.split("[")[0] + "[" + str(time.time()) + "].onnx"

                    print ("Dissimilar Image Tau: " + str(dissimilar_evaluation["tau"]))
                    if (dissimilar_evaluation["tau"] >= 0.98 and not self.continue_repair_when_image_fixed):
                        
                        modifications_size = modifier.get_num_log_modifications()
                        if (modifications_size != 0):
                            repaired_modifications += modifications_size
                            self.modification_log.append(modifier.get_log_modifications())
                            modifier.save(tmp_target_path)
                            best_target_path = tmp_target_path
                            
                        break

                    if (dissimilar_evaluation["tau"] >= full_evaluation["images"][dissimilar_image]["tau"]):
                    #dissimilar_evaluation["tau"] > 0):

                        if (dissimilar_evaluation["tau"] > full_evaluation["images"][dissimilar_image]["tau"]):
                            print("Fix improved dissimilar image evaluation!")
                            modifier.save(tmp_target_path)
                        
                            # Place to restore
                            if initial_target_onnx_path != best_target_path:
                                remove(best_target_path)

                            best_target_path = tmp_target_path
                        
                            full_evaluation["images"][dissimilar_image]["tau"] = dissimilar_evaluation["tau"]
                        #best_target_path = tmp_target_path
                    else:
                        print("Change degraded performance. Reverting...")
                        if strategy_type == "graph":
                            modifier.revert(onnx_model_backup)
                        else:
                            modifier.revert_last()
                    #     remove(tmp_target_path)

                if (dissimilar_evaluation["tau"] >= full_evaluation["images"][dissimilar_image]["tau"]):
                    tmp_target_path = best_target_path.split("[")[0] + "[" + str(time.time()) + "].onnx"
                    modifier.save(tmp_target_path)

                    best_target_path = tmp_target_path
                    target_onnx_path = best_target_path

                    if (dissimilar_evaluation["tau"] >= 0.98 and not self.continue_repair_when_image_fixed):
                        print("Image fixed!")
                        total_modifications += modifier.get_overall_num_log_modifications()
                        break
