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
        self.enable_layer_analysis = config["settings"]["enable_layer_analysis"] if "settings" in config else False
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
        self.enable_transpose = True
        self.align_dimension = True
        self.enable_model_repair = True
        self.continue_repair_when_image_fixed = False

        # The order of strategies is determined by potential effect in the layers.
        # First, we ho with the one related to inputs - symbolic dimensions.
        # Then we update the params and hyperparams, to go layer-wise.
        # Then we focus on graph differences across key layers.
        # Then we check flattening, with higher probability to happen around the end of graph.
        # Adjust weights if transpose is deprecated.

        # Try different orders.
        # FIND a rationale for the order. try partial order multiple to prove that it does not matter.
        self.strategies = ["params[conv]"] #"symbolic_dimensions", "params[dequantize]", "params[conv]", "params[batch]", "params", "graph", "hyperparams", "flatten"
        self.adjust_weights = False,
        self.strategies_config = {
            "params[conv]" : {"op_types": ["Conv"], "input_param_indexes": [1, 2]},
            "params[batch]" : {"op_types": ["BatchNormalization"], "input_param_indexes": [1, 2, 3, 4]},
            "params[dequantize]": {"op_types": ["DequantizeLinear"], "input_param_indexes": [1, 2]}
        }    

    def evaluate(self, source_run_obj, target_run_obj):
        return self.evaluation_generator.generate_objects_comparison(source_run_obj, target_run_obj)

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
        # Execute and return all data.
        # images_paths = [f for f in listdir(self.images_folder) if isfile(join(self.images_folder, f))]
        self.preprocessing_data = {
            "input": config["input_shape"],
            "image_dimension": config["image_dimension"],
            "library": None
        }
        # Set library to None, so that the name is utilized for preprocessing selection.
        model_preprocessor = ModelPreprocessor(self.preprocessing_data)

        model = onnx.load(onnx_path)
        ort_sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

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
            model_name = Path(onnx_path).name
            img = model_preprocessor.preprocess(config["models_data"]["model_name"], img, True)

            input_name = model.graph.input[0].name
            output = ort_sess.run(None, {input_name : img.astype(np.float32)})

            scores = softmax(output)
            if(len(scores) > 2):
                squeeze(scores, [0, 2])

            scores = np.squeeze(scores)
            ranks = np.argsort(scores)[::-1]
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
        print(source_onnx_path)
        print(target_onnx_path)

        initial_target_onnx_path = target_onnx_path  
        folder_paths = [join(self.images_folder, f) for f in listdir(self.images_folder) \
            if isdir(join(self.images_folder, f))]

        print(folder_paths)
        images_paths_dir = {}

        images_paths = []
        if len(folder_paths) != 0:
            
            for folder in folder_paths:
                for f in listdir(folder):
                    images_paths.append(join(folder, f))
                    images_paths_dir[f] = folder

                # images_paths.extend([join(folder, f)  \
                # if isfile(join(folder, f))])
                
        else:
            images_paths = [join(self.images_folder, f) for f in listdir(self.images_folder) \
                if isfile(join(self.images_folder, f))]
        
        print(len(images_paths))

        # Note: Use the target path, to avoid wrong caching for tests with the same base.
        target_stem = basename(target_onnx_path).split('.')[0]
        source_dir = self.script_dir + "/" + configuration["models_out_relative"] + "/source"
        makedirs(source_dir, exist_ok=True)
        repair_file_path = join(source_dir, target_stem + "_source_run.json")
        full_repair_file_path = join(source_dir, target_stem + "_repaired.json")
        full_repair_log_file_path = join(source_dir, target_stem + "_repaired_log.json")
        
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

        # count_top1 = 0
        # Calculate accuracy over ground truths.
        # for img in source_object:

        #     path_for_img = images_paths_dir[img]
        #     last_dir = basename(normpath(path_for_img))
        #     top_1_prediction = str(source_object[img][0])

        #     # print(last_dir)
        #     # print(top_1_prediction)

        #     if last_dir == top_1_prediction:
        #         count_top1 += 1
        # print("Top-1 accuracy for source: " + str(count_top1/len(images_paths_dir)*100) + "%")
        # InceptionV3:
        # Top5: 92.84 over 93.9
        # Top1: 75.32 over 78.1
        
        prev_target_onnx_path = target_onnx_path
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
        total_modifications = 0

        while (self.continue_repair_when_image_fixed == True or dissimilarity_percentage != 0):      
            
            print ("Running target ONNX model...")
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
                (dissimilarity_percentage == 0 and self.continue_repair_when_image_fixed == True and same_dissimilarity_count == 2) or \
                (same_dissimilarity_count == 2) or \
                (curr_timer >= self.timer_limit):     
                
                print("Model repair complete! Path: " + target_onnx_path)
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
                    "total_modifications": str(total_modifications),
                    "execution_time": str(curr_timer)
                })
                json_log_object = json.dumps(self.modification_log, indent=2)
    
                # Writing json
                with open(full_repair_file_path, "w") as outfile:
                    outfile.write(json_object)
                    print ("Saved repaired output at " + full_repair_file_path) 

                with open(full_repair_log_file_path, "w") as outfile:
                    outfile.write(json_log_object)
                    print ("Saved repaired output modification log at " + full_repair_log_file_path)    
                break

            print("---- Current Dissimilarity Percentage -----: " + str(dissimilarity_percentage) + "%")

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
            dissimilar_image_path = dissimilar_image_paths[0] if len(dissimilar_image_paths) > 0 else join(folder_path, full_evaluation["similar"][0])

            dissimilar_explored.append(dissimilar_image)

            target_tvm_path = ""
            if self.enable_layer_analysis:
                try:
                    target_tvm_path = self.build_tvm(target_onnx_path, configuration["models_out_relative"] + "/target/", \
                        configuration["target_onnx"]["input_shape"])
                except Exception as e:
                    print("WARNING: TVM build failed for targer. Disabling layer analysis...")
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
                
                if dissimilar_evaluation["tau"] > 0.98:
                    print ("Weights adjustment fixed image!")

                    modifications_size = modifier.get_num_log_modifications()
                    if (modifications_size != 0):
                        total_modifications += modifications_size
                        self.modification_log.append(modifier.get_log_modifications())
                    continue
                
            if align_dimension == True and configuration["target_onnx"]["input_shape"] != configuration["source_onnx"]["input_shape"]:
                print("Warning: the dimensions of source model differ from the target. This might lead to different results.")
                configuration["target_onnx"]["input_shape"] = configuration["source_onnx"]["input_shape"] 
                configuration["target_onnx"]["image_dimension"] = configuration["source_onnx"]["image_dimension"]
                
                # Disable if run once.
                align_dimension = False
                # TODO: Refactor so that it works for any shape.
                if self.enable_transpose == True:
                    print("Note: Repair will attempt transposition neutralization and fixing input dimension.")
                    strategy = "transpose"
                    tmp_target_path = target_onnx_path.replace(".onnx", "_transpose.onnx")
                    custom_config={}
                else:
                    print("Note: Repair will attempt repairing input dimension.")
                    strategy = "repair_input_dimension"
                    tmp_target_path = target_onnx_path.replace(".onnx", "_repair_input_dimension.onnx")
                    custom_config={"param_indexes": [0]}

                # Repairs without requiring layer analysis.
                modifier = StrategiesModifier(source_onnx_path, target_onnx_path).load()
                #tmp_target_path = target_onnx_path.replace(".onnx", "_transpose.onnx")
                # When converted from source to target and dimension has changed,
                # a transposition layer is inserted to the model.
                # We neutralize it in that case.
                modifier.apply(strategy, custom_config=custom_config).save(tmp_target_path)
                target_onnx_path = tmp_target_path
                dissimilar_evaluation = self.execute_and_evaluate_single(target_onnx_path, source_object, \
                        dissimilar_image_path, configuration["target_onnx"])

                print("Dissimilar Image Tau: " + str(dissimilar_evaluation["tau"]))
                print(target_onnx_path)
                if dissimilar_evaluation["tau"] > 0.98:
                    print ("Dimension repair fixed image!")

                    modifications_size = modifier.get_num_log_modifications()
                    if (modifications_size != 0):
                        total_modifications += modifications_size
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
                print("Layer analysis disabled. Using original layer order.")
                analyzer = Analyzer(self.script_dir, n_jobs=self.n_jobs)
                source_onnx_model_nodes = onnx.load(source_onnx_path).graph.node
                layers_no = 0
                for node in source_onnx_model_nodes:
                    print(node.name)
                    if node.name.startswith("Conv"):
                        layers_no += 1

                print("Layers No: " + str(layers_no))
                #analyzer.get_target_nodes_length(layer_config)
                layers = [i for i in range(layers_no)]


            if self.enable_layer_analysis and len(dissimilar_images_full) >= self.dissimilar_sample:
                # print(total_layer_data)
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
            best_target_path = target_onnx_path.replace(".onnx", "[.onnx")
               
            if not self.enable_model_repair:

                print("Model repair is disabled. Exiting...")
                break


            for strategy in self.strategies:

                # Used to allow specifying crucial layer type.
                strategy_type = strategy.split("[")[0]
                strategy_config = self.strategies_config[strategy] if strategy in self.strategies_config else {}
                
                for layer in layers:
                    # Heavy memory handling and cleanup happens here, so we should call GC.
                    gc.collect()

                    strategy_in_layer = strategy + "_" + str(layer)

                    tmp_target_path = best_target_path.split("[")[0] + "[" + str(time.time()) + "].onnx"

                    # Restore file name for first use.
                    if best_target_path.endswith("[.onnx"):
                        best_target_path = best_target_path.replace("[.onnx", ".onnx")

                    # Note for params:
                    # Since eventually all layers will be accessed,
                    # we keep params running per-layer, to have a better image of the
                    # repair process and the effect of each layer.
                    # Therefore, we use them as a normal strategy.
                    
                    if (strategy != "flatten" and "batch" not in strategy):
                        # Note: batch analysis will run for all batch layers in order of appearance.
                        # The layer variable holds the Conv2D layers sorted.
                        # Therefore this is skipped for batch elements, which are taken by default order.
                        strategy_config["param_indexes"] = [layer]
                    
                    try:

                        modifier = StrategiesModifier(source_onnx_path, best_target_path).load()
                        modifier.apply(strategy_type, custom_config=strategy_config)

                        # There are many cases, where a graph change might be associated with a hyperaparameter
                        # to achieve proper dimension. For that reason, attempt a check on the node under test
                        # for necessary hyperparameter adjustments.
                        if strategy_type == "graph":
                            modifier.apply("hyperparams", custom_config=strategy_config)
                        modifier.save(tmp_target_path)
                        
                        # Restore later.
                        modifications_size = modifier.get_num_log_modifications()
                        if (modifications_size != 0):
                            total_modifications += modifications_size
                            self.modification_log.append(modifier.get_log_modifications())
                        dissimilar_evaluation = self.execute_and_evaluate_single(tmp_target_path, source_object, \
                            dissimilar_image_path, configuration["target_onnx"])

                    except Exception as e:
                        print("Conversion failed. Rolling back...")
                        print(e)
                        if exists(tmp_target_path):
                            remove(tmp_target_path)
                        continue

                    print ("Dissimilar Image Tau, Layer:" + str(layer) + ": " + str(dissimilar_evaluation["tau"]))
                    if (dissimilar_evaluation["tau"] >= 0.98 and not self.continue_repair_when_image_fixed):
                        best_target_path = tmp_target_path

                        modifications_size = modifier.get_num_log_modifications()
                        if (modifications_size != 0):
                            total_modifications += modifications_size
                            self.modification_log.append(modifier.get_log_modifications())
                        break

                    if (dissimilar_evaluation["tau"] >= full_evaluation["images"][dissimilar_image]["tau"]):
                        if (dissimilar_evaluation["tau"] > full_evaluation["images"][dissimilar_image]["tau"]):
                            print("Fix improved dissimilar image evaluation!")

                        
                        # Place to restore

                        if initial_target_onnx_path != best_target_path:
                            remove(best_target_path)
                        
                        full_evaluation["images"][dissimilar_image]["tau"] = dissimilar_evaluation["tau"]
                        best_target_path = tmp_target_path
                    else:
                        remove(tmp_target_path)

                if (dissimilar_evaluation["tau"] >= full_evaluation["images"][dissimilar_image]["tau"]):
                    target_onnx_path = best_target_path

                    if (dissimilar_evaluation["tau"] >= 0.98 and not self.continue_repair_when_image_fixed):
                        print("Image fixed!")
                        break


                