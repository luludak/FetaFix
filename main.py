# Note: Before running, always make sure to add proper TVM vars to your environment:
# export TVM_HOME=/home/nickl/PhD/tvm
# export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}

import copy
import time
import math
import os
import onnx
from onnx import numpy_helper
import numpy as np
import json
import traceback

from os import path
from os import listdir
from os.path import isfile, isdir, join

import torch

import tvm
import tvm.relay as relay
from tvm.contrib.download import download_testdata
from tvm import rpc
from tvm.relay import transform
from tvm.relay.dataflow_pattern import rewrite

from scipy.special import softmax

from mutators.generators import model as model_generator
from executors import mutations, libraries

from evaluators.evalgenerator import EvaluationGenerator
from loaders.model_loader import ModelLoader
from loaders.model_exporter import ModelExporter

import numpy as np
from numpy.linalg import norm

import seaborn as sns
import matplotlib.pylab as plt
import matplotlib.ticker as mtick
import pandas as pd

from modifiers.strategies import StrategiesModifier
from repairers.onnxrepair import Repairer

import re

script_dir = os.path.dirname(os.path.realpath(__file__))

def quantize(mod, params):
    with relay.quantize.qconfig(calibrate_mode="global_scale", global_scale=8.0):
        mod = relay.quantize.quantize(mod, params)
    return mod

def props(cls):
  return [i for i in cls.__dict__.keys() if i[:1] != '_']

def load_config():
    with open('./config.json') as f:
        return json.load(f)

def load_weights(path):

    onnx_model   = onnx.load(path)
    print ("Path to load: " + path)
    INTIALIZERS  = onnx_model.graph.initializer
    onnx_weights = {}
    for initializer in INTIALIZERS:
        W = numpy_helper.to_array(initializer)
        onnx_weights[initializer.name] = {"weights": W, "dims": initializer.dims}
    return onnx_weights

config = load_config()
model_loader = ModelLoader()
model_exporter = ModelExporter()

# Common to all models configuration
device_name = config["devices"]["selected"]
build = config["devices"][device_name]
build["device_name"] = device_name
build["id"] = config["devices"]["id"]

# ----- Setup target configuration -----
target = tvm.target.Target(build["target"], host=build["host"])

# Prepare Device
host_type = build["host_type"]
device_id = build["id"]
if(host_type == "local_no_rpc"):
    remote = None
elif (host_type == "local"):
    print ("Preparing using Local RPC connection. - Device ID: " + str(device_id))
    remote = rpc.LocalSession()
else:
    address = build["address"]
    port = build["port"]
    print ("Preparing on : " + address + ":" + str(port) + " - Device ID: " + str(device_id))
    remote = rpc.connect(address, port)

datasets_info = config["datasets"]
default_dataset_info = datasets_info[config["selected_dataset"]]

images_path = script_dir + "/" + default_dataset_info["dataset_path_relative"]
image_names  = [f for f in listdir(images_path) if isfile(join(images_path, f))]

evaluation_base_folder = "/mutations/ts_full/"
evaluation_path = evaluation_base_folder + "/evaluate_mutation.txt"
evaluation_single_device_path = evaluation_base_folder + "/evaluate_single_device_mutation.txt"

mutation_model_evaluations = {}
model_names = []

opt_level = "opt" + str(config["opt_level"])
print ("Preprocessing is " + ("enabled." if config["preprocessing_enabled"] else "disabled."))

#------------------------- Localization Analysis -------------------------
if(config["model_repair_config"]["settings"]["enable_model_repair"]):
    build["error_base_folder"] = join(script_dir, "error_log", "repair")

    source_path = script_dir + config["model_repair_config"]["source_rel_path"]
    target_path = script_dir + config["model_repair_config"]["target_rel_path"]

    source_model_data = config["model_repair_config"]["source_model_data"]
    target_model_data = config["model_repair_config"]["target_model_data"]
    

    repairer = Repairer({
        "remote": remote,
        "images_folder": images_path,
        "build": build,
        "script_dir": script_dir,
        "settings": config["model_repair_config"]["settings"]
    })

    print(config["model_repair_config"]["settings"])
    repairer.repair(source_path, target_path, {
        "source_onnx": {
            "input_shape": source_model_data["input"],
            "image_dimension": source_model_data["image_dimension"],
            "models_data": source_model_data
        },
        "target_onnx": {
            "input_shape": target_model_data["input"],
            "image_dimension": target_model_data["image_dimension"],
            "transpose_order": target_model_data["transpose_order"] if "transpose_order" in target_model_data else None,
            "models_data" : target_model_data
        },
        "models_out_relative": config["model_repair_config"]["out_rel_path"]
    })

#------------------------- Models Processing -------------------------

for loop_count in range(config["runs_no"]):
    print("Run " + str(loop_count + 1) + " out of " + str(config["runs_no"]) + ".")
    for model_info in config["models"]:

        if("skip_analysis" in model_info and model_info["skip_analysis"]):
            print("Skipping analysis on " + model_info["alias"])
            continue

        print("Processing model " + model_info["alias"])

        model_url = model_info["url"]
        input_name = model_info["input_name"]
        output_name = model_info["output_name"] if "output_name" in model_info else None
        paths_info = model_info["paths"]
        models_path = script_dir + "/" + paths_info["models_out_relative"]
        generated_path = script_dir + "/generated/"

        input_layer_shape = tuple(model_info["layers"]["input"])
        x = np.zeros(shape=input_layer_shape)
        shape_dict = {input_name: x.shape}

        out_path = script_dir + paths_info["exec_out_relative"]
        dll_out_path = script_dir + paths_info["dll_exec_out_relative"]
        error_base_folder = join(script_dir, "error_log", model_info["name"])
        build["error_base_folder"] = error_base_folder

        models_data = {
            "name": model_info["name"],
            "model": model_info["model"],
            "model_name": model_info["name"],
            "raw_model_name": model_info["name"],
            "input_model_folder": models_path,
            "output_model_folder": models_path,
            "image_dimension": tuple(model_info["layers"]["image_dimension"]),
            "input": tuple(model_info["layers"]["input"]),
            "input_name": input_name,
            "output": tuple(model_info["layers"]["output"]),
            "output_name": output_name,
            "library": model_info["library"] if "library" in model_info else "unknown",
            "preprocessing_enabled": config["preprocessing_enabled"],
            "debug_enabled": config["debug_enabled"],
            "dll_models_path": models_path,
            "dtype": model_info["dtype"] if "dtype" in model_info else "float32",
            "opset": model_info["opset"] if "opset" in model_info else 11
        }

        # ----- Model building & mutations phase -----
        if(loop_count == 0 and ("build" in model_info and model_info["build"])):
            print("Building: " + model_info["name"])
            os.makedirs(models_path, exist_ok=True)

            # ----- Model Download and Loading Phase -----
            if ("type" in model_info and model_info["type"] == "library"):
                mod, params = model_loader.load_model(models_data)
            else:
                if ("type" in model_info and model_info["type"] == "remote"):
                    model_path = script_dir + "/" + model_url
                else:
                    model_path = download_testdata(model_url, model_info["alias"], module=model_info["type"])
                onnx_model = onnx.load(model_path)
                mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
                
            if("quantize" in config and config["quantize"]):
                print("Quantization enabled.")
                mod = quantize(mod, params)

            model_generator.generate_original_model(mod, target, params, paths_info["models_out_relative"], opt_level=config["opt_level"], required_pass=config["required_pass"], disabled_pass=config["disabled_pass"], quantize=config["quantize"], opt_alias=config["opt_alias"])
            
            if ("enable_mutations" in config and config["enable_mutations"]):
                if ("positions" in mutations_info):
                    info = mutations_info["positions"][model_info["name"]]
                    relay_positions = info["relay"]
                    for relay_position in relay_positions:
                        for mutation in mutations_info["relay"]:
                            mutation["start"] = relay_position
                            mutation["end"] = relay_position
                        model_generator.generate_relay_mutations(mod, target, params, mutations_info["relay"], paths_info["models_out_relative"], opt_level=config["opt_level"], required_pass=config["required_pass"], disabled_pass=config["disabled_pass"], quantize=config["quantize"], opt_alias=config["opt_alias"])
                        
                    tir_positions = info["tir"]
                    for tir_position in tir_positions:
                        for mutation in mutations_info["tir"]:
                            mutation["start"] = tir_position
                            mutation["end"] = tir_position
                        model_generator.generate_tir_mutations(mod, target, params, mutations_info["tir"], paths_info["models_out_relative"], opt_level=config["opt_level"], required_pass=config["required_pass"], disabled_pass=config["disabled_pass"], quantize=config["quantize"], opt_alias=config["opt_alias"])
            
                else:
                    model_generator.generate_relay_mutations(mod, target, params, mutations_info["relay"], paths_info["models_out_relative"], opt_level=config["opt_level"], required_pass=config["required_pass"], disabled_pass=config["disabled_pass"], quantize=config["quantize"], opt_alias=config["opt_alias"])
                    model_generator.generate_tir_mutations(mod, target, params, mutations_info["tir"], paths_info["models_out_relative"], opt_level=config["opt_level"], required_pass=config["required_pass"], disabled_pass=config["disabled_pass"], quantize=config["quantize"], opt_alias=config["opt_alias"])

        mutations_names = []
        if (path.exists(models_path)):
            mutations_names = [f for f in listdir(models_path) if isfile(join(models_path, f)) and f.endswith(".tar") and ("_ignore_" not in f)]

        mutations_names.sort(key=lambda x: "original" in x, reverse=True)
        print("Mutations generated:")
        print(mutations_names)
        models_data["mutations_names"] = mutations_names
        dll_models_path = script_dir + "/" + paths_info["dll_models_out_relative"]
        dll_libs = model_info["dll_libraries"]
        
        if(loop_count == 0 and ("build_dlls" in model_info and model_info["build_dlls"])):
            
            print("Building DL models for " + model_info["name"])
            os.makedirs(dll_models_path, exist_ok=True)
            
            for dll_lib in dll_libs:
                if("noop" not in dll_lib["library"]):
                    dll_lib["script_dir"] = script_dir
                    dll_lib["name"] = model_info["name"]
                    
                    # TODO: Refactor.
                    dll_lib["model_name"] = dll_lib["dependency"]
                    dll_lib["dtype"] = model_info["dtype"] if "dtype" in model_info else "float32"
                    dll_lib["dll_models_path"] = dll_models_path

                    dll_model, dll_params = model_loader.load_model(dll_lib)
                    print("MODEL LOADED!")


                    if("quantize" in config and config["quantize"]):
                        print("Quantization enabled.")
                        dll_model = quantize(dll_model, dll_params)
                    if(dll_model != None):
                        model_generator.generate_original_model(dll_model, target, dll_params, paths_info["dll_models_out_relative"], (dll_lib["model"] + "_" + dll_lib["library"]).replace(".", "_"), opt_level=config["opt_level"], required_pass=config["required_pass"], disabled_pass=config["disabled_pass"], quantize=config["quantize"], opt_alias=config["opt_alias"])


        if(loop_count == 1 and "export_dlls" in model_info and model_info["export_dlls"]):

            print("Exporting DL models for " + model_info["name"])
            export_dll_models_path = script_dir + "/" + paths_info["export_dll_models_out_relative"] if "export_dll_models_out_relative" in paths_info else dll_models_path
            os.makedirs(export_dll_models_path, exist_ok=True)
            export_dll_libs = model_info["export_dll_libraries"]

            for export_dll_lib in export_dll_libs:
                if("noop" not in export_dll_lib["library"]):
                    export_dll_lib["script_dir"] = script_dir
                    export_dll_lib["name"] = model_info["name"]
                    export_dll_lib["dtype"] = model_info["dtype"] if "dtype" in model_info else "float32"
                    export_dll_lib["export_dll_models_path"] = export_dll_models_path

                    model_exporter.export_model(export_dll_lib)

        folders_to_execute = []
        folders_to_execute.append(images_path)

        images_data = {
            "input_images_folders": folders_to_execute,
            "output_images_base_folder": out_path
        }

        # -----Models execution phase-----
        try:
            # Direct NN model Execution from ONNX.
            if("execute" in model_info and model_info["execute"]):

                print("Executing: " + model_info["name"])

                if mutations_executor is None:
                    mutations_executor = mutations.MutationsExecutor(models_data, images_data, build)
                mutations_executor.execute(remote)

            # Execution of DLLs.
            if("execute_dlls" in model_info and model_info["execute_dlls"]):
                
                dll_models_data = models_data.copy()
                dll_models_data["input_model_folder"] = dll_models_path
                dll_models_data["output_model_folder"] = dll_models_path

                dll_images_data = images_data.copy()
                dll_images_data["output_images_base_folder"] = dll_out_path

                # Run with all given libraries.
                for dll_lib in dll_libs:
                    if("noop" not in dll_lib["library"]):
                        print("Executing: " + model_info["name"] + "(" + dll_lib["library"] + ")")

                        # TODO: Refactor this naming convention, it is very confusing!
                        dll_models_data["name"] = model_info["name"]
                        dll_models_data["model_name"] = dll_lib["dependency"]
                        dll_models_data["converted_lib_model_name"] = dll_lib["converted_lib_model_name"] if "converted_lib_model_name" in dll_lib else None
                        dll_models_data["raw_model_name"] = dll_lib["model"]
                        dll_model_name = (dll_lib["model"] + "_" + dll_lib["library"])
                        dll_model_name = dll_model_name.replace("_", "{DASH}").replace(".", "_").replace("{DASH}", "_")
                        dll_model_name = dll_model_name + ("_quant" if ("quantize" in config and config["quantize"]) else "")
                        dll_model_name = dll_model_name + "_opt" + str(config["opt_level"]) 
                        dll_model_name = dll_model_name + config["opt_alias"] + ".tar"
                        dll_models_data["mutations_names"] = [dll_model_name]
                        dll_input_name = dll_lib["input_name"]
                        dll_output_name = dll_lib["output_name"]
                        dll_models_data["input_name"] = dll_input_name
                        dll_models_data["output_name"] = dll_output_name
                        dll_models_data["input"] = dll_lib["input"]
                        dll_models_data["library"] = dll_lib["library"]
                        dll_models_data["output"] = dll_lib["output"]
                        dll_models_data["args_no"] = dll_lib["args_no"] if "args_no" in dll_lib else 1

                        if("image_dimension" in dll_lib):
                            dll_models_data["image_dimension"] = tuple(dll_lib["image_dimension"])
                        else:
                            dll_models_data["image_dimension"] = tuple(model_info["layers"]["image_dimension"])

                        if (config["backend"] != "tvm" or
                            ("execution_type" in dll_lib and dll_lib["execution_type"] == "library")):
                            libraries_executor = libraries.LibrariesExecutor(dll_models_data, dll_images_data, build)
                            libraries_executor.execute()
                        else:
                            mutations_executor = mutations.MutationsExecutor(dll_models_data, dll_images_data, build)
                            mutations_executor.execute(remote)
            
        except Exception as e:
            print(traceback.print_exc())

        # Evaluate executions.
        if ("evaluate" in model_info and model_info["evaluate"]):
            model_base = script_dir + "/" + paths_info["evaluation_out_relative"]
            mutations_names = [f for f in listdir(model_base) if isfile(join(model_base, f)) and f.endswith(".tar") and ("_ignore_" not in f)]
            generated_models_prettified = [model.replace(".tar", "").replace(".", "_") for model in mutations_names]
            eg_mts = EvaluationGenerator()
            
            print("Evaluating: " + model_info["name"])

            device_index = 0
            device_folders = [d for d in listdir(model_base) if isdir(join(model_base, d))]

            # Compare everything.
            for device_folder in device_folders:
                print("Device Folder:" + device_folder)
                eg_mts.generate_libraries_comparison(join(model_base, device_folder), device_folder=device_folder) #base_case_folder="model_original_opt4"
                eg_mts.generate_base_folder_comparison(generated_models_prettified, join(model_base, device_folder), "mutations")
                eg_mts.generate_devices_comparison(model_base, replace_evaluated_suffix=True)
                eg_mts.get_time_stats_folder(join(model_base, device_folder))
