import time
import json
import os
import onnx
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
import tvm.runtime as runtime

script_dir = os.path.dirname(os.path.realpath(__file__))

def get_code_dir_path(model_name, out_path):
    return script_dir + "/../../" + out_path + "/" + model_name + "_code"
    

def generate_original_model(model, target, params, out_path_relative, file_name=None, \
    opt_level=None, required_pass=None, disabled_pass=None, quantize=False, opt_alias=""):
    print("Generating original model.")
    print("Folder: " + out_path_relative)

    model_name = file_name if file_name is not None else "model_original"
    model_name = model_name if not quantize else model_name + "_quant" 
    #model_name = model_name + ("_opt" + str(opt_level) if opt_level is not None else "")
    model_name = model_name + opt_alias

    opt_level = opt_level if opt_level is not None else 3
    
    os.makedirs(get_code_dir_path(model_name, out_path_relative), exist_ok=True)
    with tvm.transform.PassContext(opt_level=opt_level, required_pass=required_pass, disabled_pass=disabled_pass):
        lib = relay.build(model, target=target, params=params)

        file_path = script_dir + "/../../" + out_path_relative  + "/" + model_name + ".tar"
        lib.export_library(file_path)
        print("Path: " + file_path)
        generate_model_end_code(lib, model, model_name, out_path_relative)

        output_json_file = script_dir + "/../../" + out_path_relative  + "/" + model_name + ".json"
        with open(output_json_file, 'w') as outfile:
            print(lib.graph_json, file=outfile)
            outfile.close()

        print(lib.params.keys())
        output_params_json_file = script_dir + "/../../" + out_path_relative  + "/" + model_name + "_lib_params.params"
        with open(output_params_json_file, 'wb') as outfile:
            outfile.write(relay.save_param_dict(lib.params))
            outfile.close()
            
    return file_path


def generate_model_end_code(lib, model, model_name, out_path_relative):
    output_host_file = open(get_code_dir_path(model_name, out_path_relative) + "/output_host.txt",'w')
    output_device_file = open(get_code_dir_path(model_name, out_path_relative) + "/output_device.txt",'w')
    output_relay_file = open(get_code_dir_path(model_name, out_path_relative) + "/output_relay.txt",'w')
    imported_modules = lib.get_lib().imported_modules

    if imported_modules is not None and len(imported_modules) > 0:
        dev_module = lib.get_lib().imported_modules[0]
        print(dev_module.get_source(), file=output_device_file)
        output_device_file.close()
    else:
        print("WARNING: Device code not generated for " + model_name + ".")
    print(lib.get_lib().get_source(), file=output_host_file)
    print(model, file=output_relay_file)
    output_host_file.close()
