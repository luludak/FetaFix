import tvm

import pathlib

import tvm.relay as relay

from tvm.contrib.download import download_testdata
from google.protobuf import text_format

import tf2onnx
# TODO: Refactor to remove.
import tflite2onnx
from tf2onnx import tf_loader
from tf2onnx.tfonnx import process_tf_graph
from tf2onnx.optimizer import optimize_graph
from tf2onnx import utils, constants
from tf2onnx.handler import tf_op
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import tvm.relay.testing.tf as tf_testing

from onnx_tf.backend import prepare

import onnxmltools

# PyTorch Conversion
import torch
import torchvision
import torch.onnx 
import torchvision.models as torchvision_models

# TensorFlow conversion
import tensorflow
import tensorflow.compat.v1 as tf
from tensorflow import keras

# ONNX to PyTorch
from onnx2torch import convert as convert_to_torch

from external.onnx2tflite.converter import onnx_converter
from onnx2keras import onnx_to_keras

_TENSORFLOW_DOMAIN = "ai.onnx.converters.tensorflow"

# MXNet Conversion
# from mxnet.gluon.model_zoo import vision as mxnetvision
import mxnet.contrib.onnx as mxnet_onnx
import mxnet as mx

from mxnet.gluon.model_zoo.vision import get_model

#To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
# NOTE: Use this separately to generate Keras models.
# TODO: Add this in config.
#tf.disable_eager_execution()

# Utility functions
import onnx
import os
import numpy as np
from PIL import Image
import subprocess as subp

class ModelLoader:

    def __init__(self):
        self.lookup_table = {
            "tflite": self.load_tflite_model,
            "keras": self.load_keras_model,
            "torch": self.load_torch_model,
            "tf": self.load_tf_model,
            "onnx": self.load_onnx_model_func,
            "noop": self.noop
        }

    def noop(self, data):
        return (None, None)

    # ------------ Loaders ------------
    def load_model(self, data):
        print ("Loading : " + data["name"] + " (" + data["library"] + ") model.")

        # Code responsible for library conversions dynamically.
        if ("_to_" in data["library"]):
            func_to_call = eval("self.convert_" + data["library"] + "_and_load_model")
            return func_to_call(data)
        elif ("onnx" in data["library"]):
            return self.load_onnx_model_func(data)
        return self.lookup_table[data["library"]](data)

    def load_tf_model(self, data):
        model_path = data["dll_models_path"] + data["model"]
        onnx_path = self.convert_tf_to_onnx(model_path, data)
        return self.load_onnx_model(onnx_path, data)

    def load_onnx_model_func(self, data):
        return self.load_onnx_model(data["dll_models_path"] + data["model"] , data)

    def load_tflite_model(self, data, keep_dims=False, use_model_name=True):
        out_path = data["dll_models_path"] + ((data["name"] + "_" + data["library"] + ".tflite") if not use_model_name \
            else data["model_name"])
        onnx_path = self.convert_tflite_path_to_onnx(data, source_path=out_path)
        return self.load_onnx_model(onnx_path, data, keep_dims)

    def load_keras_model(self, data):
        if(data["load"] == "local"):
            return load_keras_from_model_path(data["dll_models_path"] + data["model"].replace(".h5", "") + "/" + data["model"])
        else:
            onnx_path = self.convert_keras_library_to_onnx(data)
            return self.load_onnx_model(onnx_path, data)

    def load_keras_from_model_path(self, path, data):
        if(not pathlib.Path(path).is_file()):
            path_postfix = pathlib.PurePath(path).name
            path = os.path.join(path, path_postfix + ".h5")
        keras_model = tf.keras.models.load_model(path)
        keras_onnx_path = self.convert_keras_to_onnx(keras_model, data)
        return self.load_onnx_model(keras_onnx_path, data, skip_inputs=True)
    
    def convert_keras_library_to_onnx(self, data):
        shape = data["input"]
        output = data["output"]
        shape_chopped = shape[1:]
        if("weights_url" in data):
            weights_url = data["weights_url"]
            weights_file = data["weights_file"]
            weights_path = download_testdata(weights_url, weights_file, module=data["library"])
            model = eval(data["model"] + "(include_top=True, weights=None, input_shape=" + str(shape_chopped) + ", classes=" + str(output[1]) + ")")
            model.load_weights(weights_path)
        else:
            model = eval(data["model"] + "(include_top=True, weights=\"imagenet\", input_shape=" + str(tuple(shape_chopped))  + ", classes=" + str(output[1])  + ")")


        return self.convert_tf_to_onnx_using_api(model, data)

    def load_torch_model(self, data):

        if("load" in data and data["load"] == "local"):
            model_path = torch.load(data["dll_models_path"] + data["model"])
        else:
            model = eval(data["model"] + "(pretrained=True)")
            
        onnx_path = self.convert_torch_to_onnx(model, data)
        return self.load_onnx_model(onnx_path, data)

    def load_onnx_model(self, model_path, data, keep_dims=False, skip_inputs=False):
        print("Model Path: " + model_path)
        model = onnx.load(model_path)
        shape = data["input"]
        data_opset = data["opset"] if "opset" in data else 11
        shape_dict = {}
        model = self.change_input_dim(model)
        if("library" in data and "tflite" in data["library"] and keep_dims == False):
            di = shape
            shape = (di[0], di[3], di[2], di[1])
        if(not skip_inputs):
            shape_dict[data["input_name"]] = shape
            return relay.frontend.from_onnx(model, shape_dict, dtype=data["dtype"], opset=data_opset, freeze_params=True)
        else:
            return relay.frontend.from_onnx(model, dtype=data["dtype"], opset=data_opset, freeze_params=True)


    # ------------ Converters ------------

    def convert_keras_to_onnx(self, model, data, path=None):
        data_opset = data["opset"] if "opset" in data else 11
        onnx_model = onnxmltools.convert_keras(model, target_opset=data_opset) 
        output_path = data["dll_models_path"] + data["name"] + "_" + data["library"] + "_final.onnx" if path is None else path
        onnxmltools.utils.save_model(onnx_model, output_path)
        return output_path

    def convert_torch_to_onnx(self, model, data):

        onnx_model_path = data["dll_models_path"] + data["name"] + "_" + data["library"] + ".onnx"
        # Let's create a dummy input tensor
        shape = data["input"]
        dummy_element = torch.randn(shape[0], shape[1], shape[2], shape[3], requires_grad=True)
        dummy_input = []


        # Export the model   
        torch.onnx.export(model,         # model being run 
            dummy_element,       # model input (or a tuple for multiple inputs) 
            onnx_model_path,       # where to save the model  
            export_params = True,  # store the trained parameter weights inside the model file 
            opset_version = data["opset"] if "opset" in data else 11,    # the ONNX version to export the model to 
            do_constant_folding=True,  # whether to execute constant folding for optimization 
            input_names = ['inputs:0'],   # the model's input names 
            output_names = ['output'] # the model's output names 
            ),

        print(onnx_model_path + ": Torch to ONNX Conversion complete!")
        return onnx_model_path

    def convert_tf_to_tflite_and_load_model(self, data):
        path = data["dll_models_path"] + data["model"]
        out_path = data["dll_models_path"] + data["name"] + "_" + data["library"] + ".tflite"
        self.convert_tf_path_to_tflite(path, out_path, data)
        onnx_path = self.convert_tflite_path_to_onnx(data, source_path=out_path)
        return self.load_tflite_model(data, keep_dims=True, use_model_name=False)

    def convert_torch_to_tflite_and_load_model(self, data):
        path = self.convert_torch_to_tflite(data)
        return self.load_tflite_model(data, keep_dims=True, use_model_name=False)

    def convert_torch_to_keras_and_load_model(self, data):
        model = eval(data["model"] + "(pretrained=True)")
        onnx_path = self.convert_torch_to_onnx(model, data)
        output_path = self.convert_onnx_to_keras(onnx_path, data)
        return self.load_keras_from_model_path(output_path, data)
        

    def convert_tflite_to_keras_and_load_model(self, data):
        # Converter reverts format, counteracting this.
        onnx_path = self.convert_tflite_path_to_onnx(data, invert_nchw_inputs=True)
        output_path = self.convert_onnx_to_keras(onnx_path, data)
        return self.load_keras_from_model_path(output_path, data)

    def convert_keras_library_to_torch_and_load_model(self, data):
        onnx_keras_path = self.convert_keras_library_to_onnx(data)
        onnx_model = onnx.load(onnx_keras_path)
        torch_model = convert_to_torch(onnx_model)
        torch_onnx_path = self.convert_torch_to_onnx(torch_model, data)
        return self.load_onnx_model(torch_onnx_path, data, skip_inputs=True)

    def convert_keras_to_torch_and_load_model(self, data):
        onnx_model = self.load_keras_from_model_path(data["dll_models_path"] + data["name"] + "_" + data["library"], data)
        torch_model = convert_to_torch(onnx_model)
        torch_onnx_path = self.convert_torch_to_onnx(torch_model, data)
        return self.load_onnx_model(torch_onnx_path, data, skip_inputs=True)

    def convert_tf_to_keras_and_load_model(self, data):
        model_path = data["dll_models_path"] + data["model"]
        tf_onnx_path = self.convert_tf_to_onnx(model_path, data)
        keras_output_path = self.convert_onnx_to_keras(tf_onnx_path, data)
        return self.load_keras_from_model_path(keras_output_path, data)
    
    def convert_tf_to_torch_and_load_model(self, data):
        model_path = data["dll_models_path"] + data["model"]
        tf_onnx_path = self.convert_tf_to_onnx(model_path, data)
        tf_onnx_model = onnx.load(tf_onnx_path)
        torch_model = convert_to_torch(tf_onnx_model)
        torch_onnx_path = self.convert_torch_to_onnx(torch_model, data)
        return self.load_onnx_model(torch_onnx_path, data, skip_inputs=True)
        
    def convert_tflite_to_torch_and_load_model(self, data):
        onnx_tflite_path = self.convert_tflite_path_to_onnx(data)
        onnx_model = onnx.load(onnx_tflite_path)
        torch_model = convert_to_torch(onnx_model)
        torch_onnx_path = self.convert_torch_to_onnx(torch_model, data)
        return self.load_onnx_model(torch_onnx_path, data, skip_inputs=True)

    def convert_keras_library_to_tf_and_load_model(self, data):
        onnx_keras_path = self.convert_keras_library_to_onnx(data)
        torch_model = self.convert_onnx_to_torch(onnx_keras_path)
        torch_onnx_path = self.convert_torch_to_onnx(torch_model, data)
        tf_onnx_model = self.convert_onnx_to_tf_and_load_via_onnx(torch_onnx_path)
        
        graph_tf_path = data["dll_models_path"] + data["name"] + "_" + data["library"] 
        tf_onnx_model.export_graph(graph_tf_path)
        self.convert_tf_saved_model_to_onnx(graph_tf_path, data, export_path=graph_tf_path + ".onnx")
        return self.load_onnx_model(graph_tf_path + ".onnx", data, skip_inputs=True)

    def convert_keras_library_to_tflite_and_load_model(self, data):
        shape = data["input"]
        output = data["output"]
        shape_chopped = shape[1:]
        model = eval(data["model"] + "(include_top=True, weights=\"imagenet\", input_shape=" + str(tuple(shape_chopped))  + ", classes=" + str(output[1])  + ")")
        print(model.input_names)
        print(model.output_names)
        converter = tensorflow.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tf_lite_model = converter.convert()
        out_tflite_path = data["dll_models_path"] + data["name"] + "_" + data["library"] + ".tflite"
        open(out_tflite_path, 'wb').write(tf_lite_model)
        onnx_path = self.convert_tflite_path_to_onnx(data, source_path=out_tflite_path)
        return self.load_onnx_model(onnx_path, data, skip_inputs=False, keep_dims=True)

    def convert_torch_to_tf_and_load_model(self, data):
        model = eval(data["model"] + "(pretrained=True)")
        onnx_path = self.convert_torch_to_onnx(model, data)

        tf_model = self.convert_onnx_to_tf_and_load_via_onnx(onnx_path)
        
        graph_tf_path = data["dll_models_path"] + data["name"] + "_final_model_torch_to_tf"
        tf_model.export_graph(graph_tf_path)
        self.convert_tf_saved_model_to_onnx(graph_tf_path, data, export_path=graph_tf_path + ".onnx")
        return self.load_onnx_model(graph_tf_path + ".onnx", data, skip_inputs=True)

   
    def convert_tflite_to_tf_and_load_model(self, data):
        onnx_path = self.convert_tflite_path_to_onnx(data)
        tf_model = self.convert_onnx_to_tf_and_load_via_onnx(onnx_path)
        graph_tf_path = data["dll_models_path"] + data["name"] + "_final_model_tflite_to_tf"
        tf_model.export_graph(graph_tf_path)
        self.convert_tf_saved_model_to_onnx(graph_tf_path, data, export_path=graph_tf_path + ".onnx")
        model = self.load_onnx_model(graph_tf_path + ".onnx", data, skip_inputs=True)
        return model

    def convert_tf_saved_model_path_to_tflite(self, model_path, out_path, data):
         

        converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(model_path)
            
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
        #                                     tf.lite.OpsSet.SELECT_TF_OPS]
                                            
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tf_lite_model = converter.convert()
        open(out_path, 'wb').write(tf_lite_model)

        return out_path


    def convert_tf_path_to_tflite(self, model_path, out_path, data):
         

        converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(model_path, #TensorFlow freezegraph,
                                                input_shapes={data["input_name"]: data["input"]},
                                                input_arrays=[data["input_name"]], # name of input
                                                output_arrays=[data["output_name"]]  # name of output
                                                )
            
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
        #                                     tf.lite.OpsSet.SELECT_TF_OPS]      
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tf_lite_model = converter.convert()
        open(out_path, 'wb').write(tf_lite_model)

        return out_path

    # -- TORCH TO TF
    def convert_torch_to_tf(self, data, path=None):

        model = eval(data["model"] + "(pretrained=True)")
        model_path = self.convert_torch_to_onnx(model, data)
        tf_rep = self.convert_onnx_to_tf_and_load_via_onnx(model_path)
        path = data["dll_models_path"] + data["name"] + "_" + data["library"] if path is None else path
        tf_rep.export_graph(path)
        return path

    def convert_torch_to_tflite(self, data):
        pb_path = self.convert_torch_to_tf(data)
        return self.convert_tf_saved_model_path_to_tflite(pb_path, pb_path + "/" + data["name"]  + "_torch_to_tflite.tflite", data)


    def convert_tf_saved_model_to_onnx(self, path, data, export_path="model_to_onnx.onnx"):
        data_opset = data["opset"] if "opset" in data else 11

        command = 'python3 -m tf2onnx.convert --opset ' + str(data_opset) + ' --saved-model "' + path + '" --output "' + export_path + '"'
        subp.check_call(command, shell=True)
        return export_path


    def convert_tf_to_onnx(self, path, data):
        
        onnx_model_path = data["dll_models_path"] + data["name"] + "_" + data["library"] + ".onnx"
        di = data["input"]

        if("scalar_input" in data and data["scalar_input"]):
            data_opset = data["opset"] if "opset" in data else 11
            command = 'python3 -m tf2onnx.convert --opset ' + str(data_opset) + ' --graphdef "' + path + '" --inputs "' + \
            data["input_name"] + '[' + str(di[0]) + ',' + str(di[1]) + ',' + str(di[2]) + ',' + str(di[3]) + ']" --outputs "' \
            + data["output_name"] + '" --output "' + onnx_model_path + '" --inputs-as-nchw "' + \
            str(di[0])  + ' ' + str(di[3])  + ' ' + str(di[2])   + ' ' + str(di[1]) + '"'
            subp.check_call(command, shell=True)
            
            print ("Model exported at" + onnx_model_path)
            return onnx_model_path

        graph_def = tf.compat.v1.GraphDef()
        with open(path, 'rb') as f:
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')

        inputs = [data["input_name"]]
        outputs = [data["output_name"]]
        # optional step, but helpful to facilitate readability and import to Barracuda.
        newGraphModel_Optimized = tf2onnx.tfonnx.tf_optimize(inputs, outputs, graph_def)

        # saving the model
        tf.compat.v1.reset_default_graph()
        tf.import_graph_def(newGraphModel_Optimized, name='')
        
        with tf.compat.v1.Session() as sess:
            # print(type(sess.graph))
            model_proto, o = tf2onnx.convert.from_graph_def(sess.graph.as_graph_def(), input_names=inputs, output_names=outputs, inputs_as_nchw=(di[0], di[3], di[2], di[1]))
            checker = onnx.checker.check_model(model_proto)

            self.change_input_dim(model_proto)
            utils.save_protobuf(onnx_model_path, model_proto)
            print(onnx_model_path + ": Conversion complete!")
            return onnx_model_path



    def convert_tf_to_onnx_using_api(self, model, data):
        onnx_model_path = data["dll_models_path"] + data["name"] + "_" + data["library"] + ".onnx"
        full_model = tf.function(lambda inputs: model(inputs[0]))
        # print(dir(full_model))

        tensor_shape = model.inputs[0].shape if model.inputs[0].shape is not None \
             else tf.TensorShape([1, model.inputs[0].shape[1], model.inputs[0].shape[2], model.inputs[0].shape[3]])
        full_model = full_model.get_concrete_function([tf.TensorSpec(tensor_shape, model.inputs[0].dtype)])
        input_names = [data["input_name"]]
        output_names = [data["output_name"]]
        # print("Inputs:", input_names)
        # print("Outputs:", output_names)

        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()

        data_opset = data["opset"] if "opset" in data else 11

        extra_opset=[utils.make_opsetid(_TENSORFLOW_DOMAIN, 1)]
        with tf.Graph().as_default() as tf_graph:
            tf.import_graph_def(frozen_func.graph.as_graph_def(), name='')

        with tf_loader.tf_session(graph=tf_graph):
            g = process_tf_graph(tf_graph, input_names=input_names, 
            output_names=output_names, extra_opset=extra_opset, opset=data_opset)
            
        onnx_graph = optimize_graph(g)
        model_proto = onnx_graph.make_model("converted")

        self.change_input_dim(model_proto)
        utils.save_protobuf(onnx_model_path, model_proto)
        print(onnx_model_path + ": Conversion complete!")
        return onnx_model_path

    def convert_tflite_path_to_onnx(self, data, source_path=None, invert_nchw_inputs=False):
        onnx_model_path = data["dll_models_path"] + data["name"] + "_" + data["library"] + ".onnx" 
        model_path = data["dll_models_path"] + data["model"] if source_path is None else source_path
        di = data["input"]
        try:
            data_opset = data["opset"] if "opset" in data else 11
            input_format = "inputs" + ("-as-nchw" if invert_nchw_inputs else "")
            command = 'python3 -m tf2onnx.convert --opset ' + str(data_opset) + ' --tflite "' + model_path + '" --output "' + onnx_model_path + '" --' + input_format + ' "' + data["input_name"] + '" --outputs "' + data["output_name"] + '"'
            subp.check_call(command, shell=True)
        except subp.CalledProcessError as e:
            print("Error on command call.")
        print ("Converted TFLite From Path to ONNX.")
        return onnx_model_path

    # ------------ ONNX Converters ------------

    def convert_onnx_to_tf_and_load_via_onnx(self, onnx_path):
        loaded_model = onnx.load(onnx_path)
        self.change_input_dim(loaded_model)
        return prepare(loaded_model)

    def convert_onnx_to_torch(self, onnx_path):
        loaded_model = onnx.load(onnx_path)
        return convert_to_torch(loaded_model)


    def convert_onnx_to_tflite(self, onnx_path, data):
        output_path = onnx_path.replace(".onnx", ".tflite")
        onnx_converter(onnx_model_path=onnx_path, int8_model=False, output_path=output_path, target_formats=['tflite'], input_node_names=[data["input_name"]], output_node_names=[data["output_name"]])
        return output_path

    # Generates h5 model in folder with the same name.
    def convert_onnx_to_keras(self, onnx_path, data):
        output_path = onnx_path.replace(".onnx", "")
        onnx_converter(onnx_model_path=onnx_path, int8_model=False, output_path=output_path, target_formats=['keras'],  input_node_names=[data["input_name"]], output_node_names=[data["output_name"]])
        return output_path

    # ------------ Helpers ------------

    def change_input_dim(self, model):
    # Use some symbolic name not used for any other dimension
        sym_batch_dim = "N"
        # or an actal value
        actual_batch_dim = 1

        # The following code changes the first dimension of every input to be batch-dim
        # Modify as appropriate ... note that this requires all inputs to
        # have the same batch_dim 
        inputs = model.graph.input
        outputs = model.graph.output
        # print("Inputs:")
        for input in inputs:
            # print(input)
            
            dim = input.type.tensor_type.shape.dim

            i = 0
            while i < len(dim):
                dim1 = input.type.tensor_type.shape.dim[i]
                if("unk" in dim1.dim_param):
                    print ("Changed symbolic input dimension to actual for model.")
                    input.type.tensor_type.shape.dim[i].dim_value = 1
                    input.type.tensor_type.shape.dim[i].dim_param = "1"
                i += 1

        for output in outputs:
            
            dim = output.type.tensor_type.shape.dim

            i = 0
            while i < len(dim):
                dim1 = output.type.tensor_type.shape.dim[i]
                if("unk" in dim1.dim_param):
                    print ("Changed symbolic input dimension to actual for model.")
                    output.type.tensor_type.shape.dim[i].dim_value = 1
                    output.type.tensor_type.shape.dim[i].dim_param = "1"
                i += 1
 

        outputs = model.graph.output
        # print("Outputs:")
        for output in outputs:
            # print(output)
            # Checks omitted.This assumes that all inputs are tensors and have a shape with first dim.
            # Add checks as needed.
            dim1 = output.type.tensor_type.shape.dim[0]
            if("unk" in dim1.dim_param or "dim" in dim1.dim_param):
                print (dim1)
                print ("Changed symbolic output dimension to actual for model.")
                output.type.tensor_type.shape.dim[0].dim_param = "1"
            
        
        return model
