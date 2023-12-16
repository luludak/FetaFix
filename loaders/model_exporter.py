# PyTorch Conversion
import torch
import torchvision
import torch.onnx 
import torchvision.models as torchvision_models

# TensorFlow conversion
import tensorflow
import tensorflow.compat.v1 as tf
from tensorflow import keras

class ModelExporter:

    def __init__(self):
        self.lookup_table = {
            "keras": self.export_keras_model,
            "torch": self.export_torch_model,
            "noop": self.noop
        }

    def noop(self, data):
        return (None, None)


    def export_model(self, data):
        print ("Exporting : " + data["name"] + " (" + data["library"] + ") model.")
        return self.lookup_table[data["library"]](data)


    def export_keras_model(self, data):
        # TODO: Implement URL load.

        model = shape = data["input"]
        output = data["output"]
        shape_chopped = shape[1:]
        model = eval(data["model"] + "(include_top=True, weights=\"imagenet\", input_shape=" + str(tuple(shape_chopped))  + ", classes=" + str(output[1])  + ")")

        # Replace h5 in case it exists
        output_path = data["export_dll_models_path"] + data["name"] + data["library"] + "/" + data["name"] + data["library"] + ".h5"
        model.save(output_path, save_format='h5')
        
        print("Keras Model " + data["name"] + " exported.")
        return output_path

    def export_torch_model(self, data):

        model = eval(data["model"] + "(pretrained=True)")

        if("format" in data and data["format"] == "onnx"):
            onnx_model_path = data["export_dll_models_path"] + data["name"] + "_" + data["library"] + ".onnx"
            # Let's create a dummy input tensor
            shape = data["input"]
            dummy_element = torch.randn(shape[0], shape[1], shape[2], shape[3], requires_grad=True)
            dummy_input = []


            # Export the model   
            torch.onnx.export(model,         # model being run 
                dummy_element,       # model input (or a tuple for multiple inputs) 
                onnx_model_path,       # where to save the model  
                export_params = True,  # store the trained parameter weights inside the model file 
                opset_version = 11,    # the ONNX version to export the model to 
                do_constant_folding=True,  # whether to execute constant folding for optimization 
                input_names = ['inputs:0'],   # the model's input names 
                output_names = ['output'] # the model's output names 
                )

        
        output_path = data["export_dll_models_path"] + data["name"] + data["library"] + ".pt"
        torch.save(model.state_dict(), output_path)
        print("PyTorch Model " + data["name"] + " exported.")
        return output_path