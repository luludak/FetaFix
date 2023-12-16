import time
import math
import tvm
import os
import json
import traceback
import tvm.runtime as runtime
from tvm.contrib import graph_executor
from tvm.contrib import utils
from PIL import Image
import numpy as np
from scipy.special import softmax
from os import listdir
from os.path import isfile, join
import onnxruntime as ort

from processors.model_preprocessor import ModelPreprocessor

from tvm.contrib.debugger.debug_executor import GraphModuleDebug

from tensorflow import keras
from keras import backend as K
from keras.models import load_model
import tensorflow.compat.v1 as tf
import torch
import torchvision.models as torchvision_models

from tensorflow.python.platform import gfile

class Executor:
    def __init__(self, models_data, images_data, connection_data, extra_folder):
        self.name = models_data["name"]
        self.model_name = models_data["model_name"]
        self.converted_lib_model_name = models_data["converted_lib_model_name"] if "converted_lib_model_name" in models_data else None
        self.raw_model_name = models_data["raw_model_name"] if "raw_model_name" in models_data else self.model_name
        self.input_model_folder = models_data["input_model_folder"]
        self.output_model_folder = models_data["output_model_folder"]
        self.mutations_names = models_data["mutations_names"] if "mutations_names" in models_data else [""]
        self.image_dimension = models_data["image_dimension"]
        self.input = models_data["input"]
        self.input_name = models_data["input_name"]
        self.output = models_data["output"]
        self.output_name = models_data["output_name"]
        self.library = models_data["library"] if "library" in models_data else None
        self.input_images_folders = images_data["input_images_folders"]
        self.output_images_base_folder = images_data["output_images_base_folder"]
        self.extra_folder = extra_folder
        self.connection_id = "local"
        self.connection_data = connection_data
        self.error_base_folder = connection_data["error_base_folder"]
        self.module = None
        self.device = None
        self.preprocessing_enabled = models_data["preprocessing_enabled"]
        self.debug_enabled = models_data["debug_enabled"]
        self.loaded_module = None
        self.json_str = None
        self.op_type =  models_data["op_type"] if "op_type" in models_data else None
        self.params = models_data["params"] if "params" in models_data else None
        self.model_path = join(self.input_model_folder, self.model_name)

        self.same_ranks_threshold = 100

    def prepare(self, variant="", remote=None):

        device_id = self.connection_data["id"] or 0
        host_type = self.connection_data["host_type"] or None
        if(host_type == "local_no_rpc" or host_type == "local" or remote == None):
            self.loaded_module = runtime.module.load_module(self.input_model_folder + variant)
            target_framework = self.connection_data["target_framework"]
            self.device = tvm.device(str(target_framework), device_id)
        else:

            model_path = join(self.input_model_folder, variant)
            remote.upload(model_path)
            target_framework = self.connection_data["target_framework"]
            self.loaded_module = remote.load_module(variant)
            self.device = remote.device(target_framework, device_id)

        
        if (not self.debug_enabled):
            self.module = graph_executor.GraphModule(self.loaded_module["default"](self.device))
        else:
            # Prepare graph.
            complete_path = self.input_model_folder + variant
            with open(complete_path.replace(".tar", ".json")) as f:
                self.json_str = f.read()
                f.close()

                self.module = GraphModuleDebug(
                    self.loaded_module["debug_create"]("default", self.device),
                    [self.device],
                    self.json_str,
                    dump_root=self.output_images_base_folder + "/" + self.library + "/debug/")
            
        return self

    def execute(self):
        if(self.module == None or self.device == None):
            raise Exception("Error: Device not initialized.")
        return "ts_0"

    def extract_image_names(self, folder_path):
        return [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

    def process_images(self):
        self.process_images_with_output(module, self.input_images_folder, self.output_images_folder)
        return self

    def get_last_folder(self, folder_path):
        return os.path.basename(os.path.normpath(folder_path))


    def process_library_images_with_io(self, input_folder, output_folder, specific_images = None):
        os.makedirs(output_folder, exist_ok=True)

        print("Library Processing - Dataset images in folder: " + input_folder)

        if specific_images is None:
            image_names = self.extract_image_names(input_folder)
        else:
            image_names = specific_images

        image_names_length = len(image_names)
        image_length = (image_names_length - 1) if (image_names_length > 1) else image_names_length
        step = (image_length // 4) if image_length > 4 else image_length
        
        count = 0
        same_ranks = 0
        prev_ranks = None

        input_shape_inferred = self.input

        data = {
            "input": input_shape_inferred,
            "image_dimension": self.image_dimension,
            "library": self.library
        }
        model_preprocessor = ModelPreprocessor(data)
        error_occured = False
        errors_no = 0
        for image_name in image_names:

            if(count % step == 0):
                print("Complete: " + str((count // step) * 25) + "%")

            count += 1
            if (image_name.startswith(".")):
                print("Skipping " + image_name)
                continue

            img_path = input_folder + "/" + image_name

            img = Image.open(img_path)
            # Convert monochrome to RGB.
            if(img.mode == "L" or img.mode == "BGR"):
                img = img.convert("RGB")

            img = img.resize(self.image_dimension)
            img = model_preprocessor.preprocess(self.model_name, img, self.preprocessing_enabled)

            image_name_extracted = image_name.rsplit('.', 1)[0]
        
            output_file_path = output_folder + "/" + image_name_extracted.replace(".", "_") + ".txt"

            try:
                
                output = None
                start_timestamp = self.get_epoch_timestamp(False)
                if("onnx" in self.library):
                    ort_sess = ort.InferenceSession(join(self.input_model_folder, self.model_name), providers=['CPUExecutionProvider'])
                    output = ort_sess.run(None, {self.input_name : img.astype(np.float32)})
                    end_timestamp = self.get_epoch_timestamp(False)

                elif(self.library.startswith("torch")):
                    
                    if(not self.library.endswith("library")):
                        model = torch.load(self.model_path)

                    model_to_execute = eval(self.model_name + "(pretrained=True)")
                    model_to_execute.eval()
                    output = model_to_execute(torch.from_numpy(img)).detach().numpy()
                    end_timestamp = self.get_epoch_timestamp(False)

                elif(self.library == "h5" or self.library.startswith("keras")):
                    if(not self.library.endswith("library")):
                        model_to_execute = tf.keras.models.load_model(self.model_path)
                    else:
                        model_to_execute = eval(self.raw_model_name + "(include_top=True, weights=\"imagenet\", input_shape=None, classes=" + str(self.output[1])  + ")")
                    with tf.Graph().as_default():
                        with tf.Session() as sess:
                            K.set_session(sess)

                            output = model_to_execute.predict(img)
                            end_timestamp = self.get_epoch_timestamp(False)

                elif(self.library == "pb" or self.library == "tf" or self.library == "tensorflow"):

                    with tf.Graph().as_default() as graph:
                        with tf.compat.v1.Session() as sess:
                            with open(model_path, 'rb') as f:
                                graph_def = tf.compat.v1.GraphDef()
                                graph_def.ParseFromString(f.read())
                                sess.graph.as_default()

                                tf.import_graph_def(graph_def, input_map=None, return_elements=None, name="", op_dict=None, producer_op_list=None)

                                l_output = graph.get_tensor_by_name(self.output_name)
                                l_input = graph.get_tensor_by_name(self.input_name)
                                tf.global_variables_initializer()

                                output = sess.run(l_output, feed_dict = {l_input : img})
                                end_timestamp = self.get_epoch_timestamp(False)

                elif(self.library == "tflite"):

                    model_path = join(self.input_model_folder, self.model_name)
                    # Transpose dimensions
                    img = np.transpose(img, (0, 2, 3, 1))
                    # Normalize between 0 and 1.
                    img = tf.divide(img, tf.reduce_max(img))

                    interpreter = tf.lite.Interpreter(model_path=model_path)
                    interpreter.allocate_tensors()

                    # Get input and output tensors.
                    input_details = interpreter.get_input_details()
                    output_details = interpreter.get_output_details()

                    # Test the model on random input data.
                    input_data = np.array(img, dtype=np.float32)

                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()

                    output = interpreter.get_tensor(output_details[0]['index'])
                    end_timestamp = self.get_epoch_timestamp(False)

                # Score process.
                scores = softmax(output)
                if(len(scores) > 2):
                    squeeze(scores, [0, 2])

                scores = np.squeeze(scores)
                ranks = np.argsort(scores)[::-1]
                extracted_ranks = ranks[0:5]
                if(prev_ranks is not None and np.array_equal(prev_ranks, extracted_ranks)):
                    same_ranks += 1
                else:
                    prev_ranks = extracted_ranks
                    same_ranks = 0

                if(same_ranks == self.same_ranks_threshold):
                    print ("Warning: Execution produced same ranks for 100 consecutive inputs. Stopped.")
                    return self

                with open(output_file_path, 'w') as output_file:
                    for rank in extracted_ranks:
                        print("%s, %f" % (rank, scores[rank]), file = output_file)
                    
                    print("\n%f" % (end_timestamp - start_timestamp), file = output_file)
                    output_file.close()
                    
            except Exception as e:
                start_timestamp = self.get_epoch_timestamp(False)
                ts_floor = str(math.floor(start_timestamp))

                folder_to_write = join(self.error_base_folder, ts_floor)
                os.makedirs(folder_to_write, exist_ok=True)
                f = open(join(folder_to_write, 'error_log.txt'), 'a+')
                f.write("Model: " + self.model_name + "\nOutput folder:" + output_folder + "\nImage: "+ img_path + "\n")
                traceback.print_exc(file=f)
                f.close()

        return self


    def reset_module(self, variant):
            
        self.module = GraphModuleDebug(
            self.loaded_module["debug_create"]("default", self.device),
            [self.device],
            self.json_str,
            dump_root=self.output_images_base_folder + "/" + self.library + "/debug/")
        print("-------------------------------------------------Module Reset.")

    def process_images_with_io(self, input_folder, output_folder, model_name=None, variant=None, specific_images=None, should_write_to_file=True):

        if (model_name == None):
            model_name = self.model_name

        os.makedirs(output_folder, exist_ok=True)

        print("Processing Dataset images in folder: " + input_folder)

        if specific_images is None:
            image_names = self.extract_image_names(input_folder)
        else:
            print("(Image subset considered.)")
            image_names = specific_images

        image_names_length = len(image_names)
        image_length = (image_names_length - 1) if (image_names_length > 1) else image_names_length
        step = (image_length // 4) if image_length > 4 else image_length
        
        count = 0
        same_ranks = 0
        prev_ranks = None

        input_shape_inferred = self.module.get_input(0).shape

        data = {
            "input": input_shape_inferred,
            "image_dimension": self.image_dimension,
            "library": self.library
        }
        model_preprocessor = ModelPreprocessor(data)
        error_occured = False
        errors_no = 0
        output_object = {}
        return_obj = {}

        for image_name in image_names:

            if(count % step == 0):
                print("Complete: " + str((count // step) * 25) + "%")

            count += 1
            if (image_name.startswith(".")):
                print("Skipping " + image_name)
                continue

            img_path = input_folder + "/" + image_name
           
            img = Image.open(img_path)
            if(img.mode == "L"):
                img = img.convert("RGB")

            img = img.resize(self.image_dimension)
            img = model_preprocessor.preprocess(self.model_name, img, self.preprocessing_enabled)

            image_name_extracted = image_name.rsplit('.', 1)[0]

            output_file_path = output_folder + "/" + image_name_extracted.replace(".", "_") + ".txt"
            
            try:
                start_timestamp = self.get_epoch_timestamp(False)

                self.module.set_input(self.input_name, img)
                self.module.run()
                
                out = tvm.nd.empty(self.output, device=self.device)

                tvm_output = self.module.get_output(0, out).numpy()
                end_timestamp = self.get_epoch_timestamp(False)

                scores = softmax(tvm_output)
                if(len(scores) > 2):
                    squeeze(scores, [0, 2])

                scores = np.squeeze(scores)
                ranks = np.argsort(scores)[::-1]
                extracted_ranks = ranks[0:5]
                if(prev_ranks is not None and np.array_equal(prev_ranks, extracted_ranks)):
                    same_ranks += 1
                else:
                    prev_ranks = extracted_ranks
                    same_ranks = 0

                if(same_ranks == 100):
                    print ("Warning: Execution produced same ranks for 100 consecutive inputs. Stopped.")
                    return self

                if (should_write_to_file):
                    with open(output_file_path, 'w') as output_file:
                        # TOP-K of ranks, with K=5.
                        for rank in extracted_ranks:
                            print("%s, %f" % (rank, scores[rank]), file = output_file)
                        
                        print("\n%f" % (end_timestamp - start_timestamp), file = output_file)
                        output_file.close()
        
                output_object[image_name_extracted] = {
                    "ranks": extracted_ranks,
                    "scores": scores
                }

                if(self.debug_enabled):
                    
                    debug_base = self.output_images_base_folder + "/" + self.library + "/debug/_tvmdbg_device_OPENCL_0/"
                    
                    os.rename(debug_base + "/output_tensors.params", debug_base + "/output_tensors_" + image_name.split(".")[0] + ".params")
                    return_obj["debug_base"] = debug_base
                    self.reset_module(variant)

                return_obj["output"] = output_object

            except Exception as e:
                ts_floor = str(math.floor(self.get_epoch_timestamp(False)))
                errors_no += 1
                if(not error_occured):
                        error_occured = True
                        print("One or more image errors have occured. See execution log of the model for details (ts: " + ts_floor + ").")
                
                if(errors_no == 100):
                    print ("100 Errors have occured. Stopping execution.")
                    return

                folder_to_write = join(self.error_base_folder, ts_floor)
                os.makedirs(folder_to_write, exist_ok=True)
                f = open(join(folder_to_write, 'error_log.txt'), 'a+')
                f.write("Model: " + model_name + "\nOutput folder:" + output_folder + "\nImage: "+ img_path + "\n")
                traceback.print_exc(file=f)
                f.close()

        return return_obj

    
    def get_epoch_timestamp(self, use_floor = True):
        time_to_return = time.time()
        return math.floor(time_to_return) if use_floor else time_to_return
