# ConRepAIr

ConRepAIr is a comprehensive suite for compiling, optimizing, executing and analyzing pretrained DNNs under different computational environment settings.

ConRepAIr supports:

- Automatic Fault Localization across ONNX model correspondents (Source and Target) between a DL framework conversion.
- Automatic Fault Repair of the Target model, if behavior is deviating from Source, by repairing model input, graph, parameters and hyperparameters.

Also, it can:
- Build of Neural Networks using different backend DL Frameworks.
- Build of DNNs under different optimization settings.
- Build of DNNs using different GPU backends (CUDA, OpenCL, etc).
- Convert oDNNs from one backend framework to another (currently supporting all conversions across Keras, PyTorch, TF, TFlite).
- Execute DNNs in different hardware acceleration environments.
- Analyze the results in a bundled and automated manner.

The suite is based on [Apache TVM](https://tvm.apache.org/) for its capabilities.

## Installation

The system needs TVM to be installed.
We also use `Python v3.8.5` and `Pip` as the package installer.

In addition, the system requires a number of pip packages, which you can find in the requirements.txt file.

## Instructions:

1. Install Python and Pip on your system.
- Python comes with linux distros usually, but this is not always the case for Pip. You can install it by running "sudo apt install python3-pip"
2. Download and install TVM:
For instructions of how to install TVM, please refer to the [TVM related guide for developers](https://tvm.apache.org/docs/install/from_source.html#developers-get-source-from-github).
Follow the installation from source instructions, and consider enabling the LLVM and the OPENCL flags.

3. Install necessary packages by executing the command:
`pip3 install -r requirements.txt`

4. Download necessary TF/TFLite models, if you wish to run them.
Although system utilizes already provided models for Keras and PyTorch, we utilized some TF/TFlite models from the GitHub repo of Tensorflow for slim Models for the experiments. These are:
- `MobileNetV2`
- `ResNet101`
- `InceptionV3`

You can download them manually and place them in the models folder of each model from [the official TensorFlow repo](https://github.com/tensorflow/models/tree/master/research/slim).

Following download, extract and put the models into `<script_folder>/generated/<Model_Folder>/models` folder. Do so for both .pb and .tflite models.
Also, make sure the names of the models are the same as in configuration for each model.

## Configuration
The configuration of the system is included into the config.json file.
Each section is self-explanatory and defines which part it concerns.
Important notes:
- You can run the models **without** TVM, directly using the library of your choice. In this case, set the flag `backend` to `libraries` instead of `tvm`.
- You can utilize the TVM debugger, by setting `debug_enabled: true`.
- `build` and `execute` flags concerns the ONNX model defined in the URL and will apply actions only to this. If you want DLLs to be built or executed, mark flag `build_dlls` or `execute_dlls` as true.
- `evaluate` flag concerns DLLs as well.
- Device settings have been cleared out to preserve anonymity. If you wish, you can set up your own TVM RPC server on your own device and run everything following the instructions [here](
https://tvm.apache.org/docs/tutorial/cross_compilation_and_rpc.html).

## Example
In order to verify your installation and be able to run the framework with your own configuration, we have setup the configuration to build the system utilizing 3 libraries:
1. TFLite (Downloaded an included from the TF repo aforementioned).
2. Keras (Using pre-built library of keras).
3. PyTorch (Same as keras).

As Dataset, we provide a small dataset, obtained from [unsplash](https://unsplash.com/images/stock/public-domain). No copyright infingement intended.
We provide the native TF and TFLite models, obtained from [TensorFlow zoo slim repo](https://github.com/tensorflow/models/tree/master/research/slim/), while the system supports inference and conversion across the pretrained models that are part of the Keras and PyTorch DL frameworks API.

Once you set up the framework, you can execute it by doing:
`python3 main.py`

The example case will build, run and execute evaluation for `MobileNetV2`, in `TFLite` DL Framework. The evaluation will give an empty devices file, as no simultaneous library runs are performed, and there are no other runs to additional devices.

#### Build: 
The system will generate the models in the folder defined in config.json, along with their generated Host/Kernel code, but also their TVM Relay IR code:
`<script_folder>/generated/MobileNet-2-7/models`

In total, the framework will generate the models compiled on TVM, utilizing the `opt=2` optimization setting, to be executed using `OpenCL` for hardware acceleration, for `TFLite`, `Keras` and `PyTorch`.

##### Convert:
ConRepAIr supports conversions of DL frameworks, for Keras, PyTorch, TF, TFlite. This can be enabled by setting <source>_to_<target> model in `dll_libraries` configuration of a model, in `config.json` file. For Keras, add `keras_library` as Source. You can use the provided sample `config.json` in order to perform your conversions. Just remove `noop_` prefix from the conversion title and enable the model conversions by setting `"skip_analysis": false`, and `"build_dlls": true` in the respective model settings. Note: For TF/TFLite conversions, you will need to download the `.pb/.tflite` files from the official TF repo, as described above. PyTorch and Keras use the native implementations provided along the libraries, but you need to install them as project dependencies.


#### Execute:
You can perform inference using the system, by enabling `"execute_dlls": false`. Set global setting `"backend": "tvm"` to run it via TVM, or `"backend": "library"` to run it using the respective library. Libraries supported: Keras, PyTorch, TensorFlow/TFLite, ONNX (Using ONNX Runtime).

ConvRepAIr will generate 1 file per-input, containing the top-5 predictions, along with the execution time per-prediction at the bottom. In addition, you will find an execution_log.txt file in the aforementioned fonder, containing info about the run.

Console will indicate the status of the running model and update accordingly.

#### Fault Localization/Repair:
Set the global setting to `"model_repair_enabled": true`. Also, adjust the `model_repair_config` object to your needs, setting paths, options and settings for `Source` and `Target` models.

### Raw Results Data:
Accompanying our contribution, we provide raw data for our experiments.
In particular, we provide the errors detected and repaired for all our experiment sets.
Note that TF & TFLite were using the same preprocessing settings by definition.
Also, we provide the label outputs against ILSVRC2017 - which you can download from ImageNet upon request. You can find them inside `generated/<model_name>/model/data` folders for each model.
The folder contains data for each case for Source and Target preprocessing settings (apart from TF/TFLite, which had the same setting), as well as a layer_analysis folder, demonstrating the data for running activation analysis for "suspicious" layer order, for all cases related to parameters (weights and biases).

