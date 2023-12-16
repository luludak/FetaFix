import os
from .comparator import *
import scipy.stats as stats

from executors import libraries
from evaluators import evalgenerator
from helpers import search

class ConversionSequenceAnalyzer:

    def __init__(sequences, images, paths):
        self.sequences = sequences
        self.script_dir = paths["script_dir"]
        self.images = images
        self.error_base_folder = paths["errors_folder"]
        self.models_folder = paths["models_folder"]
        self.inputs_folder = paths["inputs_folder"]
        self.outputs_folder = paths["outputs_folder"]
        self.input_analysis_file = paths["input_analysis_file"] if "input_analysis_file" in paths else None
        self.output_analysis_file = paths["output_analysis_file"] if "output_analysis_file" in paths else \
            join(self.outputs_folder, "analysis.json")
        

    def execute_sequence(key):

        sequence = self.sequences[key]


        if self.input_analysis_file is None: 
            specific_images = self.images[key]
        else:

            with open(self.input_analysis_file) as f:
                json_data = json.load(f)["comparisons"]

                source = key.split("_to_")[0]
                target = key

                search_inst = search.Search()
                source_obj = search_inst.searchKeysAndReturnValues(json_data, source, "_to_")[0]
                target_key = [k for k in source_obj.keys() if target in k][0]
                target_obj = source_obj[target_key]
                images_data = target_obj["images_dissimilar_sample"]
                

        images_data = {
            "input_images_folders": [self.inputs_folder],
            "output_images_base_folder": self.output_folder
        }

        connection_data = {
            "error_base_folder" : self.errors_folder
        }

        for model_obj in sequence:
            full_model_path = join(self.script_dir, self.models_folder, model_obj["model"])

            model_name = model_obj["model"]

            models_data = {
                "library" : model_name.split(".")[1],
                "model" : model_name,
                "input_name" : model_obj["input_name"],
                "output_name" : model_obj["output_name"],
                "input" : model_obj["input"],
                "output" : model_obj["output"],

            }

            executor = libraries.LibrariesExecutor(models_data, images_data, connection_data, specific_images = specific_images)

            executor.execute()


    def analyze_sequence(key):

        sequence = self.sequences[key]
        evaluation_generator = evalgenerator.EvaluationGenerator()
        comparisons = []
        output_file = join(self.output_analysis_file)
        for i in range (len() - 1):
            model_obj1 = sequence[i]
            model_obj2 = sequence[i + 1]

            model_obj1_name = model_obj1["model"].replace(".", "_")
            model_obj2_name = model_obj2["model"].replace(".", "_")
            model_obj1_path = join(self.output_folder, model_obj1_name)
            model_obj2_path = join(self.output_folder, model_obj2_name)

            evaluation = evaluation_generator.get_basic_evaluation(model_obj1_path, model_obj2_path)
            comparison["source"] = model_obj1_name
            comparison["target"] = model_obj2_name
            comparison["evaluation"] = evaluation
            comparisons.append(comparison)
            

        with open(output_file, 'w') as outfile:
            print(json.dumps(comparisons, indent=2, sort_keys=True), file=outfile)
            outfile.close()
