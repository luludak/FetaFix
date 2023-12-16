
import os
from .comparator import *
import scipy.stats as stats


class Evaluator:
    def __init__(self, topK=5):
        self.topK = topK
        pass

    def evaluate_objects(self, source_object, target_object, off_by_one=False):
        source_preds = source_object
        target_preds = target_object

        if(off_by_one):
            target_preds = [int(t) - 1 for t in target_preds]

        #first_only = 1 if (source_preds[0] == target_preds[0] or source_preds[0] + 1 == target_preds[0] or source_preds[0] - 1 == target_preds[0]) else 0

        first_only = 1 if (source_preds[0] == target_preds[0]) else 0
        
        tau, p_value = stats.kendalltau(source_preds, target_preds)

        return {
            "base_label1": source_preds[0],
            "eval_label1": target_preds[0],
            "comparisons": {
               "kendalltau": {
                    "tau": tau,
                    "p-value": p_value
                },
                "first_only": first_only
            }
            #             "base_label1": source_preds[0],
            # "eval_label1": target_preds[0],
            # "comparisons": {
            #     "jaccard": str(jaccard_similarity(source_preds, target_preds)),
            #     "euclideanDistance" :  str(euclidean_distance(source_preds, target_preds)),
            #     "manhattanDistance": str(manhattan_distance(source_preds, target_preds)),
            #     "minkowskiDistance": str(minkowski_distance(source_preds, target_preds, 1.5)), # p values: 1 is for Manhattan, 2 is for Euclidean. Set it in between.
            #     "kendalltau": {
            #         "tau": tau,
            #         "p-value": p_value
            #     },
            #     "first_only": first_only,
            #     "rbo": str(rbo(source_preds, target_preds, 0.8))
            # }
        }


    def evaluate(self, original_file_path, mutant_file_path):

        original_obj, lines_no, original_first_line, exec_time1 = self.file_to_object(original_file_path)
        mutant_obj, mutant_lines_no, mutant_first_line, exec_time2 = self.file_to_object(mutant_file_path)
        
        original_keys = map(lambda x: int(x), original_obj.keys())
        mutant_keys = map(lambda x: int(x), mutant_obj.keys())

        total_value = 0

        first_only = 1 if (original_first_line == mutant_first_line) else 0

        for mutant_class in mutant_obj:
            class_value = 0

            if mutant_class in original_obj:
                
                class_value = 1 - float(abs(original_obj[mutant_class]["order"] - mutant_obj[mutant_class]["order"])) / (lines_no*2)
            total_value += class_value

        total_value_percentage = (float(total_value)/lines_no)*100
        original_list = list(original_obj.keys())
        mutants_list = list(mutant_obj.keys())
        tau, p_value = stats.kendalltau(original_list, mutants_list)
        

        return {
            "path_to_file": mutant_file_path,
            "base_comparison_file": original_file_path,
            "base_label1": original_first_line,
            "eval_label1": mutant_first_line,
            "base_exec_time": exec_time1,
            "exec_time": exec_time2,
            "comparisons": {
                "jaccard": str(jaccard_similarity(original_obj.keys(), mutant_obj.keys())),
                "euclideanDistance" :  str(euclidean_distance(original_keys, mutant_keys)),
                "manhattanDistance": str(manhattan_distance(original_keys, mutant_keys)),
                "minkowskiDistance": str(minkowski_distance(original_keys, mutant_keys, 1.5)), # p values: 1 is for Manhattan, 2 is for Euclidean. Set it in between.
                "kendalltau": {
                    "tau": str(tau),
                    "p-value": str(p_value)
                },
                "custom": str(total_value_percentage),
                "first_only": str(first_only),
                "rbo": str(rbo(list(original_obj.keys()), list(mutant_obj.keys()), 0.8))
            }
            
        }
        

    def compare_to_original(self, original_file_path, mutant_file_path):

        if (not os.path.exists(original_file_path) or not os.path.exists(mutant_file_path)):
            return False


        original_obj, lines_no, original_first_line, exec_time1 = self.file_to_object(original_file_path)
        mutant_obj, mutant_lines_no, mutant_first_line, exec_time2 = self.file_to_object(mutant_file_path)
        
        
        original_keys = map(lambda x: int(x), original_obj.keys())
        mutant_keys = map(lambda x: int(x), mutant_obj.keys())

        for mutant_class in mutant_obj:
            if mutant_class not in original_obj or (original_obj[mutant_class]["order"] != mutant_obj[mutant_class]["order"]):
                return False
        return True

    def reset_output_file(self, output_file_path):
        output_file = open(output_file_path, 'w')
        output_file.close()

    def file_to_object(self, path):
        obj = {}

        fileObj = open(path, 'r')

        first_class = None

        lines = fileObj.readlines()
        order = 1

        # Set this for Top-K (default: 5).
        count = 0
        exec_time = 0.0
        for line in lines:
            if count < self.topK:
                line_split = line.split(", ")
                class_name = line_split[0]
                
                # Comparison of libraries (does not apply in conversions).
                if(not "_to_" in path):

                    # Torch and Keras-related models correspond to N classes, while TF/TFLite to N+1,
                    # Therefore we subtract 1 from them.
                    if (not "torch" in path and not "keras" in path):
                	    class_name = str(int(class_name) - 1)

                if (count == 0):
                    first_class = class_name
                    
                class_prob = line_split[1]
                obj[class_name] = {
                    "probability": float(class_prob),
                    "order": int(order)
                }
                order += 1
            elif(count == self.topK + 1):
                exec_time = float(line)
            count += 1


        return obj, order - 1, first_class, exec_time
