import json
import os
from os import path
from os import listdir
from os.path import isfile, isdir, join, exists
import matplotlib.pyplot as plt
import fnmatch
from evaluators.classification import Evaluator
import statistics
from scipy import stats
import math
import numpy as np
import csv

# TODO: Rewrite object keys usage.
class EvaluationGenerator:

    def __init__(self, out_path=None, original_model_name="model_original", 
    build_ts = "ts_1", exec_log_prefix = "execution_log", evaluate_log_prefix="evaluate"):
        self.evaluator = Evaluator()
        self.exec_log_prefix = exec_log_prefix
        self.evaluate_log_prefix = evaluate_log_prefix
        self.paths_to_check_threshold = 100
        self.problematic_occurences_threshold = 90
        self.images_similar_sample_no = 10
        self.images_dissimilar_sample_no = 10
        self.threshold = 0.7

    def count_problematic_occurences(self, original_path, samples_paths):

        problematic_count = 0
        for sample_path in samples_paths:
            if(self.evaluator.compare_to_original(original_path, sample_path)):
               problematic_count += 1 
        
        # If problematic occurences over threshold, mark as problematic.
        return problematic_count


    def generate_base_folder_comparison(self, model_names, base_folder, mutations_folder):

        for model_name in model_names:
            new_base_folder = join(base_folder, model_name, mutations_folder) 
            print("Evaluating folder " + new_base_folder + ".")
            
            if(exists(new_base_folder)):
                evaluation_data_obj = self.get_same_folder_comparison(new_base_folder)              
                output_path_file = new_base_folder + "/same_folder_evaluation.json"

                with open(output_path_file, 'w') as outfile:
                    print(json.dumps(evaluation_data_obj, indent=2, sort_keys=True), file=outfile)
                    outfile.close()


    def get_time_stats_folder(self, base_folder, base_case=None):

        exec_times = []
        folder_inner  = join(base_folder, [d for d in listdir(base_folder) if isdir(join(base_folder, d))][0])
        evaluation_folders = [d for d in listdir(folder_inner) if isdir(join(folder_inner, d))]
        for evaluation_folder in evaluation_folders:
            total_path = join(folder_inner, evaluation_folder)
            comparison_stats = self.get_basic_evaluation(total_path, total_path, total_path, False)
            exec_times.append(comparison_stats["comparison"]["orig_total_exec_time"])

        print("***** Time Analysis for " + folder_inner + ": *****")
        print("Mean Variance:" + str(float(np.var(exec_times))))
        print("STDev:" + str(float(np.std(exec_times))))
        print("Mean:" + str(float(np.mean(exec_times))))
        np_mean = np.mean(exec_times)
        if (np_mean != 0):
            print("Percentage:" + str(float(np.std(exec_times))/float(np_mean)))

        return self

    def get_same_folder_comparison(self, base_folder, base_case=None):

        evaluation_folders  = [d for d in listdir(base_folder) if isdir(join(base_folder, d))]
        if evaluation_folders == None or len(evaluation_folders) == 0:
            return

        evaluation_folders_sorted = sorted(evaluation_folders)
        evaluation_folders = evaluation_folders_sorted[-10:] if (len(evaluation_folders_sorted) >= 10) else evaluation_folders_sorted
        comparisons = {}
        times = {}
        total_images_dissimilar = {}
        total_avg_exec_time = 0

        base_folders = evaluation_folders if base_case is None else [base_case]

        for base in base_folders:

            if (base not in comparisons):
                comparisons[base] = {}

            for evaluated in evaluation_folders:

                if (base == evaluated):
                    continue

                comparison_stats = self.get_basic_evaluation(join(base_folder, base), join(base_folder, evaluated), base_folder, False)
                if(comparison_stats is None):
                    continue
                evaluated_avg_time = comparison_stats["comparison"]["average_exec_time"]
                evaluated_total_time = comparison_stats["comparison"]["total_exec_time"]
                images_dissimilar = comparison_stats["comparison"]["images_dissimilar"]
                del comparison_stats["comparison"]["average_exec_time"]
                del comparison_stats["comparison"]["total_exec_time"]
                if(evaluated not in times):
                    times[evaluated] = {
                        "average_exec_time": evaluated_avg_time,
                        "total_exec_time": evaluated_total_time
                    }
                    total_avg_exec_time += evaluated_avg_time
                comparisons[base][evaluated] = comparison_stats["comparison"]
                # comparisons.append({
                #     "base": base,
                #     "evaluated": evaluated,
                #     "comparison": comparison_stats["comparison"]
                # })

                for image_dissimilar in images_dissimilar:
                    if image_dissimilar not in total_images_dissimilar:
                        total_images_dissimilar[image_dissimilar] = 0
                    else:
                        total_images_dissimilar[image_dissimilar] += 1 

        evaluation_folders_len = len(evaluation_folders) if len(evaluation_folders) > 0 else 1
        full_object = {
            "times": times,
            "total_avg_exec_time": total_avg_exec_time / evaluation_folders_len,
            "comparisons": comparisons,
            "total_images_dissimilar": total_images_dissimilar
        }

        return full_object      

    def generate_devices_comparison(self, base_folder, replace_evaluated_suffix=False):
        devices = [d for d in listdir(base_folder) if isdir(join(base_folder, d))]

        comparisons = {}
        times = {}
        total_images_dissimilar_across_devices = {}
        total_csv_obj = []
        print("Generating Comparison for " + base_folder)

        for base in devices:

            libraries = [l for l in listdir(join(base_folder, base)) if isdir(join(base_folder, base, l))]

            for library in libraries:

                if(not library in comparisons):
                    comparisons[library] = []

                base_lib_path = join(base_folder, base, library)
                if (not exists(base_lib_path) or not exists(join(base_lib_path, "mutations"))):
                    print("Base Path does not exist. Skipping.....")
                    continue

                folder_base = [db for db in listdir(join(base_lib_path, "mutations")) if isdir(join(base_lib_path, "mutations", db))]

                if(len(folder_base) == 0):
                    continue

                last_base = folder_base[-1]

                for evaluated in devices:

                    evaluated_lib_path = join(base_folder, evaluated)


                    if (base == evaluated):
                        continue

                    elif (not isdir(join(evaluated_lib_path, "mutations"))):
                        print("Evaluated mutations folder does not exist. Skipping.....")
                        continue

                    folder_eval = [db for db in listdir(join(evaluated_lib_path, "mutations")) if isdir(join(evaluated_lib_path, "mutations", db))]
                    
                    if(len(folder_eval) == 0):
                        print("Evaluated folder is empty. Skipping.....")
                        continue
                    
                    last_eval = folder_eval[-1] 


                    comparison_stats = self.get_basic_evaluation(join(base_lib_path, "mutations", last_base), join(evaluated_lib_path, "mutations", last_eval), base_folder, False)
                    
                    if(comparison_stats is None):
                        continue

                    images_dissimilar = comparison_stats["comparison"]["images_dissimilar"]
                    for image_dissimilar in images_dissimilar:
                        if image_dissimilar not in total_images_dissimilar_across_devices:
                            total_images_dissimilar_across_devices[image_dissimilar] = 0
                        else:
                            total_images_dissimilar_across_devices[image_dissimilar] += 1 

                    comparisons[library].append({
                        "base": base,
                        "evaluated": evaluated,
                        "comparison": comparison_stats["comparison"]
                    })

                    stats = comparison_stats["comparison"]["oneway_exec_time"]
                    if stats["p-value"] is not None and float(stats["p-value"]) > 0.05:
                        
                        total_csv_obj.append({
                            "lib": library,
                            "device1" : base,
                            "device2": evaluated,
                            "p-value" : stats["p-value"],
                            "statistic" : stats["statistic"]
                        })
        
        comparisons["total_images_dissimilar"] = total_images_dissimilar_across_devices

        output_path_file = base_folder + "/device_evaluation.json"
        
        type_under_test = base_folder.split("/")[-1] if not base_folder.endswith("/") else base_folder.split("/")[-2]    

        output_stats_total_csv = base_folder + "/" + type_under_test + "_" + "output_stats_total_1.csv"

        with open(output_path_file, 'w') as outfile:
            print(json.dumps(comparisons, indent=2, sort_keys=True), file=outfile)
            outfile.close()

        
        csv_fields = ['lib', 'device1', 'device2', 'p-value', 'statistic']


        with open(output_stats_total_csv, 'a', newline='') as file: 
            writer = csv.DictWriter(file, fieldnames=csv_fields)
            writer.writerows(total_csv_obj)


    def generate_libraries_comparison(self, base_folder, base_case_folder=None, device_folder=None):

        libraries = [l for l in listdir(base_folder) if isdir(join(base_folder, l))]

        comparisons = {}
        times = {}
        total_images_dissimilar = {}
        images_similar_on_all_comparisons = {}
        images_dissimilar_on_all_comparisons = {}
        total_exec_time_percentages = {}
        total_exec_time_stats = {}
        total_csv_obj = []

        print ("Generating Library Comparisons for " + base_folder + " - Device " + device_folder)

        base_libraries = libraries if base_case_folder is None else [base_case_folder]

        for base in base_libraries:
            
            if (base not in comparisons):
                comparisons[base] = {}

            base_mutations = join(base_folder, base, "mutations")
            if(not exists(base_mutations)):
                continue
            
            # Skip using converted models as base.
            if ("_to_" in base):
                continue

            folder_base = sorted([db for db in listdir(base_mutations) if isdir(join(base_mutations, db))])
            if(len(folder_base) == 0):
                continue

            last_base = folder_base[-1]
            
            if(base not in total_exec_time_percentages):
                total_exec_time_percentages[base] = {}
                total_exec_time_stats[base] = {}

            if(base not in images_dissimilar_on_all_comparisons):
                images_dissimilar_on_all_comparisons[base] = {}
            if(base not in images_similar_on_all_comparisons):
                images_similar_on_all_comparisons[base] = {}

            for evaluated in libraries:

                if (base == evaluated):
                    continue
            
                eval_mutations = join(base_folder, evaluated, "mutations")
                if(not exists(eval_mutations)):
                    continue

                last_eval = sorted([d for d in listdir(eval_mutations) if isdir(join(eval_mutations, d))])[-1]

                comparison_stats = self.get_basic_evaluation(join(base_mutations, last_base), join(eval_mutations, last_eval), base_folder, False, dissimilar_images_max_no=-1, similar_images_max_no=-1)
                if(comparison_stats is None):
                    continue

                exec_time_percentages = comparison_stats["comparison"]["exec_time_percentages"]
                total_exec_time_percentages[base][evaluated] = exec_time_percentages
                total_exec_time_percentages[base]["oneway_exec_time"] = comparison_stats["comparison"]["oneway_exec_time"]

                total_exec_time_stats[base][evaluated] = comparison_stats["comparison"]["oneway_exec_time"] 

                evaluated_avg_time = comparison_stats["comparison"]["average_exec_time"]
                evaluated_total_time = comparison_stats["comparison"]["total_exec_time"]
                images_similar = comparison_stats["comparison"]["images_similar"]
                images_dissimilar = comparison_stats["comparison"]["images_dissimilar"]

                # Just rename the key for similar images.
                comparison_stats["comparison"]["images_similar_sample"] = images_similar[:self.images_similar_sample_no]
                comparison_stats["comparison"]["images_dissimilar_sample"] = images_dissimilar[:self.images_dissimilar_sample_no]
                images_similar_on_all_comparisons[base][evaluated] = images_similar
                images_dissimilar_on_all_comparisons[base][evaluated] = images_dissimilar
                
                del comparison_stats["comparison"]["images_similar"]
                del comparison_stats["comparison"]["images_dissimilar"]
                del comparison_stats["comparison"]["average_exec_time"]
                del comparison_stats["comparison"]["total_exec_time"]
                del comparison_stats["comparison"]["exec_time_percentages"]

                if(evaluated not in times):
                    times[evaluated] = {
                        "average_exec_time": evaluated_avg_time,
                        "total_exec_time": evaluated_total_time
                    }

                comparisons[base][evaluated] = comparison_stats["comparison"]

                if total_exec_time_stats[base][evaluated]["p-value"] is not None and float(total_exec_time_stats[base][evaluated]["p-value"]) > 0.05:
                        
                    total_csv_obj.append({
                        "device": device_folder,
                        "lib1": base,
                        "lib2": evaluated,
                        "p-value": total_exec_time_stats[base][evaluated]["p-value"],
                        "statistic": total_exec_time_stats[base][evaluated]["statistic"]
                    })
                
                for image_dissimilar in images_dissimilar:
                    if image_dissimilar not in total_images_dissimilar:
                        total_images_dissimilar[image_dissimilar] = 0
                    else:
                        total_images_dissimilar[image_dissimilar] += 1 
            full_object = {
                "times": times,
                "comparisons": comparisons,
                "total_images_dissimilar": total_images_dissimilar
            }

            output_path_file = base_folder + "/library_evaluation.json"
            output_path_file_images_similar = base_folder + "/library_evaluation_images_similar.json"
            output_path_file_images_dissimilar = base_folder + "/library_evaluation_images_dissimilar.json"
            output_exec_time_file = base_folder + "/library_evaluation_time_percentages.json"
            output_stats_time_file = base_folder + "/library_evaluation_time_stats.json"
            
            type_under_test = base_folder.split("/")[-2] if not base_folder.endswith("/") else base_folder.split("/")[-3]

            output_stats_total_csv = base_folder + "/../" + type_under_test + "_" + "output_stats_total_2.csv"


            csv_fields = ['device', 'lib1', 'lib2', 'p-value', 'statistic', 'percentage_similar']


            with open(output_stats_total_csv, 'a', newline='') as file: 
                writer = csv.DictWriter(file, fieldnames=csv_fields)
                writer.writerows(total_csv_obj)

            with open(output_path_file_images_similar, 'w') as outfile:
                print(json.dumps(images_similar_on_all_comparisons, indent=2, sort_keys=True), file=outfile)
                outfile.close()
                
            with open(output_path_file_images_dissimilar, 'w') as outfile:
                print(json.dumps(images_dissimilar_on_all_comparisons, indent=2, sort_keys=True), file=outfile)
                outfile.close()

            with open(output_path_file, 'w') as outfile:
                print(json.dumps(full_object, indent=2, sort_keys=True), file=outfile)
                outfile.close()

            with open(output_exec_time_file, 'w') as outfile:
                print(json.dumps(total_exec_time_percentages, indent=2, sort_keys=True), file=outfile)
                outfile.close()

            with open(output_stats_time_file, 'w') as outfile:
                print(json.dumps(total_exec_time_stats, indent=2, sort_keys=True), file=outfile)
                outfile.close()      

    def generate_folder_placeholder_comparison(self, model_names, folder1_path, folder2_path, output_path_file, label = "model"):

        for model_name in model_names:
            print("Evaluating model " + model_name + " (" + label + ").")
            folder1_path_gen = folder1_path.replace("{placeholder}", model_name)
            folder2_path_gen = folder2_path.replace("{placeholder}", model_name)
            output_path_gen = output_path_file.replace("{placeholder}", model_name)
            evaluation_data_obj = self.get_basic_evaluation(folder1_path_gen, folder2_path_gen, output_path_file, label)           


            # Create output folder if not os.path.exists.
            parent_folder = os.path.abspath(os.path.join(output_path_file, os.pardir))
            os.makedirs(parent_folder, exist_ok=True)

            with open(output_path_file, 'w') as outfile:
                print(json.dumps(evaluation_data_obj, indent=2, sort_keys=True), file=outfile)
                outfile.close()

    # Generate tau comparison for images.
    def generate_objects_comparison(self, source_object, target_object):

        # Loop images
        # Run object comparison.
        # Count similar & Dissimilar, along with distances, in object.
        # Trick: not all source images should be at target, but if a target image
        # is not found in source, it will crash.
        source_images = source_object.keys()
        target_images = target_object.keys()
        evaluation_object = {
            "images": {},
            "similar": [],
            "dissimilar": [],
        }
        images_similar = 0
        images_dissimilar = 0

        for image in target_images:

            source_img = source_object[image]
            target_img = target_object[image]

            if len(source_img) == 2 and source_img[0] == source_img[1]:
                source_img = source_img[0][0:len(target_img)]
                    
            elif len(target_img) == 2 and target_img[0] == target_img[1]:
                target_img = target_img[0][0:len(source_img)]

            
            image_evaluation = self.evaluator.evaluate_objects(source_img, target_img)
            evaluation_object["images"][image] = {
                "tau": image_evaluation["comparisons"]["kendalltau"]["tau"],
                "p-value": image_evaluation["comparisons"]["kendalltau"]["p-value"],
            }

            if image_evaluation["comparisons"]["first_only"] == 1:
                evaluation_object["similar"].append(image)
                images_similar += 1
            else:
                evaluation_object["dissimilar"].append(image)
                images_dissimilar += 1

        # and len(source_images) == len(target_images)
        if (len(source_images) != 0):
            evaluation_object["percentage_similar"] = (len(evaluation_object["similar"])/len(target_images)) * 100
            evaluation_object["percentage_dissimilar"] = (len(evaluation_object["dissimilar"])/len(target_images)) * 100
        return evaluation_object

    def get_basic_evaluation(self, folder1_path, folder2_path, output_path_file="", include_individual_analysis=True, write_to_file=False, similar_images_max_no=10, dissimilar_images_max_no=10, max_no_of_diff_labels=0, verbose_time_data=False):

        original_model_path = folder1_path
        mutation_model_path = folder2_path

        mut_no = len(fnmatch.filter(os.listdir(mutation_model_path), '*.txt')) 
        orig_no = len(fnmatch.filter(os.listdir(original_model_path), '*.txt'))


        if(not os.path.exists(original_model_path)):
            #print("Warning: original path " + original_model_path + " does not exist. Skipping evaluation.....")
            return
        elif(not os.path.exists(mutation_model_path)):
            #print("Warning: mutation path " + mutation_model_path + " does not exist. Skipping evaluation.....")
            return


        elif(abs(mut_no - orig_no) >= 5):
            #print("Warning: " + mutation_model_path + " contains different number of files than base folder. Skipping evaluation.....")
            return

        evaluation_data_obj = {}

        image_txt_names  = [f for f in listdir(original_model_path)
                            if isfile(join(original_model_path, f)) and f.endswith(".txt") and
                            f not in output_path_file and self.evaluate_log_prefix not in f and self.exec_log_prefix not in f and "Error" not in f]
        
        total_images_no = len(image_txt_names)
        images_similar_no = 0
        images_dissimilar = 0
        images_dissimilar_no = 0


        paths_to_check = [join(mutation_model_path, f) for f in image_txt_names]
        files_no_diff = abs(len(image_txt_names) - len(paths_to_check))

        # Check that NN is not "stuck" - producing the same results in every input.
        problematic_occurences = self.count_problematic_occurences(paths_to_check[0], paths_to_check[1:101]) if len(paths_to_check) > 100 else len(paths_to_check)
        if(len(paths_to_check) > self.paths_to_check_threshold and problematic_occurences > self.problematic_occurences_threshold):
            output_error_file = join(path.dirname(output_path_file), "Error.txt")
            error_text = mutation_model_path + " was found " + str(problematic_occurences) + "% problematic. Skipped generation of evaluation file.\n"
            error_text += mutation_model_path + " was found to have a different number of files by " + str(files_no_diff) + "."
            print(error_text)
            with open(output_error_file, 'w') as outfile:
                print (error_text, file=outfile)

                outfile.close()
            return
        total_exec_time = 0.0
        orig_total_exec_time = 0.0
        images_similar = []
        images_dissimilar = []
        exec_time_percentages = []
        total_count_of_labels = {}
        diff_labels = {}
        orig_exec_times = []
        mut_exec_times = []
         
        for image_txt in image_txt_names:

            original_img_file_path = join(original_model_path, image_txt)
            mutation_img_file_path = join(mutation_model_path, image_txt)
            
            if not path.isfile(mutation_img_file_path):
                print("- Warning: File " + mutation_img_file_path + " does not exist in model folder. Skipping...")
                continue

            image_name_extracted = image_txt.split('.')[0]
            evaluated = self.evaluator.evaluate(original_img_file_path, mutation_img_file_path)
            
            orig_exec_time = evaluated["base_exec_time"]
            orig_exec_times.append(orig_exec_time)
            mut_exec_time = evaluated["exec_time"]
            mut_exec_times.append(mut_exec_time)
            
            total_exec_time += mut_exec_time
            orig_total_exec_time += orig_exec_time
            comparison = float(evaluated["comparisons"]["first_only"])
            exec_percentage = ((mut_exec_time - orig_exec_time) / (orig_exec_time)) * 100
            exec_time_percentages.append(round(exec_percentage, 2))

            if evaluated["base_label1"] not in total_count_of_labels:
                total_count_of_labels[evaluated["base_label1"]] = 1
            else:
                total_count_of_labels[evaluated["base_label1"]] += 1

            if (comparison >= 0.7):
                if(similar_images_max_no == -1 or len(images_similar) < similar_images_max_no):
                    images_similar.append(image_txt)
                images_similar_no += 1
            else:
                images_dissimilar_no += 1
                if (evaluated["base_label1"] not in diff_labels):
                    diff_labels[evaluated["base_label1"]] = 1
                else:
                    diff_labels[evaluated["base_label1"]] += 1

                # Set flag to -1 in order to include all dissimilar images to analysis.
                if(dissimilar_images_max_no == -1 or len(images_dissimilar) < dissimilar_images_max_no):
                    images_dissimilar.append(image_txt)

            if(include_individual_analysis):
                evaluation_data_obj[image_name_extracted] = evaluated

        div_total_images_no = total_images_no if total_images_no > 0 else 1

        diff_labels = dict(sorted(diff_labels.items(), key=lambda item: item[1], reverse=True)[0:max_no_of_diff_labels])
        
        total_diff_label_info = {}
        for key in diff_labels:
            different = diff_labels[key]
            total_diff_label_info[key] = {
                "different": different,
                "total": total_count_of_labels[key],
                "percentage": (different/total_count_of_labels[key])*100
            }

	    # Calculate One-way ANOVA for execution time analysis.
        t_test_result = stats.f_oneway(orig_exec_times, mut_exec_times)        

        evaluation_data_obj["comparison"] = {
            "total_no_of_images": total_images_no,
            "no_of_images_similar": str(images_similar_no),
            "no_of_images_dissimilar": str((images_dissimilar_no)),
            "percentage_similar": str((images_similar_no / (div_total_images_no)) * 100),
            "percentage_dissimilar": str((images_dissimilar_no / div_total_images_no) * 100),
            "total_exec_time": total_exec_time,
            "orig_total_exec_time": orig_total_exec_time,
            "average_exec_time": total_exec_time / div_total_images_no,
            "average_orig_exec_time": orig_total_exec_time / div_total_images_no,
            "images_dissimilar": images_dissimilar,
            "images_similar": images_similar,
            "exec_time_percentages": exec_time_percentages,
            "diff_labels": total_diff_label_info,
            "oneway_exec_time": {
                "statistic": t_test_result[0] if not math.isnan(t_test_result[0]) else "NaN",
                "p-value": t_test_result[1] if not math.isnan(t_test_result[1]) else "NaN"
            }
        }

        if (verbose_time_data):
            evaluation_data_obj["comparison"]["times"] = {
                "orig_exec_times" : orig_exec_times,
                "mut_exec_times" : mut_exec_times
            }

        if(write_to_file):
            with open(output_path_file, 'w') as outfile:
                print(json.dumps(evaluation_data_obj, indent=2, sort_keys=True), file=outfile)
                outfile.close()

        return evaluation_data_obj