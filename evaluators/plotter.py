
import json
import os
from os import path, listdir
from os.path import isdir, join
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from itertools import cycle

# Note: This class is deprecated.

class Plotter:

    def __init__(self, outpot_plot_file_name, outpot_plot_hist_file_name, output_metadata_file_name = "metadata.txt", **kwargs):
        # Default Settings
        self.outpot_plot_file_name = outpot_plot_file_name
        self.outpot_plot_hist_file_name = outpot_plot_hist_file_name
        self.plot_width = 300
        self.hist_plot_width = 100
        self.plot_height = 100
        self.plot_dpi = 60
        self.plot_size = 1024
        self.x_dim_start = 1 
        self.y_dim_start = -100
        self.y_dim_end = 100
        self.family="serif"
        self.title_fontsize = 30
        self.fontsize = 10
        self.large_fontsize = 150
        self.large_title_fontsize = 200
        self.yticks = [-100, -75, -50, -25, 0, 25, 50, 75, 100]
        self.output_plot_file_name = outpot_plot_file_name
        self.xticks_hist = [0, 5, 10]
        self.output_metadata_file_name = output_metadata_file_name
        
        self.no_of_devices = 5
        self.samples_no = 5500
        self.x_dim_end = 100
        self.alpha = 0.5

        # Override default settings set above if needed using kwargs.
        for key in kwargs:
            value = kwargs[key]
            setattr(self, key, value)

    def plot_dir(self, evaluations_root_dir, output_path, evaluation_file="evaluate_mutation.txt", plot_title="Mutations Evaluation", 
        exclude_list=None):
        print("Plotting multiple files....")
        subdir_names  = [d for d in listdir(evaluations_root_dir) if isdir(join(evaluations_root_dir, d))]
        evaluation_files_arr = [{"file_path": evaluations_root_dir + "/" + dir_name + "/" + evaluation_file, "model_name": dir_name,
            "title": dir_name.replace("model_", "").replace("_", " ")} for dir_name in subdir_names]
        self.plot(evaluation_files_arr, output_path, plot_title, exclude_list)

    def plot(self, evaluations, output_path, plot_title="Mutations Evaluation", exclude_list=None):

        # Calculate x ticks. Example: on 100 samples, you should have [25, 50, 75, 100]
        xticks = [1]
        step = self.samples_no // 4
        for i in range(step, self.samples_no, step):
            xticks.append(i)
        xticks.append(self.samples_no)

        plt.figure(figsize=(self.plot_width, self.plot_height), dpi=self.plot_dpi)
        plt.axis([self.x_dim_start, self.x_dim_end, self.y_dim_start, self.y_dim_end])
        plt.title(plot_title, fontsize=self.large_title_fontsize)
        plt.xlabel("Samples", fontsize=self.large_title_fontsize)
        plt.ylabel("Similarity %", fontsize=self.large_title_fontsize)
        plt.xticks(xticks, fontsize=self.large_fontsize)
        plt.yticks(self.yticks, fontsize=self.large_fontsize)

        # Generate different colors based on no of evaluations.
        colors = iter(cm.rainbow(np.linspace(0, 1, len(evaluations))))
        
        
        #parent_folder = os.path.abspath(os.path.join(output_plot_path, os.pardir))

        os.makedirs(output_path, exist_ok=True)
        output_metadata_path = join(output_path, self.output_metadata_file_name)
        output_plot_path = join(output_path, self.outpot_plot_file_name)
        evaluation_data = {}

        color_dict = {}
        colors = iter(cm.viridis(np.linspace(0, 1, len(evaluations))))


        for evaluation in evaluations:

            evaluation_file_path = evaluation["file_path"]
            evaluation_title = evaluation["title"]
            evaluation_model_name = evaluation["model_name"]
            color_dict[evaluation_model_name] = next(colors)

            # Skip original
            if exclude_list is not None and evaluation_model_name in exclude_list:
                continue

            if(not path.exists(evaluation_file_path)):
                print("Warning: Evaluate file path " + evaluation_file_path + " does not exist. Skipping evaluation.....")
                continue
            print(evaluation_file_path)
            with open(evaluation_file_path) as f:
                evaluation_data = json.load(f)

            x_coord = []
            y_coord = []
            count = 1

            metadata = evaluation_data["metadata"]
            for key in evaluation_data:
                if (key == "metadata"):
                    continue
                entry = evaluation_data[key]
                x_coord.append(count)
                val = (float(entry["comparisons"]["rbo"])*100)
                y_coord.append(val)
                count += 1

            with open(output_metadata_path, 'a+') as outfile:
                metadata["title"] = plot_title + ": " + evaluation_title
                metadata["model_name"] = evaluation_model_name
                metadata["device"] =  evaluation["device"]
                print(metadata, file=outfile)
                outfile.close()

            plt.scatter(x_coord, y_coord, self.plot_size, color=color_dict[evaluation_model_name], label=evaluation_model_name, alpha=self.alpha, marker=".")

            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fontsize=self.large_fontsize, shadow=True, ncol=5)
        plt.savefig(output_plot_path)
        plt.close()
        print("Plot exported at " + output_plot_path)


    def plot_gpus(self, evaluation_file_path, output_path, keys_threshold=0, plot_title="Mutations Evaluation", exclude_list=None):

        xticks = [1]
        step = self.samples_no // 4
        for i in range(step, self.samples_no, step):
            xticks.append(i)
        xticks.append(self.samples_no)

        # Calculate x ticks. Example: on 100 samples, you should have [25, 50, 75, 100]
        plt.figure(figsize=(self.hist_plot_width, self.plot_height), dpi=self.plot_dpi)
        plt.title(plot_title, fontsize=self.large_title_fontsize)
        plt.xlabel("Output Groups", fontsize=self.large_title_fontsize)
        plt.ylabel("Samples", fontsize=self.large_title_fontsize)

        plt.xticks(xticks, fontsize=self.fontsize)
        plt.yticks(self.yticks, fontsize=self.fontsize)
        # Generate different colors based on no of evaluations.


        output_plot_path = join(output_path, self.outpot_plot_file_name)
        output_plot_hist_path = join(output_path, self.outpot_plot_hist_file_name)
        output_metadata_path = join(output_path, self.output_metadata_file_name)
        parent_folder = os.path.abspath(os.path.join(output_plot_path, os.pardir))
        os.makedirs(parent_folder, exist_ok=True)

        if(not path.exists(evaluation_file_path)):
            print("Warning: Evaluate file path " + evaluation_file_path + " does not exist. Skipping evaluation.....")
            return None
        
        y_coord_hist = []
        with open(evaluation_file_path) as f:
            evaluation_data = json.load(f)

            devices = [f for f in evaluation_data["devices"] if exclude_list is None or f not in exclude_list]
            colors = iter(cm.rainbow(np.linspace(0, 1, len(devices))))
            color_dict = {}
            for device in devices:
                color_dict[device] = next(colors)

            count = 1
            count_threshold = 0
            devices_elems = {}

            for entry in evaluation_data:
                if entry == "devices":
                    continue
                
                entry_data = evaluation_data[entry]
                if(len(entry_data.keys()) > keys_threshold):
                    count_threshold += 1
                    y_coord_hist.append(len(entry_data.keys()))

                            
                count += 1

            yticks_hist = []
            yticklabels_hist = []
            step = self.samples_no // 4
            for i in range(step, self.samples_no, step):
                yticks_hist.append(i)
                yticklabels_hist.append(i)
            yticks_hist.append(self.samples_no)
            yticklabels_hist.append(self.samples_no)

            plt.hist(y_coord_hist, bins=np.arange(self.no_of_devices + 1) - 0.5, edgecolor='black', align="mid", color="navy")
            plt.xlim([0, self.no_of_devices])
            plt.xticks(range(0, self.no_of_devices + 1), range(0, self.no_of_devices + 1), fontsize=self.large_fontsize)
            plt.yticks(yticks_hist, yticklabels_hist, fontsize=self.large_fontsize)

            plt.savefig(output_plot_hist_path)
            plt.close()
            
            print("Plot exported at " + output_plot_path)




    def plot_models_gpus(self, data, output_path, keys_threshold=0, plot_title="Mutations Evaluation", exclude_list=None):


        plt.figure(figsize=(self.plot_width, self.plot_height), dpi=self.plot_dpi)
        plt.title(plot_title, fontsize=self.large_title_fontsize)
        plt.xlabel("Output Groups", fontsize=self.large_title_fontsize)
        plt.ylabel("Samples", fontsize=self.large_title_fontsize)

        yticks_hist = []
        yticklabels_hist = []
        step = self.samples_no // 4
        for i in range(step, self.samples_no, step):
            yticks_hist.append(i)
            yticklabels_hist.append(i)
            plt.axhline(i, linestyle='--')
        yticks_hist.append(self.samples_no)
        yticklabels_hist.append(self.samples_no)
        plt.axhline(self.samples_no, linestyle='--')


        output_plot_path = join(output_path, self.outpot_plot_file_name)
        output_plot_hist_path = join(output_path, self.outpot_plot_hist_file_name)
        output_metadata_path = join(output_path, self.output_metadata_file_name)
        parent_folder = os.path.abspath(os.path.join(output_plot_path, os.pardir))
        os.makedirs(parent_folder, exist_ok=True)

        y_data = []

        model_evaluation_paths = data["model_evaluation_paths"]
        model_names = data["models"]
        color_dict = {}
        color_arr = []
        colors = iter(cm.viridis(np.linspace(0, 1, len(model_names))))

        for name in model_names:
            color = next(colors)
            color_dict[name] = color
            color_arr.append(color) 

        for evaluation_file_path in model_evaluation_paths:

            if(not path.exists(evaluation_file_path)):
                print("Warning: Evaluate file path " + evaluation_file_path + " does not exist. Skipping evaluation.....")
                return None
            
            y_coord_hist = []
            devices = []
            
            with open(evaluation_file_path) as f:
                evaluation_data = json.load(f)

                devices = [f for f in evaluation_data["devices"] if exclude_list is None or f not in exclude_list and f not in devices]

                count = 1
                count_threshold = 0
                devices_elems = {}

                for entry in evaluation_data:
                    if entry == "devices":
                        continue
                    
                    entry_data = evaluation_data[entry]
                    # If greater than keys_threshold, plot it.

                    if(len(entry_data.keys()) > keys_threshold):
                        count_threshold += 1

                        y_coord_hist.append(len(entry_data.keys()))

                                
                    count += 1

                for dev in devices_elems:
                    if exclude_list is not None and dev in exclude_list:
                        continue


            y_data.append(y_coord_hist)
            
        
        plt.xticks(range(0, self.no_of_devices + 1), fontsize=self.large_fontsize)
        plt.xlim([0, self.no_of_devices])
        plt.yticks(yticks_hist, yticklabels_hist, fontsize=self.large_fontsize)
        plt.hist(y_data, bins=np.arange(self.no_of_devices + 1) - 0.5, edgecolor='black', label=model_names, color=color_arr)
        plt.legend(loc="upper right", fontsize=self.large_fontsize, shadow=True, ncol=2)

        plt.savefig(output_plot_hist_path)
        plt.close()
        
        print("Multi-GPU Plot exported at " + output_plot_path)