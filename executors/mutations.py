import sys
sys.path.append("..")

from .executor import Executor
from helpers import logger

import os

class MutationsExecutor(Executor):
    def __init__(self, models_data, images_data, connection_data, extra_folder = "mutations"):
        Executor.__init__(self, models_data, images_data, connection_data, extra_folder)

    def execute(self, remote):
        device_id = self.connection_data["id"] or 0
        timestamp_str = str(self.get_epoch_timestamp()) + "_" + self.connection_data["device_name"] + str(device_id)

        for mutation_name in self.mutations_names:
            start_timestamp = self.get_epoch_timestamp(False)
            self.prepare(mutation_name, remote)

            print("Executing model " + mutation_name + ", execution timestamp: " + timestamp_str)

            mutation_name_extracted = mutation_name.replace(".tar", "").replace(".", "_")
            output_images_base_folder = self.output_images_base_folder + "/" + mutation_name_extracted + "/" + self.extra_folder + "/" + "/ts_" + timestamp_str + \
                ("_" + self.connection_id if self.connection_id != "local" else "")
                
            logger_instance = logger.Logger(output_images_base_folder)
        
            self.process_images_with_io(self.input_images_folders[0], output_images_base_folder, self.model_name, mutation_name)
            end_timestamp = self.get_epoch_timestamp(False)
            logger_instance.log_time("Model " + mutation_name_extracted + " execution time: ", end_timestamp - start_timestamp)

        return {"timestamp_str": timestamp_str}
