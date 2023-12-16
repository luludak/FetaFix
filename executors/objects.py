import sys
sys.path.append("..")

from .executor import Executor
from helpers import logger

import os

class ObjectsExecutor(Executor):
    def __init__(self, models_data, images_data, connection_data):
        Executor.__init__(self, models_data, images_data, connection_data, "mutations")

    def execute(self, remote, specific_images=None):
        device_id = self.connection_data["id"] or 0
        timestamp_str = str(self.get_epoch_timestamp()) + "_" + self.connection_data["device_name"] + str(device_id)

        start_timestamp = self.get_epoch_timestamp(False)
        self.prepare(remote=remote)

        print("Executing model " + self.name + ", execution timestamp: " + timestamp_str)

        # mutation_name_extracted = mutation_name.replace(".tar", "").replace(".", "_")
        # output_images_base_folder = self.output_images_base_folder + "/" + mutation_name_extracted + "/" + self.extra_folder + "/" + "/ts_" + timestamp_str + \
        #     ("_" + self.connection_id if self.connection_id != "local" else "")
            
        logger_instance = logger.Logger(self.output_images_base_folder)
    
        output_obj = self.process_images_with_io(self.input_images_folders[0], self.output_model_folder, self.model_name, "", specific_images, should_write_to_file=False)
        end_timestamp = self.get_epoch_timestamp(False)
        logger_instance.log_time("Model " + self.name + " execution time: ", end_timestamp - start_timestamp)
    
        output_obj["timestamp_str"] = timestamp_str

        return output_obj
