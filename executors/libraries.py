import sys
sys.path.append("..")

from .executor import Executor
from helpers import logger

import os
import math

class LibrariesExecutor(Executor):
    def __init__(self, models_data, images_data, connection_data, extra_folder = "mutations", specific_images = None):
        Executor.__init__(self, models_data, images_data, connection_data, extra_folder)
        self.specific_images = specific_images

    def execute(self):

        start_timestamp = self.get_epoch_timestamp(False)
        library_extracted = self.library.replace(".", "_")
        timestamp_str = str(math.floor(start_timestamp))
        model_name_prettified = self.model_name.replace(".", "_")

        output_images_base_folder = self.output_images_base_folder + "/libraries/" + model_name_prettified + "_" + library_extracted + "/" + self.extra_folder + "/" + "/ts_" + timestamp_str

        self.process_library_images_with_io(self.input_images_folders[0], output_images_base_folder, specific_images = self.specific_images)

        end_timestamp = self.get_epoch_timestamp(False)
        logger_instance = logger.Logger(output_images_base_folder)
        logger_instance.log_time("Model " + self.model_name + " execution time: ", end_timestamp - start_timestamp)

        timestamp_str = str(self.get_epoch_timestamp()) + "_" + self.library

        return {"timestamp_str": timestamp_str}
