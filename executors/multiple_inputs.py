import sys
sys.path.append("..") 

from .executor import Executor
from helpers import logger
from tvm import relay

class MultipleInputsExecutor(Executor):
    def __init__(self, models_data, images_data, connection_data, extra_folder = "multiple_datasets"):
        Executor.__init__(self, models_data, images_data, connection_data, extra_folder)

    def execute(self, remote):
        device_id = self.connection_data["id"] or 0
        timestamp_str = str(self.get_epoch_timestamp()) + "_" + self.connection_data["device_name"] + str(device_id)

        mutation_name = self.mutation_names[0]
        self.prepare(mutation_name, remote)

        mutation_name_extracted = mutation_name.replace(".tar", "").replace(".", "_")
        output_base_folder = self.output_images_base_folder + "/" + mutation_name_extracted + "/" + self.extra_folder + "/ts_" + timestamp_str + \
            ("_" + self.connection_id if self.connection_id != "local" else "")

        logger_instance = logger.Logger(output_base_folder)
        
        for image_folder in self.input_images_folders:
            start_timestamp = self.get_epoch_timestamp(False)
            images_dataset = self.get_last_folder(image_folder)
            print("Executing model " + mutation_name + ", execution timestamp: " + timestamp_str + ", dataset: " + images_dataset)
            self.process_images_with_io(image_folder, output_base_folder + "/" + images_dataset, self.model_name)
            end_timestamp = self.get_epoch_timestamp(False)
            logger_instance.log_time("Dataset "+ images_dataset + " execution time: ", end_timestamp - start_timestamp)
        
        return timestamp_str
