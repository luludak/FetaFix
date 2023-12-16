import sys
sys.path.append("..") 

from .executor import Executor
from helpers import logger

class RepetitionExecutor(Executor):
    def __init__(self, models_data, images_data, connection_data, no_of_repetitions = 1, extra_folder = "repetitions"):
        Executor.__init__(self, models_data, images_data, connection_data, extra_folder)
        self.no_of_repetitions = no_of_repetitions


    def execute(self, remote):
        device_id = self.connection_data["id"] or 0
        timestamp_str = str(self.get_epoch_timestamp()) + "_" + self.connection_data["device_name"] + str(device_id)

        mutation_name = self.mutation_names[0]
        self.prepare(mutation_name)

        mutation_name_extracted = mutation_name.replace(".tar", "").replace(".", "_")
        output_images_base_folder = self.output_images_base_folder + "/" + mutation_name_extracted + "/" + self.extra_folder + "/ts_" + timestamp_str + \
            ("_" + self.connection_id if self.connection_id != "local" else "")

        logger_instance = logger.Logger(output_images_base_folder)
        
        for i in range(1, self.no_of_repetitions + 1):
            start_timestamp = self.get_epoch_timestamp(False)
            print("Executing model " + mutation_name + ", execution timestamp: " + timestamp_str + ", iteration: " + str(i))
            output_folder = output_images_base_folder + "/" + str(i)
            self.process_images_with_io(self.input_images_folders[0], output_folder, self.model_name)
            end_timestamp = self.get_epoch_timestamp(False)
            logger_instance.log_time("Iteration " + str(i) + " execution time: ", end_timestamp - start_timestamp)
            
        return timestamp_str
