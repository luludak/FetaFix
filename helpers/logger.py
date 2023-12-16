
class Logger:

    def __init__(self, folder_path):
        self.file_path = folder_path + "/execution_log.txt"

    # Important Note: We utilize logging that depends (slightly) on minor ops, such as print, but also internal ops
    # such as file save. All this, in combination with computer resources utilization, can seriously affect execution time
    # and must be considered in any evaluation scenario, for all dependencies that utilize this.
    def log_time(self, message, time_diff):
        with open(self.file_path, 'a+') as output_file:
            print(message + " " + str(time_diff) + " seconds", file = output_file)
            output_file.close()