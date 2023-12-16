import os
from os import listdir
from os.path import isfile, join
from PIL import Image  
import PIL
from numpy import asarray
from imagecorruptions import corrupt

class Corruptor:

    def __init__(self):
        pass

    def extract_image_names(self, folder_path):
        return [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

    def corrupt_images(self, image_folder, image_output_folder, params):
        image_names = self.extract_image_names(image_folder)

        output_folder = image_output_folder + "/" + params["corruption_name"].replace(" ", "_") + "_" + str(params["severity"])
        os.makedirs(output_folder, exist_ok=True)
        
        for image_name in image_names:
            
            image = Image.open(image_folder + "/" + image_name)
            image_asarray = asarray(image)
            corrupted_image_array = corrupt(image_asarray, corruption_name=params["corruption_name"], severity=params["severity"])
            corrupted_image = Image.fromarray(corrupted_image_array)

            # TODO: Add exception handling
            output_image_path = output_folder + "/" + image_name
            print("Generating corrupt image at : " + output_image_path)
            corrupted_image.save(output_image_path)
