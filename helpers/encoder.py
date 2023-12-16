import tvm
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, tvm.nd.NDArray):
            return obj.numpy().tolist()
        return json.JSONEncoder.default(self, obj)