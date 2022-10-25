import os
import src.colab_vision_client as cv
import time
import atexit
from test_data import test_data_loader as data_loader
if __name__ == '__main__':
    client = cv.FileClient('172.17.0.2:8893')
    atexit.register(client.safeClose)

    # Model = blah
    # client.setModel(Model)
    # data_loader = blah
    # client.setDataLoader(data_loader)

    # for i in range(num_tests):
    # print(os.path.exists(in_file_name))
    test_data = data_loader()
    while test_data.has_next():
        client.initiateInference(test_data)
        # overall += results["inference"]
        # for i in results:
        #     val = results[i]
        #     if i not in ["uuid", "results"]:
        #         val = float(val)
        #         print(f"{i} : {results[i]:.04f}")
        #     else:
        #         print(f"{i} : {results[i]}")
    # print(f"Average over {num_tests} runs: {overall/num_tests:0.04f}")
    # for i in results:
    #     val = results[i]
    #     if i not in ["uuid", "results"]:
    #         val = float(val)
    #         print(f"{i} : {results[i]:.04f}")
    #     else:
    #         print(f"{i} : {results[i]}")
