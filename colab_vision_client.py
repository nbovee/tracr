import os
import ColabVision.src.colab_vision as cv
import time

if __name__ == '__main__':
    client = cv.FileClient('localhost:8892')
    in_file_name = './tmp/morbius_in.jpg'
    # print(os.path.exists(in_file_name))
    results = client.upload(in_file_name)
    for i in results:
        val = results[i]
        if i not in ["uuid", "results"]:
            val = float(val)
            print(f"{i} : {results[i]:.04f}")
        else:
            print(f"{i} : {results[i]}")
