import os
import ColabVision.src.colab_vision as cv
import time

if __name__ == '__main__':
    client = cv.FileClient('localhost:8892')
    # in_file_name = './tmp/morbius_in.jpg'
    in_file_name = './tmp/9.png'
    overall = 0
    num_tests = 1
    # Model = blah
    # client.setModel(Model)
    # data_loader = blah
    # client.setDataLoader(data_loader)

    for i in range(num_tests):
    # print(os.path.exists(in_file_name))
        results = client.upload(in_file_name)
        overall += results["inference"]
        # for i in results:
        #     val = results[i]
        #     if i not in ["uuid", "results"]:
        #         val = float(val)
        #         print(f"{i} : {results[i]:.04f}")
        #     else:
        #         print(f"{i} : {results[i]}")
    print(f"Average over {num_tests} runs: {overall/num_tests:0.04f}")
    for i in results:
        val = results[i]
        if i not in ["uuid", "results"]:
            val = float(val)
            print(f"{i} : {results[i]:.04f}")
        else:
            print(f"{i} : {results[i]}")
