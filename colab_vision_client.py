from doctest import testfile
import os
import ColabVision.src.colab_vision as cv

def demo_funct():
    client = cv.FileClient('localhost:8892')

    # demo for file uploading
    in_file_name = './tmp/morbius_in.jpg'
    # print(os.path.exists(in_file_name))
    response = client.upload(in_file_name)
    # print(response)
    # demo for file downloading:
    out_file_name = './tmp/morbius_out.jpg'
    if os.path.exists(out_file_name):
        os.remove(out_file_name)
    test_dict = client.processingTime(response.id)
    dl_time_s = cv.timer()
    client.download(response.id, out_file_name)
    test_dict['download'] = cv.timer() - dl_time_s
    # os.system(f'sha1sum {in_file_name}')
    # os.system(f'sha1sum {out_file_name}')
    # print(test_dict)
    return test_dict
    

if __name__ == '__main__':
    _ = demo_funct()
