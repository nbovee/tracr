import os
import ColabVision.src.colab_vision as cv

def demo_funct():
    client = cv.FileClient('localhost:8888')

    # demo for file uploading
    in_file_name = './tmp/morbius_in.jpg'
    print(os.path.exists(in_file_name))
    client.upload(in_file_name)

    # demo for file downloading:
    out_file_name = './tmp/morbius_out.jpg'
    if os.path.exists(out_file_name):
        os.remove(out_file_name)
    client.download('whatever_name', out_file_name)
    os.system(f'sha1sum {in_file_name}')
    os.system(f'sha1sum {out_file_name}')
    return out_file_name

if __name__ == '__main__':
    _ = demo_funct()
