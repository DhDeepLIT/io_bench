import numpy as np
from libtiff import TIFF
from libtiff import TIFFfile, TIFFimage
import libtiff
from  argparse import ArgumentParser
import os

def build_fake_image(size, data_type):
    data = None
    if data_type == 'uint8':
        data = np.random.randint(0,255, size, data_type)

    return data

def run(config):
    dir_file = 'tmp_uint8'
    os.makedirs(dir_file, exist_ok=True)
    with open(config, 'rt') as filenames:
        file_name = filenames.readline()
        while file_name != '':
            if file_name[0] == "#":
                file_name = filenames.readline().rstrip('\n')
            else:
                
                print(f' {file_name} extracted from file')
                data_file = os.path.join(dir_file, file_name)
                print(f' {data_file} to be written')
                data = build_fake_image((256,256,3), 'uint8')
                tiff = TIFFimage(data, description='mock file')
                tiff.write_file(data_file, compression='none', verbose=True)
                del tiff
                #print(f' {data_file} written')
                file_name = filenames.readline().rstrip('\n')
                

def main():
    parser = ArgumentParser()
    parser.add_argument("config", help="Path to file list", type=str)
    #parser.add_argument("-UTIFF", "--use_tiff", default=0, help=" use TIFF backend ? 0 false anything else true ", type=int, required=False)
    
    args = parser.parse_args()
    print(f' cli args are {args}')
    global config
    config = args.config
    run(config)


if __name__ == "__main__":
    main()


