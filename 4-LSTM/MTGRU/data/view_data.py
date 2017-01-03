import tensorflow as tf
import struct
from tensorflow.core.example import example_pb2
import os
import glob
def get_names_in_dir(dirname):
    filelist = glob.glob(dirname)
    mainloc = "output-3000_name"
    for f in filelist:
        namefile = open(f, "rb")
        reader = namefile
        while True:
            len_bytes = reader.read(8)
            if not len_bytes: 
                break
            str_len = struct.unpack('q', len_bytes)[0]
            reader.read(4)
            example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
            reader.read(4)
            #print(example_str)
            example = example_pb2.Example.FromString(example_str)
            abstract = example.features.feature['abstract'].bytes_list.value[0]
            article = example.features.feature['article'].bytes_list.value[0]
            name = example.features.feature['name'].bytes_list.value[0]

            print(abstract)
            print(article)
            print(str(name))
    

get_names_in_dir("data/test/data-51")
'''
for serialized_example in tf.python_io.tf_record_iterator(filename):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)

    # traverse the Example format to get data
    image = example.features.feature['article'].int64_list.value
    label = example.features.feature['abstract'].int64_list.value[0]
    # do something
    print image, label
'''
