import tensorflow as tf
import struct
from tensorflow.core.example import example_pb2
import os
import glob
def get_names_in_dir(dirname):
    filelist = glob.glob(dirname)
    mainloc = "output-3000_name"
    for f in filelist:
        name = os.path.basename(f)
        infile = open(f, "rb")
        namefile = open(mainloc + "/" + name, "rb")
        reader = namefile
        reader2 = infile
        while True:
            len_bytes = reader.read(8)
            if not len_bytes: 
                break
            len_bytes2 = reader2.read(8)
            if not len_bytes2: 
                break
            str_len = struct.unpack('q', len_bytes)[0]
            reader.read(4)
            example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
            reader.read(4)

            str_len2 = struct.unpack('q', len_bytes2)[0]
            reader2.read(4)
            example_str2 = struct.unpack('%ds' % str_len2, reader2.read(str_len2))[0]
            reader2.read(4)
            #print(example_str)
            example = example_pb2.Example.FromString(example_str)
            abstract = example.features.feature['abstract'].bytes_list.value[0]
            article = example.features.feature['article'].bytes_list.value[0]
            name = example.features.feature['name'].bytes_list.value[0]

            example2 = example_pb2.Example.FromString(example_str2)
            abstract2 = example.features.feature['abstract'].bytes_list.value[0]
            article2 = example.features.feature['article'].bytes_list.value[0]
            #print(abstract)
            #print(article)
            if( abstract != abstract2):
                print("Unequal Abstracts")
            if( article != article2):
                print("unequal article")
            print(str(name))
    

get_names_in_dir("data/test/data-1*")
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
