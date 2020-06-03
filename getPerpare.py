import wget 
import os
import zipfile
import tarfile

script_path = os.path.abspath(__file__)
print(script_path)
parent = os.path.dirname(script_path)
#print(parent)
elmo_folder = os.path.join(parent, 'elmo')
print(elmo_folder)

weightLink = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'
optionLink = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'
wget.download(weightLink, elmo_folder)
wget.download(optionLink, elmo_folder)
