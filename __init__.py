from .problems import TranslateManyToMany

#from tensorflow.distribute.cluster_resolver import TPUClusterResolver
#TPUClusterResolver.__init__.__defaults__ = ('tpu-1', 'europe-west4-a', 'neural-stuff-215413', 'worker', None, None, 'default', None, None)
#print(TPUClusterResolver.__init__.__defaults__)

#import json
#import tensorflow as tf


#with open("data_config_6to6.json") as fp:
#    langs = json.load(fp)['languages']
#
#def encode(self, s, tgt_language="de"):
#    inputs = self.real_encode(s)
#    if tgt_language is not None:
#        inputs.insert(0, 2 + langs.index(tgt_language))
#    return inputs

#def modify_encoder():
#    from tensor2tensor.bin.t2t_decoder import create_hparams
#    hp = create_hparams()
#    print(hp.problem_hparams.vocabulary['inputs'])
#    raise

#modify_encoder()

#from tensor2tensor.data_generators import text_encoder

#text_encoder.SubwordTextEncoder.real_encode = text_encoder.SubwordTextEncoder.encode
#text_encoder.SubwordTextEncoder.encode = encode

