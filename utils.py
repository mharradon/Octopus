from keras.engine.topology import Layer
from collections import OrderedDict

def build_ser_deser(example):
  # Return funcs that given dict of dict x return a list in a fixed order and a func that takes flat list and returns dict

  x_map = OrderedDict(sorted(list(build_dict_of_dict_map(example).items()),key=lambda x: x[0]))

  def ser(x):
    result = []
    for k in x_map.keys():
      ks = list(k)
      sub = x
      while len(ks)>0:
        sub = sub[ks.pop(0)]
      result.append(sub)
    return result

  def deser(x):
    result = example.__class__()
    for k in x_map.keys():
      ks = list(k)
      sub = result
      while len(ks)>1:
        next_k = ks.pop(0)
        if next_k not in sub:
          sub[next_k] = example.__class__()
        sub = sub[next_k]
      sub[ks.pop(0)] = x.pop(0)
    return result

  return ser,deser

def build_dict_of_dict_map(x):
  if hasattr(x,'items'):
    x_map = {}
    for k,v in x.items():
      x_map.update({(k,) + sub_k: sub_v for sub_k,sub_v in build_dict_of_dict_map(v).items()})
  else: 
    return {tuple():x}
  return x_map

class LossLayer(Layer):
  def __init__(self, loss_func, loss_size, **kwargs):
    self.loss_func = loss_func
    self.loss_size = loss_size
    super(LossLayer,self).__init__(**kwargs)

  def call(self,x):
    return self.loss_func(x)

  def compute_output_shape(self, input_shape):
    return [(input_shape[0],1)]*self.loss_size
