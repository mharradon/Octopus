from collections import OrderedDict

import numpy as np
from box import Box

from keras.models import Model as _Model
from keras.layers import Input, Embedding, Dense
from keras.layers.merge import Concatenate
from keras.engine.topology import Layer
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Lambda

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
    result = Box(OrderedDict()) 
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
    return [[input_shape[0],1]]*self.loss_size

def Model(inputs,outputs,ys,loss_func,**kwargs):
  in_ser,in_deser = build_ser_deser(inputs)
  out_ser,out_deser = build_ser_deser(outputs)
  y_ser,y_deser = build_ser_deser(ys)

  # Build input layers
  ys = y_deser([Input(y) if isinstance(y,tuple) else y for y in y_ser(ys)])

  inputs_ser = in_ser(inputs)
  outputs_ser = out_ser(outputs)
  ys_ser = y_ser(ys)

  loss_skeleton = loss_func(None,None,dummy=True)
  loss_ser,loss_deser = build_ser_deser(loss_skeleton)

  def _loss_func(args):
    return loss_ser(loss_func(out_deser(args[:len(outputs_ser)]),y_deser(args[len(outputs_ser):])))
  loss_size = len(loss_skeleton.keys())

  losses_ser = LossLayer(_loss_func,loss_size=loss_size,name='loss')(outputs_ser + ys_ser)
  if not isinstance(losses_ser,list):
    losses_ser = [losses_ser]
  losses_ser = [Lambda(lambda x: x, name=k)(l) for l,k in zip(losses_ser,loss_ser(OrderedDict([(k,k) for k in loss_skeleton.keys()])))]
  
  train_model = _Model(inputs=inputs_ser + ys_ser,outputs=losses_ser,**kwargs)
  test_model = _Model(inputs=inputs_ser,outputs=outputs_ser,**kwargs)

  return Box({'train_model': train_model,
              'test_model': test_model,
              'inputs': inputs,
              'in_ser':in_ser,
              'out_deser':out_deser,
              'outputs': outputs,
              'outputs_ser': outputs_ser,
              'ys': ys,
              'loss_func': loss_func,
              'losses_ser': losses_ser,
              'loss_deser': loss_deser})

def Compile(my_model,optimizer,**kwargs):
  #losses = {l.name.split('/')[0]: (lambda y_true,y_pred: y_true) for l in my_model.losses_ser}
  losses = [lambda y_true,y_pred: y_pred for l in my_model.losses_ser]
  my_model.train_model.compile(optimizer,
                               loss=losses,
                               **kwargs)
  my_model.test_model.compile(optimizer,
                              loss='mean_squared_error', # Shouldn't matter
                              **kwargs)

def Predict(my_model,x):
  return my_model.out_deser(my_model.test_model.predict(my_model.in_ser(x)))

def fit_generator(my_model,generator,**kwargs):
  def oct_generator(g):
    _x,_y = next(g)
    x_ser,x_deser = build_ser_deser(_x)
    y_ser,y_deser = build_ser_deser(_y)
    while True:
      x = x_ser(_x) + y_ser(_y)
      y = [np.zeros((x[0].shape[0],)+(1,)*(len(l.shape)-1)) for l in my_model.losses_ser]
      _x,_y = next(g)
      yield (x,y)

  if 'validation_data' in kwargs:
    kwargs['validation_data'] = oct_generator(kwargs['validation_data'])

  return my_model.train_model.fit_generator(oct_generator(generator),**kwargs)

if __name__=="__main__":
  test_shape = (10,)

  inp0 = Input(shape=test_shape)
  inp1 = Input(shape=test_shape)
  x0 = Dense(16,activation='relu')(inp0)
  x1 = Dense(16,activation='relu')(inp1)
  z = Concatenate()([x0,x1]) 
  y_out = Dense(1)(z)

  def loss_func(outputs,ys,dummy=False):
    # Use callable functions to allow calling without arguments 
    # and inferring count of return parameters
    loss = {'loss0': lambda: (outputs['y']-ys['y_true'])**2,
            'loss1': lambda: K.zeros_like(outputs['y'])*outputs['y'],
            'loss3': lambda: 0.1*K.ones_like(outputs['y'])}
    
    if not dummy: 
      loss = OrderedDict([(k,v()) for k,v in loss.items()])
    
    return loss

  def my_gen():
    while True:
      x0 = np.random.randn(64,test_shape[0])
      x1 = np.random.randn(64,test_shape[0])
      y = x0[:,0]*2 + x1[:,1]*3 + 1
      yield {'inp0':x0,'inp1':x1},{'y_true':y,'y_fake':y+1}

  next(my_gen())
  model = Model(inputs={'inp0':inp0,'inp1':inp1},outputs={'inp0':inp0,'y':y_out},ys={'y_true':(1,),'y_fake':(1,)},loss_func=loss_func)
  Compile(model,'rmsprop') 
  fit_generator(model,my_gen(),verbose=2,steps_per_epoch=100,epochs=100)
