from keras.models import Model
from box import Box
from keras.layers import Input, Embedding, Dense
from keras.layers.merge import Concatenate
from utils import build_ser_deser,LossLayer

def my_model(inputs,outputs,ys,loss_func,**kwargs):
  in_ser,in_deser = build_ser_deser(inputs)
  out_ser,out_deser = build_ser_deser(outputs)
  y_ser,y_deser = build_ser_deser(ys)

  # Build input layers
  ys = y_deser([Input(y) for y in y_ser(ys)])

  inputs_ser = in_ser(inputs)
  outputs_ser = out_ser(outputs)
  ys_ser = y_ser(ys)

  loss_skeleton = loss_func(None,None,dummy=True)
  loss_ser,loss_deser = build_ser_deser(loss_skeleton)

  def _loss_func(args):
    return loss_ser(loss_func(out_deser(args[:len(outputs_ser)]),y_deser(args[len(outputs_ser):])))
  loss_size = len(loss_skeleton.keys())

  losses_ser = LossLayer(_loss_func,loss_size=loss_size)(outputs_ser + ys_ser)
  if not isinstance(losses_ser,list):
    losses_ser = [losses_ser]

  train_model = Model(inputs=inputs_ser + ys_ser,outputs=outputs_ser + losses_ser,**kwargs)
  test_model = Model(inputs=inputs_ser,outputs=outputs_ser,**kwargs)

  return Box({'train_model': train_model,
              'test_model': test_model,
              'inputs': inputs,
              'outputs': outputs,
              'outputs_ser': outputs_ser,
              'ys': ys,
              'loss_func': loss_func,
              'losses_ser': losses_ser,
              'loss_deser': loss_deser})

def my_compile(my_model,optimizer,**kwargs):
  #losses = {l.name: (lambda y_true,y_pred: y_true) for l in my_model.losses_ser}
  losses = [lambda y_true,y_pred: K.zeros_like(y_pred)*y_pred for l in my_model.outputs_ser] + \
           [(lambda y_true,y_pred: y_pred) for l in my_model.losses_ser]
  my_model.train_model.compile(optimizer,
                               loss=losses,
                               **kwargs)
  my_model.test_model.compile(optimizer,
                              loss='mean_squared_error', # Shouldn't matter
                              **kwargs)

def my_fit_generator(my_model,generator,**kwargs):
  def my_generator():
    _x,_y = next(generator)
    x_ser,x_deser = build_ser_deser(_x)
    y_ser,y_deser = build_ser_deser(_y)
    while True:
      _x,_y = next(generator)
      x = x_ser(_x) + y_ser(_y)
      y = [np.zeros(x[0].shape[0]) for _ in x]
      yield (x,y)

  next(my_generator())
  return my_model.train_model.fit_generator(my_generator(),**kwargs)

if __name__=="__main__":
  import keras.backend as K
  import numpy as np

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
            'loss1': lambda: K.zeros_like(outputs['y'])*outputs['y']}

    if not dummy: 
      loss = {k:v() for k,v in loss.items()}
    
    return loss

  def my_gen():
    while True:
      x0 = np.random.randn(64,test_shape[0])
      x1 = np.random.randn(64,test_shape[0])
      y = x0[:,0]*2 + x1[:,1]*3 + 1
      yield {'inp0':x0,'inp1':x1},{'y_true':y}

  next(my_gen())
  model = my_model(inputs={'inp0':inp0,'inp1':inp1},outputs={'y':y_out},ys={'y_true':(1,)},loss_func=loss_func)
  my_compile(model,'rmsprop') 
  my_fit_generator(model,my_gen(),verbose=2,steps_per_epoch=100,epochs=100)
