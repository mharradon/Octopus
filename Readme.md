NB: This implementation is very janky / a work in progress.

Keras makes a lot of things easy, but its loss and optimization interfaces are fairly restrictive:

```
  model = Model(inputs=[i1,i2],outputs=[o1,o2])
  model.compile(self,optimizer,losses=[l1,l2])
```

Here the effective total loss is l1(i1,o1)+l2(i2,o2). This precludes any combination other than additive between the loss outputs. It also implies that each output has an associated y_true and loss, which is not generally the case.

A more flexible interface would be:

```
  model = Model(inputs={'i1':i1,'i2':i2},outputs={'o1':o1,'o2':o2},ys={'y1':y1,'y2':y2})
  model.compile(self,optimizer,loss_func)
```

Where loss func takes dictionaries outputs and ys of tf tensor objects.

```
  def loss_func(outputs,ys):
    return (outputs['o1']-ys['y1'])**2.sum()
```

And fit_generator yields x and y dicts(-of-dicts) of batch inputs.

TODO: Parameter regularization in loss function?



