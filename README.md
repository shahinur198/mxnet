# mxnet

https://mxnet.incubator.apache.org/versions/master/tutorials/basic/symbol.html
# install mxnet
```
pip install mxnet
pip install mxnet-mkl
pip install jupyter
```
# mxnet importent mathod
```
mx.sym.Convolution(**kwargs)
mx.sym.LeakyReLU(data = data, act_type='prelu', name = name)
mx.symbol.Activation(data=data, act_type=act_type, name=name)
mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
mx.sym.Pooling(data=bn3, global_pool=True, kernel=(7, 7), pool_type='avg', name=name+'_se_pool1')
mx.symbol.broadcast_mul(bn3, body)
mx.sym.identity(data=data, name='id')

mx.symbol.Variable("fc7_weight", shape=(config.num_classes, config.emb_size),
lr_mult=config.fc7_lr_mult, wd_mult=config.fc7_wd_mult, init=mx.init.Normal(0.01))
mx.symbol.Variable('fc7_bias', lr_mult=2.0, wd_mult=0.0)
mx.sym.FullyConnected(data=embedding, weight = _weight, no_bias = True, num_hidden=num_classes, name='fc7')

mx.symbol.L2Normalization(_weight, mode='instance')
mx.sym.one_hot(gt_label, depth = config.num_classes, on_value = s_m, off_value = 0.0)
mx.sym.pick(fc7, gt_label, axis=1)
mx.sym.arccos(cos_t)
mx.sym.cos(t)
mx.sym.expand_dims(diff, 1)
mx.symbol.slice_axis(nembedding, axis=0, begin=0, end=args.per_batch_size//3)
mx.symbol.sum(ap, axis=1, keepdims=1) #(T,1)
mx.symbol.mean(triplet_loss)
mx.symbol.BlockGrad(embedding)
mx.symbol.SoftmaxOutput(data=fc7, label = gt_label, name='softmax', normalization='valid')
mx.symbol.SoftmaxActivation(data=fc7)
mx.symbol.log(body)
mx.symbol.Group(out_list)
mx.gpu(i)
mx.cpu()

```
# Mxnet Tutorial
https://medium.com/@julsimon/an-introduction-to-the-mxnet-api-part-2-ce761513124e
