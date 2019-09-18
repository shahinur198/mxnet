# mxnet

https://mxnet.incubator.apache.org/versions/master/tutorials/basic/symbol.html
# install mxnet
```
pip install mxnet
pip install mxnet-mkl
```
# mxnet importent mathod
```
mx.sym.Convolution(**kwargs)
mx.sym.LeakyReLU(data = data, act_type='prelu', name = name)
mx.symbol.Activation(data=data, act_type=act_type, name=name)
mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
mx.sym.Pooling(data=bn3, global_pool=True, kernel=(7, 7), pool_type='avg', name=name+'_se_pool1')
mx.symbol.broadcast_mul(bn3, body)
```
