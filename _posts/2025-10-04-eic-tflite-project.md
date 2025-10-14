---
title: GaTech EIC TFLite project
description: 解题过程
author: cybotiger
date: 2025-10-04 12:00:00 +0800
categories: [编程语言, 深度学习框架]
tags: []
math: true
mermaid: true
---

onnx 的 nightly version 为 1.19.0；onnx-tf 的 nightly version 为 1.10.0

当 onnx<=1.18.0 时，会报以下错误

```bash
onnx.backend.test.runner.BackendIsNotSupposedToImplementIt: in user code:

    File "/home/ruihan/miniconda3/envs/deploy/lib/python3.10/site-packages/onnx_tf/backend_tf_module.py", line 99, in __call__  *
        output_ops = self.backend._onnx_node_to_tensorflow_op(onnx_node,
    File "/home/ruihan/miniconda3/envs/deploy/lib/python3.10/site-packages/onnx_tf/backend.py", line 347, in _onnx_node_to_tensorflow_op  *
        return handler.handle(node, tensor_dict=tensor_dict, strict=strict)
    File "/home/ruihan/miniconda3/envs/deploy/lib/python3.10/site-packages/onnx_tf/handlers/handler.py", line 61, in handle  *
        raise BackendIsNotSupposedToImplementIt("{} version {} is not implemented.".format(node.op_type, cls.SINCE_VERSION))

    BackendIsNotSupposedToImplementIt: ReduceMean version 18 is not implemented.
```

当 onnx==1.19.0 时，会报以下错误
```bash
File "/home/ruihan/miniconda3/envs/deploy/lib/python3.10/site-packages/onnx_tf/common/data_type.py", line 4, in <module>
    from onnx import mapping
ImportError: cannot import name 'mapping' from 'onnx' (/home/ruihan/miniconda3/envs/deploy/lib/python3.10/site-packages/onnx/__init__.py)
```

这样同时还需要修改其他 py 文件中有 mapping 的地方；一一修改完成
但最终还是同样的问题：`BackendIsNotSupposedToImplementIt: ReduceMean version 18 is not implemented.`

因为这个问题是出在 `tf_ref.export_graph` 语句，对应的是 onnx-tf 后端。于是，接下来的任务就是
+ 将 onnx-tf 升级，使其支持reducemean 18 算子；

或者

+ 修改 torch2onnx 的模型转化流程，使其不包含 reducemean 18 操作

目前的环境如下：
```
Python 3.10.18
---
Name: torch
Version: 2.8.0
---
Name: tensorflow
Version: 2.20.0
---
Name: onnx
Version: 1.19.0
---
Name: onnx-tf
Version: 1.10.0
---
Name: keras
Version: 3.11.3
---
Name: tf_keras
Version: 2.20.1
```
这样之后先遇到 `ImportError: cannot import name 'mapping' from 'onnx' (/home/ruihan/miniconda3/envs/deploy/lib/python3.10/site-packages/onnx/__init__.py)` 问题

这是由于 mapping 文件在新的 onnx 版本中 deprecated。解决方案：改用 helper 中的 api

这样之后会遇到 `ModuleNotFoundError: No module named 'tensorflow_probability'` 问题；解决方案：pip install tensorflow_probability，会下载 tensorflow_probability-0.25.0

这样之后会遇到 `ModuleNotFoundError: No module named 'tf_keras'` 这是因为 tfprob 依赖 tf_keras；解决方案：pip install tf_keras。会下载 tf_keras-2.20.1

这样之后会遇到 
```bash
File "/home/ruihan/miniconda3/envs/deploy/lib/python3.10/site-packages/tensorflow_addons/utils/types.py", line 29, in <module>
    from keras.src.engine import keras_tensor
ModuleNotFoundError: No module named 'keras.src.engine'
``` 
解决方案：将该语句改成 `from tensorflow.python.keras.engine import keras_tensor`

这样之后会遇到
```bash
  File "/home/ruihan/miniconda3/envs/deploy/lib/python3.10/site-packages/tensorflow_addons/optimizers/__init__.py", line 34, in <module>
    from tensorflow_addons.optimizers.lazy_adam import LazyAdam
  File "/home/ruihan/miniconda3/envs/deploy/lib/python3.10/site-packages/tensorflow_addons/optimizers/lazy_adam.py", line 38, in <module>
    class LazyAdam(adam_optimizer_class):
  File "/home/ruihan/miniconda3/envs/deploy/lib/python3.10/site-packages/keras/src/saving/object_registration.py", line 146, in decorator
    raise ValueError(
ValueError: Cannot register a class that does not have a get_config() method.
```
在 class LazyAdam 中添加    
```python
def get_config(self):
    return super().get_config()
```
这是在 github 原仓库 [https://github.com/tensorflow/addons/blob/master/tensorflow_addons/optimizers/lazy_adam.py#L153] 找到的


这样之后遇到
```bash
File "/home/ruihan/miniconda3/envs/deploy/lib/python3.10/site-packages/tensorflow_addons/rnn/nas_cell.py", line 30, in <module>
    class NASCell(keras.layers.AbstractRNNCell):
AttributeError: module 'keras._tf_keras.keras.layers' has no attribute 'AbstractRNNCell'
```

## torch2tf directly (Failed)
尝试直接从 torch 转 tflite，遇到报错 undefined symbol error，不想管了

```bash
ImportError: /home/ruihan/miniconda3/envs/deploy/lib/python3.10/site-packages/tensorflow/lite/python/metrics/_pywrap_tensorflow_lite_metrics_wrapper.so: 
undefined symbol: Wrapped_PyInit__pywrap_tensorflow_lite_metrics_wrapper
```

## Summary
最大的问题在于，onnx 到 tensorflow 的转换库 onnx-tf 已经停止维护了。所以后续 onnx 库的更新添加更多算子时，很多就不能与旧版本的 tensorflow converter 兼容


## Manually design tensorflow model
### fbnet-a arch
```python
FBNet(
  (backbone): FBNetBackbone(
    (stages): Sequential(
      (xif0_0): ConvBNRelu(
        (conv): Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (xif1_0): Identity()
      (xif2_0): IRFBlock(
        (pw): ConvBNRelu(
          (conv): Conv2d(16, 48, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (dw): ConvBNRelu(
          (conv): Conv2d(48, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=48)
          (bn): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (pwl): ConvBNRelu(
          (conv): Conv2d(48, 24, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (xif2_1): IRFBlock(
        (dw): ConvBNRelu(
          (conv): Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=24)
          (bn): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (pwl): ConvBNRelu(
          (conv): Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (res_conn): TorchAdd(
          (add_func): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
      )
      (xif2_2): Identity()
      (xif2_3): Identity()
      (xif3_0): IRFBlock(
        (pw): ConvBNRelu(
          (conv): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (dw): ConvBNRelu(
          (conv): Conv2d(144, 144, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=144)
          (bn): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (pwl): ConvBNRelu(
          (conv): Conv2d(144, 32, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (xif3_1): IRFBlock(
        (pw): ConvBNRelu(
          (conv): Conv2d(32, 96, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (dw): ConvBNRelu(
          (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96)
          (bn): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (pwl): ConvBNRelu(
          (conv): Conv2d(96, 32, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (res_conn): TorchAdd(
          (add_func): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
      )
      (xif3_2): IRFBlock(
        (dw): ConvBNRelu(
          (conv): Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=32)
          (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (pwl): ConvBNRelu(
          (conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (res_conn): TorchAdd(
          (add_func): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
      )
      (xif3_3): IRFBlock(
        (pw): ConvBNRelu(
          (conv): Conv2d(32, 96, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (dw): ConvBNRelu(
          (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96)
          (bn): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (pwl): ConvBNRelu(
          (conv): Conv2d(96, 32, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (res_conn): TorchAdd(
          (add_func): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
      )
      (xif4_0): IRFBlock(
        (pw): ConvBNRelu(
          (conv): Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (dw): ConvBNRelu(
          (conv): Conv2d(192, 192, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=192)
          (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (pwl): ConvBNRelu(
          (conv): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (xif4_1): IRFBlock(
        (pw): ConvBNRelu(
          (conv): Conv2d(64, 192, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (dw): ConvBNRelu(
          (conv): Conv2d(192, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=192)
          (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (pwl): ConvBNRelu(
          (conv): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (res_conn): TorchAdd(
          (add_func): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
      )
      (xif4_2): IRFBlock(
        (shuffle): ChannelShuffle()
        (dw): ConvBNRelu(
          (conv): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=64)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (pwl): ConvBNRelu(
          (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), groups=2)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (res_conn): TorchAdd(
          (add_func): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
      )
      (xif4_3): IRFBlock(
        (pw): ConvBNRelu(
          (conv): Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (dw): ConvBNRelu(
          (conv): Conv2d(384, 384, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=384)
          (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (pwl): ConvBNRelu(
          (conv): Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (res_conn): TorchAdd(
          (add_func): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
      )
      (xif4_4): IRFBlock(
        (pw): ConvBNRelu(
          (conv): Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (dw): ConvBNRelu(
          (conv): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384)
          (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (pwl): ConvBNRelu(
          (conv): Conv2d(384, 112, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (xif4_5): IRFBlock(
        (shuffle): ChannelShuffle()
        (dw): ConvBNRelu(
          (conv): Conv2d(112, 112, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=112)
          (bn): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (pwl): ConvBNRelu(
          (conv): Conv2d(112, 112, kernel_size=(1, 1), stride=(1, 1), groups=2)
          (bn): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (res_conn): TorchAdd(
          (add_func): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
      )
      (xif4_6): IRFBlock(
        (pw): ConvBNRelu(
          (conv): Conv2d(112, 336, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (dw): ConvBNRelu(
          (conv): Conv2d(336, 336, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=336)
          (bn): BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (pwl): ConvBNRelu(
          (conv): Conv2d(336, 112, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (res_conn): TorchAdd(
          (add_func): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
      )
      (xif4_7): IRFBlock(
        (shuffle): ChannelShuffle()
        (dw): ConvBNRelu(
          (conv): Conv2d(112, 112, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=112)
          (bn): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (pwl): ConvBNRelu(
          (conv): Conv2d(112, 112, kernel_size=(1, 1), stride=(1, 1), groups=2)
          (bn): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (res_conn): TorchAdd(
          (add_func): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
      )
      (xif5_0): IRFBlock(
        (pw): ConvBNRelu(
          (conv): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (dw): ConvBNRelu(
          (conv): Conv2d(672, 672, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=672)
          (bn): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (pwl): ConvBNRelu(
          (conv): Conv2d(672, 184, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(184, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (xif5_1): IRFBlock(
        (pw): ConvBNRelu(
          (conv): Conv2d(184, 1104, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(1104, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (dw): ConvBNRelu(
          (conv): Conv2d(1104, 1104, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1104)
          (bn): BatchNorm2d(1104, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (pwl): ConvBNRelu(
          (conv): Conv2d(1104, 184, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(184, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (res_conn): TorchAdd(
          (add_func): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
      )
      (xif5_2): IRFBlock(
        (pw): ConvBNRelu(
          (conv): Conv2d(184, 552, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(552, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (dw): ConvBNRelu(
          (conv): Conv2d(552, 552, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=552)
          (bn): BatchNorm2d(552, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (pwl): ConvBNRelu(
          (conv): Conv2d(552, 184, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(184, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (res_conn): TorchAdd(
          (add_func): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
      )
      (xif5_3): IRFBlock(
        (pw): ConvBNRelu(
          (conv): Conv2d(184, 1104, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(1104, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (dw): ConvBNRelu(
          (conv): Conv2d(1104, 1104, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1104)
          (bn): BatchNorm2d(1104, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (pwl): ConvBNRelu(
          (conv): Conv2d(1104, 184, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(184, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (res_conn): TorchAdd(
          (add_func): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
      )
      (xif5_4): IRFBlock(
        (pw): ConvBNRelu(
          (conv): Conv2d(184, 1104, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(1104, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (dw): ConvBNRelu(
          (conv): Conv2d(1104, 1104, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1104)
          (bn): BatchNorm2d(1104, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (pwl): ConvBNRelu(
          (conv): Conv2d(1104, 352, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(352, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (xif6_0): ConvBNRelu(
        (conv): Conv2d(352, 1504, kernel_size=(1, 1), stride=(1, 1))
        (bn): BatchNorm2d(1504, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
  )
  (head): ClsConvHead(
    (avg_pool): AdaptiveAvgPool2d(output_size=(1, 1))
    (conv): Conv2d(1504, 1000, kernel_size=(1, 1), stride=(1, 1))
  )
)

```

### fbnet-b arch
```python
FBNet(
  (backbone): FBNetBackbone(
    (stages): Sequential(
      (xif0_0): ConvBNRelu(
        (conv): Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (xif1_0): IRFBlock(
        (dw): ConvBNRelu(
          (conv): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16)
          (bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (pwl): ConvBNRelu(
          (conv): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (res_conn): TorchAdd(
          (add_func): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
      )
      (xif2_0): IRFBlock(
        (pw): ConvBNRelu(
          (conv): Conv2d(16, 96, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (dw): ConvBNRelu(
          (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96)
          (bn): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (pwl): ConvBNRelu(
          (conv): Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (xif2_1): IRFBlock(
        (dw): ConvBNRelu(
          (conv): Conv2d(24, 24, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=24)
          (bn): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (pwl): ConvBNRelu(
          (conv): Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (res_conn): TorchAdd(
          (add_func): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
      )
      (xif2_2): IRFBlock(
        (dw): ConvBNRelu(
          (conv): Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=24)
          (bn): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (pwl): ConvBNRelu(
          (conv): Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (res_conn): TorchAdd(
          (add_func): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
      )
      (xif2_3): IRFBlock(
        (dw): ConvBNRelu(
          (conv): Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=24)
          (bn): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (pwl): ConvBNRelu(
          (conv): Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (res_conn): TorchAdd(
          (add_func): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
      )
      (xif3_0): IRFBlock(
        (pw): ConvBNRelu(
          (conv): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (dw): ConvBNRelu(
          (conv): Conv2d(144, 144, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=144)
          (bn): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (pwl): ConvBNRelu(
          (conv): Conv2d(144, 32, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (xif3_1): IRFBlock(
        (pw): ConvBNRelu(
          (conv): Conv2d(32, 96, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (dw): ConvBNRelu(
          (conv): Conv2d(96, 96, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=96)
          (bn): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (pwl): ConvBNRelu(
          (conv): Conv2d(96, 32, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (res_conn): TorchAdd(
          (add_func): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
      )
      (xif3_2): IRFBlock(
        (pw): ConvBNRelu(
          (conv): Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (dw): ConvBNRelu(
          (conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192)
          (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (pwl): ConvBNRelu(
          (conv): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (res_conn): TorchAdd(
          (add_func): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
      )
      (xif3_3): IRFBlock(
        (pw): ConvBNRelu(
          (conv): Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (dw): ConvBNRelu(
          (conv): Conv2d(192, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=192)
          (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (pwl): ConvBNRelu(
          (conv): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (res_conn): TorchAdd(
          (add_func): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
      )
      (xif4_0): IRFBlock(
        (pw): ConvBNRelu(
          (conv): Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (dw): ConvBNRelu(
          (conv): Conv2d(192, 192, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=192)
          (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (pwl): ConvBNRelu(
          (conv): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (xif4_1): IRFBlock(
        (pw): ConvBNRelu(
          (conv): Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (dw): ConvBNRelu(
          (conv): Conv2d(384, 384, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=384)
          (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (pwl): ConvBNRelu(
          (conv): Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (res_conn): TorchAdd(
          (add_func): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
      )
      (xif4_2): Identity()
      (xif4_3): IRFBlock(
        (pw): ConvBNRelu(
          (conv): Conv2d(64, 192, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (dw): ConvBNRelu(
          (conv): Conv2d(192, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=192)
          (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (pwl): ConvBNRelu(
          (conv): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (res_conn): TorchAdd(
          (add_func): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
      )
      (xif4_4): IRFBlock(
        (pw): ConvBNRelu(
          (conv): Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (dw): ConvBNRelu(
          (conv): Conv2d(384, 384, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=384)
          (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (pwl): ConvBNRelu(
          (conv): Conv2d(384, 112, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (xif4_5): IRFBlock(
        (dw): ConvBNRelu(
          (conv): Conv2d(112, 112, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=112)
          (bn): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (pwl): ConvBNRelu(
          (conv): Conv2d(112, 112, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (res_conn): TorchAdd(
          (add_func): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
      )
      (xif4_6): IRFBlock(
        (dw): ConvBNRelu(
          (conv): Conv2d(112, 112, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=112)
          (bn): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (pwl): ConvBNRelu(
          (conv): Conv2d(112, 112, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (res_conn): TorchAdd(
          (add_func): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
      )
      (xif4_7): IRFBlock(
        (pw): ConvBNRelu(
          (conv): Conv2d(112, 336, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (dw): ConvBNRelu(
          (conv): Conv2d(336, 336, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=336)
          (bn): BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (pwl): ConvBNRelu(
          (conv): Conv2d(336, 112, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (res_conn): TorchAdd(
          (add_func): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
      )
      (xif5_0): IRFBlock(
        (pw): ConvBNRelu(
          (conv): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (dw): ConvBNRelu(
          (conv): Conv2d(672, 672, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=672)
          (bn): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (pwl): ConvBNRelu(
          (conv): Conv2d(672, 184, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(184, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (xif5_1): IRFBlock(
        (dw): ConvBNRelu(
          (conv): Conv2d(184, 184, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=184)
          (bn): BatchNorm2d(184, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (pwl): ConvBNRelu(
          (conv): Conv2d(184, 184, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(184, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (res_conn): TorchAdd(
          (add_func): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
      )
      (xif5_2): IRFBlock(
        (pw): ConvBNRelu(
          (conv): Conv2d(184, 1104, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(1104, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (dw): ConvBNRelu(
          (conv): Conv2d(1104, 1104, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1104)
          (bn): BatchNorm2d(1104, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (pwl): ConvBNRelu(
          (conv): Conv2d(1104, 184, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(184, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (res_conn): TorchAdd(
          (add_func): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
      )
      (xif5_3): IRFBlock(
        (pw): ConvBNRelu(
          (conv): Conv2d(184, 1104, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(1104, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (dw): ConvBNRelu(
          (conv): Conv2d(1104, 1104, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1104)
          (bn): BatchNorm2d(1104, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (pwl): ConvBNRelu(
          (conv): Conv2d(1104, 184, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(184, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (res_conn): TorchAdd(
          (add_func): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
      )
      (xif5_4): IRFBlock(
        (pw): ConvBNRelu(
          (conv): Conv2d(184, 1104, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(1104, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (dw): ConvBNRelu(
          (conv): Conv2d(1104, 1104, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1104)
          (bn): BatchNorm2d(1104, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (pwl): ConvBNRelu(
          (conv): Conv2d(1104, 352, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(352, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (xif6_0): ConvBNRelu(
        (conv): Conv2d(352, 1984, kernel_size=(1, 1), stride=(1, 1))
        (bn): BatchNorm2d(1984, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
  )
  (head): ClsConvHead(
    (avg_pool): AdaptiveAvgPool2d(output_size=(1, 1))
    (conv): Conv2d(1984, 1000, kernel_size=(1, 1), stride=(1, 1))
  )
)
```

## Depth-wise Separable Conv net
步骤 A: Depthwise Convolution (深度卷积)

    操作： 针对输入的每个通道，独立地使用一个单通道的滤波器 (K×K×1) 进行空间卷积。

    作用： 只负责提取空间特征，如边缘、纹理等，但不会在通道之间混合信息。

    结果： 保持通道数不变。如果输入有 Cin​ 个通道，输出也有 Cin​ 个通道。

    参数量： 极少，仅为 K×K×Cin​。

步骤 B: Pointwise Convolution (逐点卷积)

    操作： 使用 1×1 的卷积核。这个 1×1×Cin​ 的卷积核会在空间上滑动，对深度卷积的输出进行跨通道加权求和。

    作用： 只负责通道融合和通道数调整。它将深度卷积中分离的特征图进行线性组合，生成新的 Cout​ 个输出通道。

    参数量：1×1×Cin​×Cout​。 

