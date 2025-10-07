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