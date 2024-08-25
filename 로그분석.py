/home/hong/anaconda3/envs/csigan/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:526: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint8 = np.dtype([("qint8", np.int8, 1)])
/home/hong/anaconda3/envs/csigan/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:527: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
/home/hong/anaconda3/envs/csigan/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:528: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint16 = np.dtype([("qint16", np.int16, 1)])
/home/hong/anaconda3/envs/csigan/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:529: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
/home/hong/anaconda3/envs/csigan/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:530: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint32 = np.dtype([("qint32", np.int32, 1)])
/home/hong/anaconda3/envs/csigan/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:535: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  np_resource = np.dtype([("resource", np.ubyte, 1)])
WARNING:tensorflow:From /home/hong/anaconda3/envs/csigan/lib/python3.6/site-packages/tensorflow/python/ops/resource_variable_ops.py:435: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
Finished preprocessing.
1/1
WARNING:tensorflow:From /home/hong/anaconda3/envs/csigan/lib/python3.6/site-packages/tensorflow/python/keras/layers/core.py:143: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
Instructions for updating:
Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
Traceback (most recent call last):
  File "/home/hong/anaconda3/envs/csigan/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 1659, in _create_c_op
    c_op = c_api.TF_FinishOperation(op_desc)
tensorflow.python.framework.errors_impl.InvalidArgumentError: Dimensions must be equal, but are 32768 and 31232 for 'model_1/dense/MatMul' (op: 'MatMul') with input shapes: [?,32768], [31232,4].

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/media/hong/89ac3152-9243-4884-8b0b-981398d86390/home/hong/CSI-SemiGAN-master/main.py", line 281, in <module>
    run_exp1()
  File "/media/hong/89ac3152-9243-4884-8b0b-981398d86390/home/hong/CSI-SemiGAN-master/main.py", line 114, in run_exp1
    gan_model = define_GAN(g_model, d_model, optimizer)
  File "/media/hong/89ac3152-9243-4884-8b0b-981398d86390/home/hong/CSI-SemiGAN-master/models.py", line 51, in define_GAN
    gan_output = d_model(g_model.output)
  File "/home/hong/anaconda3/envs/csigan/lib/python3.6/site-packages/tensorflow/python/keras/engine/base_layer.py", line 554, in __call__
    outputs = self.call(inputs, *args, **kwargs)
  File "/home/hong/anaconda3/envs/csigan/lib/python3.6/site-packages/tensorflow/python/keras/engine/network.py", line 815, in call
    mask=masks)
  File "/home/hong/anaconda3/envs/csigan/lib/python3.6/site-packages/tensorflow/python/keras/engine/network.py", line 1002, in _run_internal_graph
    output_tensors = layer.call(computed_tensor, **kwargs)
  File "/home/hong/anaconda3/envs/csigan/lib/python3.6/site-packages/tensorflow/python/keras/layers/core.py", line 975, in call
    outputs = gen_math_ops.mat_mul(inputs, self.kernel)
  File "/home/hong/anaconda3/envs/csigan/lib/python3.6/site-packages/tensorflow/python/ops/gen_math_ops.py", line 5333, in mat_mul
    name=name)
  File "/home/hong/anaconda3/envs/csigan/lib/python3.6/site-packages/tensorflow/python/framework/op_def_library.py", line 788, in _apply_op_helper
    op_def=op_def)
  File "/home/hong/anaconda3/envs/csigan/lib/python3.6/site-packages/tensorflow/python/util/deprecation.py", line 507, in new_func
    return func(*args, **kwargs)
  File "/home/hong/anaconda3/envs/csigan/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 3300, in create_op
    op_def=op_def)
  File "/home/hong/anaconda3/envs/csigan/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 1823, in __init__
    control_input_ops)
  File "/home/hong/anaconda3/envs/csigan/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 1662, in _create_c_op
    raise ValueError(str(e))
ValueError: Dimensions must be equal, but are 32768 and 31232 for 'model_1/dense/MatMul' (op: 'MatMul') with input shapes: [?,32768], [31232,4].