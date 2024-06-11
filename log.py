import os
import tensorflow as tf

# Set log level to info to see GPU messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Check TensorFlow version
print("TensorFlow Version:", tf.__version__)

# Check if TensorFlow is built with CUDA support
print("Is TensorFlow built with CUDA:", tf.test.is_built_with_cuda())

# Check if GPU is available
print("Is GPU available:", tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))

# List physical devices
print("Physical Devices:", tf.config.experimental.list_physical_devices())

# Set memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Perform a simple operation to see if it uses the GPU
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    c = tf.matmul(a, b)
    print("Result of matrix multiplication:", c)
