import os
import tensorflow as tf

def setup_gpu():
    """
    Configures TensorFlow to use the GPU safely without crashing,
    and handles fallback to CPU for non-GPU machines.
    """
    # Suppress verbose TF logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=INFO, 2=WARNING, 3=ERROR
    
    # Optional: Suppress oneDNN warnings if CPU is used
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"✅ GPU Mode Active: {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs configured.")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(f"❌ Error setting up GPU: {e}")
    else:
        print("🖥️ CPU Mode Active: No compatible GPU detected. Proceeding with CPU.")
