import tensorflow as tf
from tensorflow.keras import layers, models

def focal_loss(gamma=2., alpha=.75):
    """
    Implementation of Focal Loss for imbalanced classification.
    Focuses the model on hard-to-classify examples by down-weighting easy ones.
    Alpha is set high (0.75) to strongly weight the minority oncogenic class.
    """
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        # Clip to prevent log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Calculate focal loss
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        fl = - alpha_t * tf.pow((1 - p_t), gamma) * tf.math.log(p_t)
        return tf.reduce_mean(fl)
    return focal_loss_fixed

def build_mutation_classifier(gene_size, type_size, chrom_size, origin_size):
    """
    Constructs a multi-input Neural Network for cancer mutation classification.
    Uses Embedding layers for categorical features and a simplified dense network
    proportional to the actual feature space (~15 effective features after embeddings).
    """
    # Inputs
    gene_in = layers.Input(shape=(1,), name='gene')
    type_in = layers.Input(shape=(1,), name='type')
    chrom_in = layers.Input(shape=(1,), name='chrom')
    origin_in = layers.Input(shape=(1,), name='origin')
    num_in = layers.Input(shape=(8,), name='numeric')  # 8 numeric features
    
    # Embeddings — sized proportionally to cardinality
    gene_emb = layers.Flatten()(layers.Embedding(gene_size, 32)(gene_in))
    type_emb = layers.Flatten()(layers.Embedding(type_size, 16)(type_in))
    chrom_emb = layers.Flatten()(layers.Embedding(chrom_size, 16)(chrom_in))
    origin_emb = layers.Flatten()(layers.Embedding(origin_size, 8)(origin_in))
    
    # Merge all feature streams
    merged = layers.Concatenate()([gene_emb, type_emb, chrom_emb, origin_emb, num_in])
    
    # Simplified dense network (proportional to ~80 effective input features)
    x = layers.Dense(256, activation='relu')(merged)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    
    output = layers.Dense(1, activation='sigmoid', name='prediction')(x)
    
    model = models.Model(
        inputs=[gene_in, type_in, chrom_in, origin_in, num_in], 
        outputs=output
    )
    
    return model
