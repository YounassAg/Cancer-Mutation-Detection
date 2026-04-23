import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from .model import build_mutation_classifier, focal_loss
from .config import BATCH_SIZE, EPOCHS, VALIDATION_SPLIT

def train_pipeline(X_train, y_train, feature_engineer):
    """
    Handles model compilation, callback setup, and the training loop.
    Includes class weight computation to address severe class imbalance.
    """
    # Build model using metadata from feature engineer
    model = build_mutation_classifier(
        gene_size=len(feature_engineer.gene_le.classes_),
        type_size=len(feature_engineer.type_le.classes_),
        chrom_size=len(feature_engineer.chrom_le.classes_),
        origin_size=len(feature_engineer.origin_le.classes_)
    )
    
    # Compute class weights to counteract severe imbalance
    # This complements Focal Loss — both are needed for extreme imbalance
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes.astype(int), weights))
    print(f"Class weights: {class_weights}")
    
    # Using Recall and Precision to monitor medical utility
    model.compile(
        optimizer='adam',
        loss=focal_loss(gamma=2.0, alpha=0.75),
        metrics=[
            'accuracy', 
            tf.keras.metrics.Recall(name='recall'), 
            tf.keras.metrics.Precision(name='precision')
        ]
    )
    
    # Callbacks for robust training
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            mode='min', 
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=5, 
            min_lr=1e-5
        )
    ]
    
    history = model.fit(
        [X_train['gene'], X_train['type'], X_train['chrom'], X_train['origin'], X_train['numeric']],
        y_train,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history
