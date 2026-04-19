import tensorflow as tf
from .model import build_mutation_classifier, focal_loss
from .config import BATCH_SIZE, EPOCHS, VALIDATION_SPLIT

def train_pipeline(X_train, y_train, feature_engineer):
    """
    Handles model compilation, callback setup, and the training loop.
    """
    # Build model using metadata from feature engineer
    model = build_mutation_classifier(
        gene_size=len(feature_engineer.gene_le.classes_),
        type_size=len(feature_engineer.type_le.classes_),
        chrom_size=len(feature_engineer.chrom_le.classes_)
    )
    
    # Using Recall and Precision to monitor medical utility
    model.compile(
        optimizer='adam',
        loss=focal_loss(gamma=2.0, alpha=0.5), # Optimized alpha for better precision balance
        metrics=[
            'accuracy', 
            tf.keras.metrics.Recall(name='recall'), 
            tf.keras.metrics.Precision(name='precision')
        ]
    )
    
    # Callbacks for robust training
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_recall', 
            patience=10, 
            mode='max', 
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
        [X_train['gene'], X_train['type'], X_train['chrom'], X_train['numeric']],
        y_train,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history
