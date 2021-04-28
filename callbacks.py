import tensorflow as tf

# Tensorboard callback: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard
tensorboard_cb = tf.keras.callbacks.TensorBoard(
log_dir="models/logs", 
histogram_freq=1, #frequency (in epochs) at which to compute activation and weight histograms for the layers of the model. If set to 0, histograms won't be computed. 
write_graph=False, # whether to visualize the graph in TensorBoard. The log file can become quite large when write_graph is set to True.
write_images=False, # whether to write model weights to visualize as image in TensorBoard.
update_freq="epoch", # 'batch' or 'epoch' or integer. When using 'batch', writes the losses and metrics to TensorBoard after each batch. The same applies for 'epoch'. If using an integer, let's say 1000, the callback will write the metrics and losses to TensorBoard every 1000 batches. Note that writing too frequently to TensorBoard can slow down your training.
profile_batch=0, # Profile the batch(es) to sample compute characteristics. profile_batch must be a non-negative integer or a tuple of integers. A pair of positive integers signify a range of batches to profile. By default, it will profile the second batch. Set profile_batch=0 to disable profiling.
)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
filepath="models/",
monitor='val_loss',
save_best_only=True
)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    patience=3, restore_best_weights=True
)