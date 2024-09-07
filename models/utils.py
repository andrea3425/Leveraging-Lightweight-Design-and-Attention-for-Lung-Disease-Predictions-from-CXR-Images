import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

def compute_flops(model):
    '''Compute the FLOPs of a Keras model.'''

    # Wrap the model in a tf.function to enable concrete function extraction
    model_func = tf.function(lambda x: model(x))

    # Create a concrete function from the model with a specific input shape
    concrete_func = model_func.get_concrete_function(tf.TensorSpec([1] + list(model.input[0].shape[1:]), model.input[0].dtype))

    # Convert the concrete function into a frozen graph (i.e., all variables converted to constants)
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)

    # Set up the RunMetadata object to store profiling data
    run_meta = tf.compat.v1.RunMetadata()

    # Set the options for profiling to count the floating point operations (FLOPs)
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

    # Profile the frozen graph to compute the total number of FLOPs
    flops = tf.compat.v1.profiler.profile(graph=frozen_func.graph, run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops

def plot(history, save_path=None, dpi=300, format='png', bbox_inches='tight'):
    '''Plot training history'''

    plt.figure(figsize=(12, 4))

    # Grafico della Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Grafico dell'Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Mostra o salva il grafico
    if save_path:
        plt.savefig(save_path, dpi=dpi, format=format, bbox_inches=bbox_inches)
        print(f"Figure saved as {save_path}")
    else:
        plt.show()

def save_history_as_json(history, file_path):
    '''Save training history as a .json file'''

    history_dict = history.history
    with open(file_path, 'w') as f:
        json.dump(history_dict, f)
    print(f"History saved as {file_path}")