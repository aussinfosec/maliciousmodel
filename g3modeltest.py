import os
import tensorflow as tf
import torch
import torch.nn
import onnx
import pickle
import joblib
import zipfile

# Define classes from g3modelgen.py for allowlisting
class MaliciousModel(torch.nn.Module):
    def __init__(self):
        super(MaliciousModel, self).__init__()
        self.linear = torch.nn.Linear(10, 1)
    def __setstate__(self, state):
        super(MaliciousModel, self).__setstate__(state)
        import os
        os.system('echo Malicious code executed! (PyTorch PT) > /tmp/malicious.txt')

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
    def forward(self, x):
        return self.linear(x)

# Allowlist PyTorch classes and Linear
torch.serialization.add_safe_globals([MaliciousModel, SimpleModel, torch.nn.Linear])

# Function to safely check if a file/directory exists
def check_exists(path):
    return os.path.exists(path)

# Test Case 1
def verify_test_case_1():
    path = "./malicious_model_tf"
    if check_exists(path):
        try:
            model = tf.saved_model.load(path)
            print("Test Case 1: Loaded successfully -", model.signatures.keys())
        except Exception as e:
            print(f"Test Case 1: Failed to load - {e}")
    else:
        print("Test Case 1: File not found")

# Test Case 2 (handles both .h5 and .keras)
def verify_test_case_2():
    path = "./malicious_model_keras.keras" if check_exists("./malicious_model_keras.keras") else "./malicious_model_keras.h5"
    if check_exists(path):
        try:
            model = tf.keras.models.load_model(path, custom_objects={'malicious_function': lambda x: x})
            print("Test Case 2: Loaded successfully -", type(model).__name__)
        except Exception as e:
            print(f"Test Case 2: Failed to load - {e}")
    else:
        print("Test Case 2: File not found")

# Test Case 3
def verify_test_case_3():
    path = "./malicious_saved_model_trojanized"
    if check_exists(path):
        try:
            model = tf.saved_model.load(path)
            print("Test Case 3: Loaded successfully -", model.signatures.keys())
            asset_path = os.path.join(path, "assets", "malicious.py")
            if os.path.exists(asset_path):
                print("Test Case 3: Asset 'malicious.py' found")
        except Exception as e:
            print(f"Test Case 3: Failed to load - {e}")
    else:
        print("Test Case 3: File not found")

# Test Case 4
def verify_test_case_4():
    path = "malicious_model_pytorch.pt"
    if check_exists(path):
        try:
            model = torch.load(path)
            print("Test Case 4: Loaded successfully -", type(model).__name__)
        except Exception as e:
            print(f"Test Case 4: Failed to load - {e}")
    else:
        print("Test Case 4: File not found")

# Test Case 5
def verify_test_case_5():
    path = "malicious_pkl_file.pkl"
    if check_exists(path):
        try:
            with open(path, 'rb') as f:
                obj = pickle.load(f)
            print("Test Case 5: Loaded successfully -", type(obj).__name__)
        except Exception as e:
            print(f"Test Case 5: Failed to load - {e}")
    else:
        print("Test Case 5: File not found")

# Test Case 6
def verify_test_case_6():
    path = "malicious_onnx_model_metadata.onnx"
    if check_exists(path):
        try:
            model = onnx.load(path)
            onnx.checker.check_model(model)
            metadata = {prop.key: prop.value for prop in model.metadata_props}
            print("Test Case 6: Loaded successfully - Metadata:", metadata)
        except Exception as e:
            print(f"Test Case 6: Failed to load - {e}")
    else:
        print("Test Case 6: File not found")

# Test Case 7
def verify_test_case_7():
    path = "malicious_onnx_model_node_attributes.onnx"
    if check_exists(path):
        try:
            model = onnx.load(path)
            onnx.checker.check_model(model)
            attrs = model.graph.node[0].attribute[0].s.decode()
            print("Test Case 7: Loaded successfully - Node Attribute:", attrs)
        except Exception as e:
            print(f"Test Case 7: Failed to load - {e}")
    else:
        print("Test Case 7: File not found")

# Test Case 8
def verify_test_case_8():
    path = "malicious_model_sklearn.joblib"
    if check_exists(path):
        try:
            model = joblib.load(path)
            print("Test Case 8: Loaded successfully -", type(model).__name__)
        except Exception as e:
            print(f"Test Case 8: Failed to load - {e}")
    else:
        print("Test Case 8: File not found")

# Test Case 9
def verify_test_case_9():
    path = "forged_malicious_onnx_model.onnx"
    if check_exists(path):
        try:
            model = onnx.load(path)
            onnx.checker.check_model(model)
            print("Test Case 9: Loaded successfully - Producer:", model.producer_name)
        except Exception as e:
            print(f"Test Case 9: Failed to load - {e}")
    else:
        print("Test Case 9: File not found")

# Test Case 10
def verify_test_case_10():
    zip_path = "encrypted_model.zip"
    extract_path = "model.pt"
    if check_exists(zip_path):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.setpassword(b"secret")
                zip_ref.extract(extract_path)
            model = torch.load(extract_path)
            print("Test Case 10: Loaded successfully -", type(model).__name__)
            os.remove(extract_path)
        except Exception as e:
            print(f"Test Case 10: Failed to load - {e}")
    else:
        print("Test Case 10: File not found")

# Main verification function
def verify_all_models():
    print("Verifying all generated models...")
    verify_test_case_1()
    verify_test_case_2()
    verify_test_case_3()
    verify_test_case_4()
    verify_test_case_5()
    verify_test_case_6()
    verify_test_case_7()
    verify_test_case_8()
    verify_test_case_9()
    verify_test_case_10()
    print("Verification complete.")

if __name__ == "__main__":
    verify_all_models()