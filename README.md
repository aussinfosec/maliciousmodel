# AI Security Test Suite 🔒🤖

Advanced framework for evaluating AI model security scanners through controlled malicious model generation. Provides red team tools to test ML model vulnerability detection capabilities.

## Features 🛠️

### Model Format Support
| Format               | Extension      | Injection Method               |
|----------------------|----------------|---------------------------------|
| TensorFlow           | .pb            | GraphDef embedding             |
| Keras                | .h5            | HDF5 metadata injection         |
| PyTorch              | .pt/.pth       | Module hijacking                |
| ONNX                 | .onnx          | Custom node insertion           |
| scikit-learn         | .joblib        | Model serialization abuse       |
| TensorFlow SavedModel| /              | Protocol buffer manipulation    |
| Encrypted Archive    | .zip           | AES-256 encrypted payload       |

### Advanced Capabilities
- **Multi-stage Payload Delivery**
  - AES-CFB encryption with random IV
  - Per-chunk XOR obfuscation
  - zlib compression layer
- **Context-Aware Activation**
  - Trigger phrase: `TRIGGER_PHRASE` (random per session)
  - Runtime environment validation
- **Anti-Analysis Measures**
  - Dynamic junk code generation
  - Weight matrix steganography
  - Layer structure spoofing

## Installation ⚙️

### Dependencies

pip install \
  torch>=2.0.0 \
  onnx>=1.13.0 \
  h5py>=3.8.0 \
  scikit-learn>=1.2.0 \
  pycryptodome>=3.17 \
  pyzipper>=0.3.6 \
  transformers>=4.28.0 \
  torchvision>=0.14.0
