# MaliciousModel

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Introduction

MaliciousModel is a security research project that demonstrates how machine learning model files can be weaponized with malicious content. This toolkit allows users to generate **malicious AI models** across various popular ML frameworks and formats, each containing embedded code or data intended to simulate malware behavior. The primary goal is to raise awareness about the potential risks of loading untrusted ML models, by providing controlled examples of **trojanized models** and testing their effects in a safe environment.

Key features of MaliciousModel include multiple pre-built attack scenarios (e.g., code execution on model load, dropping the EICAR test file, simulated C2 network communication, data exfiltration) and an automated testing script to validate each malicious model. This project is meant for **educational and research purposes** to help AI developers and security professionals understand and mitigate these risks.

## Installation Guide

Follow these steps to set up the MaliciousModel project on your local machine:

1. **Clone the repository**: Download the MaliciousModel repository from GitHub.  
    ```bash
    git clone https://github.com/aussinfosec/maliciousmodel.git
    cd maliciousmodel
    ```

2. **Set up a Python environment**: It’s recommended to use Python 3.8 or higher. Create a virtual environment (optional but advised) and activate it:  
    ```bash
    python3 -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies**: MaliciousModel relies on several machine learning and utility libraries. Install the required packages using pip:  
    ```bash
    pip install tensorflow torch onnx onnxruntime scikit-learn joblib
    ```
    *Note:* TensorFlow and PyTorch are large packages; ensure you have a working C/C++ build environment if needed and enough disk space. You may use `pip install --no-cache-dir` to avoid caching large wheels. Alternatively, use Anaconda/Miniconda to install these dependencies.

4. **Verify installation**: After installing, verify that all required libraries are available:  
    ```bash
    python -c "import tensorflow, torch, onnx, onnxruntime, sklearn; print('Dependencies loaded successfully')"
    ```

## Usage Instructions

MaliciousModel provides two main scripts: one for generating malicious model files and another for testing their behavior. **Always run these in an isolated environment** (see [Security Considerations & Disclaimer](#security-considerations--disclaimer)).

### Generating Malicious Models

Use the **`g3modelgen2.py`** script to generate malicious model files. When you run it, you’ll be prompted to select a test case or generate all cases at once:

```bash
python3 g3modelgen2.py

It displays a menu similar to:

Select a test case to generate (1-10) or 'all' to generate all:
1. Malicious Payload in .pb Graph (TensorFlow-Based Models) + EICAR
2. Obfuscated Code in .h5 (Keras) + C2 Connection
3. Trojanized SavedModel Directory (TensorFlow) + Data Exfiltration
4. Code Injection in .pt / .pth Checkpoints (PyTorch-Based Models) + EICAR
5. Malicious Script in .pkl (Pickle) + C2 Connection
6. Hidden Payload in ONNX Metadata (ONNX Models) + Data Exfiltration
7. Large Encoded Blobs in Node Attributes (ONNX Models) + EICAR
8. Malicious Code in scikit-learn .joblib or .pkl + C2 Connection
9. Signature Forging or Tampered Metadata (Other AI Model Formats) + Data Exfiltration
10. Encrypted / Password-Protected Model Files (Other AI Model Formats) + EICAR

Enter your choice:
```

- Enter a specific test case number (1–10) to generate that malicious model, or type `all` to generate all test cases.
- The script creates malicious model artifacts in the current directory (e.g., malicious_model_keras.h5, malicious_model_pytorch.pt, etc.) and prints a message for each generated file.

## Testing Malicious Models

After generating models, run the `g3modeltest.py` script to validate and demonstrate malicious behavior:

- This loads each malicious model, one by one, observing if the expected payload executes (e.g., dropping EICAR files, attempting network connections).
- It prints a log of success/failure for each test case.
- Make sure you have generated the models **before** testing.

**Important:** Do not open these malicious model files with arbitrary tools or libraries on your main system. They are designed to execute code on load.

## Features

- **Multi-Framework Support:** Generates malicious models for TensorFlow (.pb, SavedModel), Keras (.h5), PyTorch (.pt/.pth), scikit-learn (.pkl/.joblib), and ONNX (.onnx).
- Code execution upon model loading.
- Dropping files (including the EICAR test string).
- Simulated C2 network communication to a local or dummy address.
- Data exfiltration (posting example data to a dummy URL).
- Forged metadata or signatures to impersonate trusted model files.
- Password-protected archives containing malicious scripts.
- **Automated Validation:** A dedicated script (g3modeltest.py) verifies each test case in a controlled manner, logging which payloads executed successfully.
- **Safe for Demonstration:** Payloads are kept relatively harmless (writing benign files, using the EICAR test string, simulating outbound requests). Still, always use a sandbox or VM for testing.

## Security Considerations & Disclaimer

- **Use with Caution:** MaliciousModel generates intentional malware-like behavior in ML model files. Loading these models triggers code execution. Use them only in secure, sandboxed, or virtual machine environments.
- **Antivirus Alerts:** The EICAR test file is a known antivirus signature. Security software may flag or quarantine these files. Consider disabling AV or whitelisting your sandbox environment when running tests.
- **Network Connections:** Some payloads attempt to connect to dummy endpoints (example.com or 192.168.1.100). They won’t exfiltrate real data, but ensure you test offline or in a closed network if you want to block external calls entirely.
- **No Warranty:** This tool is provided “as is” for educational and research purposes. The authors are not liable for any misuse or damage caused by this software. By using MaliciousModel, you accept full responsibility.
- **Legal & Ethical Responsibility:** Do not use these models to compromise systems without explicit permission. Always follow local laws, regulations, and responsible disclosure guidelines.
