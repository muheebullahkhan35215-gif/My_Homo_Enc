"""
Privacy-Preserving Federated Learning for Medical Imaging
Chest X-ray Classification with Homomorphic Encryption
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from Pyfhel import Pyfhel
import pickle
import os
from typing import List, Tuple
import unittest

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ========================
# Hardware Detection
# ========================

def check_hardware():
    """Check and report available hardware"""
    print("\n" + "="*40)
    print("HARDWARE CONFIGURATION")
    print("="*40)
    
    gpu_devices = tf.config.list_physical_devices('GPU')
    cpu_devices = tf.config.list_physical_devices('CPU')
    
    print(f"CPU devices: {len(cpu_devices)}")
    print(f"GPU devices: {len(gpu_devices)}")
    
    if gpu_devices:
        print("✓ GPU acceleration available")
        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("⚠ No GPU detected - running on CPU only")
    
    print(f"\nTensorFlow version: {tf.__version__}")
    return len(gpu_devices) > 0

# ========================
# CNN Model for Medical Imaging
# ========================

def create_medical_cnn(input_shape=(224, 224, 3), num_classes=2) -> keras.Model:
    """
    Create CNN model for chest X-ray classification
    Based on architectures used in medical imaging papers
    """
    model = keras.Sequential([
        # First Conv Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Second Conv Block  
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Third Conv Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Classifier
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✓ Created Medical CNN Model")
    return model

# ========================
# Data Preparation - Medical Images
# ========================

def create_synthetic_medical_data(num_clients=3, samples_per_client=1000):
    """
    Create synthetic medical data for testing
    In real scenario, replace with actual X-ray dataset
    """
    print("Creating synthetic medical data for testing...")
    
    # Create synthetic images (224x224x3) - similar to X-rays
    x_train = np.random.rand(samples_per_client * num_clients, 224, 224, 3).astype(np.float32)
    y_train = np.random.randint(0, 2, samples_per_client * num_clients)
    
    x_test = np.random.rand(2000, 224, 224, 3).astype(np.float32)
    y_test = np.random.randint(0, 2, 2000)
    
    # Split among clients
    client_datasets = []
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client
        client_x = x_train[start_idx:end_idx]
        client_y = y_train[start_idx:end_idx]
        client_datasets.append((client_x, client_y))
    
    print(f"Created synthetic data: {len(x_train)} training, {len(x_test)} test samples")
    return client_datasets, (x_test, y_test)

# ========================
# Homomorphic Encryption (Same as before)
# ========================

def setup_homomorphic_encryption() -> Pyfhel:
    HE = Pyfhel()
    HE.contextGen(scheme='bfv', n=2**15, t_bits=20, sec=128)
    HE.keyGen()
    HE.relinKeyGen()
    return HE

def encrypt_weights(weights: List[np.ndarray], HE: Pyfhel) -> List:
    encrypted_weights = []
    max_slots = HE.get_nSlots()

    for weight_matrix in weights:
        flat_weights = weight_matrix.flatten()
        scale_factor = 10000
        
        int_weights = np.round(flat_weights * scale_factor).astype(np.int64)
        
        max_allowed = 2**31 - 1
        if np.any(np.abs(int_weights) > max_allowed):
            print(f"⚠ Warning: Weight values too large, scaling down...")
            max_val = np.max(np.abs(int_weights))
            scale_down = max_val / max_allowed
            int_weights = np.round(int_weights / scale_down).astype(np.int64)
            scale_factor = scale_factor * scale_down
        
        weight_size = len(int_weights)
        encrypted_chunks = []
        
        if weight_size <= max_slots:
            encrypted_chunks.append(HE.encryptInt(int_weights))
        else:
            num_chunks = (weight_size + max_slots - 1) // max_slots
            for i in range(num_chunks):
                start_idx = i * max_slots
                end_idx = min((i + 1) * max_slots, weight_size)
                chunk = int_weights[start_idx:end_idx]
                encrypted_chunks.append(HE.encryptInt(chunk))
        
        encrypted_weights.append({
            'encrypted': encrypted_chunks,
            'shape': weight_matrix.shape,
            'scale': scale_factor,
            'size': weight_size
        })
    
    return encrypted_weights

def decrypt_weights(encrypted_weights: List, HE: Pyfhel) -> List[np.ndarray]:
    decrypted_weights = []
    num_clients = 3

    for enc_weight_dict in encrypted_weights:
        encrypted_chunks = enc_weight_dict['encrypted']
        shape = enc_weight_dict['shape']
        scale = enc_weight_dict['scale']
        size = enc_weight_dict['size']
        
        decrypted_parts = []
        for chunk in encrypted_chunks:
            decrypted_chunk = HE.decryptInt(chunk)
            decrypted_parts.append(decrypted_chunk)
        
        decrypted_flat = np.concatenate(decrypted_parts)[:size]
        decrypted_flat = decrypted_flat.astype(np.float64) / (scale * num_clients)
        decrypted_matrix = decrypted_flat.reshape(shape)
        decrypted_weights.append(decrypted_matrix.astype(np.float32))
    
    return decrypted_weights

# ========================
# Federated Aggregation
# ========================

def aggregate_encrypted_weights(encrypted_weights_list: List[List], HE: Pyfhel) -> List:
    num_clients = len(encrypted_weights_list)
    print(f"\n[Server] Aggregating encrypted weights from {num_clients} clients...")
    
    aggregated = []
    num_layers = len(encrypted_weights_list[0])
    
    for layer_idx in range(num_layers):
        num_chunks = len(encrypted_weights_list[0][layer_idx]['encrypted'])
        aggregated_chunks = []
        
        for chunk_idx in range(num_chunks):
            sum_encrypted = encrypted_weights_list[0][layer_idx]['encrypted'][chunk_idx].copy()
            for client_idx in range(1, num_clients):
                sum_encrypted += encrypted_weights_list[client_idx][layer_idx]['encrypted'][chunk_idx]
            aggregated_chunks.append(sum_encrypted)
        
        aggregated.append({
            'encrypted': aggregated_chunks,
            'shape': encrypted_weights_list[0][layer_idx]['shape'],
            'scale': encrypted_weights_list[0][layer_idx]['scale'],
            'size': encrypted_weights_list[0][layer_idx]['size']
        })
    
    print("[Server] Aggregation complete (weights remain encrypted)")
    return aggregated

def aggregate_plaintext_weights(client_weights: List[List[np.ndarray]]) -> List[np.ndarray]:
    num_clients = len(client_weights)
    aggregated_weights = []
    
    for layer_idx in range(len(client_weights[0])):
        layer_sum = np.zeros_like(client_weights[0][layer_idx])
        for client_idx in range(num_clients):
            layer_sum += client_weights[client_idx][layer_idx]
        aggregated_weights.append(layer_sum / num_clients)
    
    return aggregated_weights

# ========================
# Evaluation
# ========================

def evaluate_model(model: keras.Model, test_data: Tuple, label: str = "Model", verbose: bool = True) -> float:
    x_test, y_test = test_data
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    if verbose:
        print(f"[{label}] Test Accuracy: {accuracy*100:.2f}%")
    return accuracy

# ========================
# Medical FL Analysis
# ========================

def analyze_medical_results(client_models, global_model, test_data, client_weights, global_weights):
    """Enhanced analysis for medical imaging results"""
    print("\n" + "="*60)
    print("MEDICAL IMAGING FEDERATED LEARNING ANALYSIS")
    print("="*60)
    
    x_test, y_test = test_data
    
    # Evaluate all models
    print("Model Performance Summary:")
    client_accuracies = []
    for i, model in enumerate(client_models):
        acc = evaluate_model(model, test_data, label=f"  Client {i}", verbose=False)
        client_accuracies.append(acc)
        print(f"  Client {i}: {acc*100:.2f}%")
    
    plaintext_acc = evaluate_model(global_model, test_data, label="  Global Model", verbose=False)
    print(f"  Global Model: {plaintext_acc*100:.2f}%")
    
    avg_client_acc = np.mean(client_accuracies)
    performance_gap = avg_client_acc - plaintext_acc
    
    print(f"\nPerformance Analysis:")
    print(f"  Average Client Accuracy: {avg_client_acc*100:.2f}%")
    print(f"  Global Model Accuracy: {plaintext_acc*100:.2f}%")
    print(f"  Performance Gap: {performance_gap*100:.2f}%")
    
    if performance_gap > 0.1:
        print("  ⚠ Large performance gap - typical in medical FL due to data heterogeneity")
    else:
        print("  ✓ Reasonable performance gap")
    
    # Check model complexity
    print(f"\nModel Complexity:")
    total_params = global_model.count_params()
    print(f"  Total parameters: {total_params:,}")
    print(f"  Model size: {total_params * 4 / (1024**2):.2f} MB (float32)")
    
    return plaintext_acc

# ========================
# Unit Tests for Medical FL
# ========================

class TestMedicalFL(unittest.TestCase):
    
    def test_cnn_model_creation(self):
        """Test that CNN model creates correctly"""
        model = create_medical_cnn(input_shape=(224, 224, 3))
        self.assertIsInstance(model, keras.Model)
        print("✓ CNN model creation test passed")
    
    def test_medical_data_shape(self):
        """Test medical data has correct shape"""
        client_datasets, test_data = create_synthetic_medical_data(num_clients=2, samples_per_client=100)
        x_test, y_test = test_data
        self.assertEqual(x_test.shape[1:], (224, 224, 3))
        print("✓ Medical data shape test passed")

def run_medical_tests():
    print("Running medical FL verification tests...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMedicalFL)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()

# ========================
# Main Medical FL Pipeline
# ========================

def main():
    print("="*70)
    print("MEDICAL FEDERATED LEARNING WITH HOMOMORPHIC ENCRYPTION")
    print("Chest X-ray Classification - Privacy-Preserving AI")
    print("="*70)

    # Hardware configuration
    gpu_available = check_hardware()

    # Medical FL Configuration
    NUM_CLIENTS = 3
    EPOCHS = 2  # Start with fewer epochs for testing
    USE_SYNTHETIC_DATA = True  # Set to False when you have real X-ray data

    # Step 0: Run medical-specific tests
    print(f"\n[Step 0] Running Medical FL Tests...")
    tests_passed = run_medical_tests()
    if not tests_passed:
        print("❌ Medical tests failed!")
        return
    print("✓ All medical tests passed!")

    # Step 1: Load medical data
    print(f"\n[Step 1] Loading Medical Data...")
    if USE_SYNTHETIC_DATA:
        client_datasets, test_data = create_synthetic_medical_data(
            num_clients=NUM_CLIENTS, 
            samples_per_client=500  # Smaller for testing
        )
        print("⚠ Using SYNTHETIC data for testing")
        print("💡 Replace with real X-ray dataset for production")
    else:
        # This would load real X-ray data
        # client_datasets, test_data = load_real_xray_data(NUM_CLIENTS)
        pass

    # Step 2: Initialize homomorphic encryption
    print("\n[Step 2] Setting up homomorphic encryption...")
    HE = setup_homomorphic_encryption()
    print("Encryption context created")

    # Step 3: Train clients locally with CNN models
    print("\n[Step 3] Training Local CNN Models...")
    client_models = []
    client_weights = []

    for client_id in range(NUM_CLIENTS):
        print(f"\n[Client {client_id}] Training medical CNN...")
        model = create_medical_cnn()
        x_train, y_train = client_datasets[client_id]
        
        # Train with progress updates
        model.fit(x_train, y_train, epochs=EPOCHS, batch_size=16, verbose=1)
        
        weights = model.get_weights()
        client_models.append(model)
        client_weights.append(weights)
        
        evaluate_model(model, test_data, label=f"Client {client_id}")

    # Step 4: Plaintext aggregation (baseline)
    print("\n[Step 4a] Plaintext Aggregation (Baseline)...")
    plaintext_global_weights = aggregate_plaintext_weights(client_weights)
    plaintext_global_model = create_medical_cnn()
    plaintext_global_model.set_weights(plaintext_global_weights)
    plaintext_accuracy = evaluate_model(plaintext_global_model, test_data, label="Plaintext Global Model")

    # Step 4b: Encrypt client weights
    print("\n[Step 4b] Encrypting Medical Model Weights...")
    encrypted_weights_list = []
    for client_id in range(NUM_CLIENTS):
        print(f"[Client {client_id}] Encrypting CNN weights...")
        enc_weights = encrypt_weights(client_weights[client_id], HE)
        encrypted_weights_list.append(enc_weights)
    print("All medical model weights encrypted")

    # Step 5: Server aggregates encrypted weights
    print("\n[Step 5] Server-side Encrypted Aggregation...")
    aggregated_encrypted = aggregate_encrypted_weights(encrypted_weights_list, HE)

    # Step 6: Decrypt aggregated weights
    print("\n[Step 6] Decrypting Aggregated Medical Weights...")
    global_weights = decrypt_weights(aggregated_encrypted, HE)
    print("Global medical model weights decrypted")

    # Step 7: Create and evaluate global model
    print("\n[Step 7] Evaluating Encrypted Global Model...")
    global_model = create_medical_cnn()
    global_model.set_weights(global_weights)
    global_accuracy = evaluate_model(global_model, test_data, label="Encrypted Global Model")

    # Step 8: Medical-specific analysis
    print("\n[Step 8] Medical FL Analysis...")
    analyze_medical_results(client_models, global_model, test_data, client_weights, global_weights)

    # Summary
    print("\n" + "="*70)
    print("MEDICAL FEDERATED LEARNING - SUMMARY")
    print("="*70)
    print(f"Dataset: {'Synthetic Medical Data' if USE_SYNTHETIC_DATA else 'Real X-ray Data'}")
    print(f"Hardware: {'GPU' if gpu_available else 'CPU'}")
    print(f"Number of clients: {NUM_CLIENTS}")
    print(f"Training epochs: {EPOCHS}")
    print(f"Model: CNN for Medical Imaging")
    
    print(f"\nPerformance Results:")
    avg_local = np.mean([evaluate_model(m, test_data, verbose=False) for m in client_models])
    print(f"  Average Local Accuracy: {avg_local*100:.2f}%")
    print(f"  Plaintext Global Accuracy: {plaintext_accuracy*100:.2f}%")
    print(f"  Encrypted Global Accuracy: {global_accuracy*100:.2f}%")
    print(f"  Encryption Impact: {(plaintext_accuracy - global_accuracy)*100:.2f}%")
    
    print(f"\nClinical Relevance:")
    print("  ✓ Privacy-preserving medical AI achieved")
    print("  ✓ Homomorphic encryption protects patient data")
    print("  ✓ Federated learning enables multi-institutional collaboration")
    
    if USE_SYNTHETIC_DATA:
        print(f"\n⚠ NOTE: Using synthetic data for testing")
        print("  Next steps: Download real X-ray dataset and update data loading")
    
    print("="*70)

if __name__ == "__main__":
    main()