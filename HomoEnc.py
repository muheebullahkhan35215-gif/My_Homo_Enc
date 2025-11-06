"""
Privacy-Preserving Federated Learning with Homomorphic Encryption
Demonstrates federated learning with encrypted model weights using Pyfhel
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from Pyfhel import Pyfhel
import pickle
from typing import List, Tuple

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ========================
# Model Creation
# ========================

def create_model() -> keras.Model:
    """
    Create a simple MLP for MNIST digit classification
    """
    model = keras.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ========================
# Data Preparation
# ========================

def load_and_split_data(num_clients: int = 3) -> Tuple:
    """
    Load MNIST data and split among clients
    Returns: (client_datasets, test_data)
    """
    # Load MNIST
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Split training data among clients
    samples_per_client = len(x_train) // num_clients
    client_datasets = []
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client if i < num_clients - 1 else len(x_train)
        
        client_x = x_train[start_idx:end_idx]
        client_y = y_train[start_idx:end_idx]
        client_datasets.append((client_x, client_y))
    
    return client_datasets, (x_test, y_test)


# ========================
# Client Training
# ========================

def train_client(client_id: int, data: Tuple, epochs: int = 3, verbose: int = 0) -> Tuple[keras.Model, List[np.ndarray]]:
    """
    Train a model on client's local data
    Returns: (trained_model, weights)
    """
    x_train, y_train = data
    
    print(f"\n[Client {client_id}] Training on {len(x_train)} samples...")
    
    model = create_model()
    model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=verbose)
    
    weights = model.get_weights()
    
    return model, weights


# ========================
# Homomorphic Encryption
# ========================

def setup_homomorphic_encryption() -> Pyfhel:
    """
    Initialize Pyfhel context for homomorphic encryption
    """
    HE = Pyfhel()
    # Use BFV scheme with larger parameters to handle more data
    # n=2^15 gives us 32768 slots for encryption
    HE.contextGen(scheme='bfv', n=2**15, t_bits=20, sec=128)
    HE.keyGen()
    HE.relinKeyGen()
    
    return HE


def encrypt_weights(weights: List[np.ndarray], HE: Pyfhel) -> List:
    """
    Encrypt model weights using homomorphic encryption
    Splits large arrays into chunks that fit within encryption slots
    Returns: list of encrypted weight arrays
    """
    encrypted_weights = []
    max_slots = HE.get_nSlots()  # Get maximum slots available
    
    for weight_matrix in weights:
        # Flatten the weight matrix
        flat_weights = weight_matrix.flatten()
        
        # Scale to integers (required for BFV scheme)
        scale_factor = 1000
        int_weights = (flat_weights * scale_factor).astype(np.int64)
        
        # Split into chunks if necessary
        weight_size = len(int_weights)
        encrypted_chunks = []
        
        if weight_size <= max_slots:
            # Encrypt the entire array at once
            encrypted_chunks.append(HE.encryptInt(int_weights))
        else:
            # Split into chunks that fit
            num_chunks = (weight_size + max_slots - 1) // max_slots
            for i in range(num_chunks):
                start_idx = i * max_slots
                end_idx = min((i + 1) * max_slots, weight_size)
                chunk = int_weights[start_idx:end_idx]
                encrypted_chunks.append(HE.encryptInt(chunk))
        
        # Store encrypted chunks with metadata
        encrypted_weights.append({
            'encrypted': encrypted_chunks,
            'shape': weight_matrix.shape,
            'scale': scale_factor,
            'size': weight_size
        })
    
    return encrypted_weights


def decrypt_weights(encrypted_weights: List, HE: Pyfhel) -> List[np.ndarray]:
    """
    Decrypt aggregated weights
    Handles chunked encrypted data
    Returns: list of decrypted weight arrays
    """
    decrypted_weights = []
    
    for enc_weight_dict in encrypted_weights:
        encrypted_chunks = enc_weight_dict['encrypted']
        shape = enc_weight_dict['shape']
        scale = enc_weight_dict['scale']
        size = enc_weight_dict['size']
        
        # Decrypt all chunks and concatenate
        decrypted_parts = []
        for chunk in encrypted_chunks:
            decrypted_chunk = HE.decryptInt(chunk)
            decrypted_parts.append(decrypted_chunk)
        
        # Concatenate all decrypted parts
        decrypted_flat = np.concatenate(decrypted_parts)[:size]
        
        # Convert back to float and reshape
        decrypted_flat = decrypted_flat.astype(np.float32) / scale
        decrypted_matrix = decrypted_flat.reshape(shape)
        
        decrypted_weights.append(decrypted_matrix)
    
    return decrypted_weights


# ========================
# Federated Aggregation
# ========================

def aggregate_encrypted_weights(encrypted_weights_list: List[List], HE: Pyfhel) -> List:
    """
    Aggregate (average) encrypted weights from multiple clients
    Performs homomorphic addition without decryption
    Handles chunked encrypted data
    Returns: aggregated encrypted weights
    """
    num_clients = len(encrypted_weights_list)
    
    print(f"\n[Server] Aggregating encrypted weights from {num_clients} clients...")
    
    # Initialize aggregated weights
    aggregated = []
    
    # For each layer
    num_layers = len(encrypted_weights_list[0])
    for layer_idx in range(num_layers):
        # Get the number of chunks for this layer
        num_chunks = len(encrypted_weights_list[0][layer_idx]['encrypted'])
        aggregated_chunks = []
        
        # Aggregate each chunk separately
        for chunk_idx in range(num_chunks):
            # Start with first client's encrypted chunk (copy)
            sum_encrypted = encrypted_weights_list[0][layer_idx]['encrypted'][chunk_idx].copy()
            
            # Add encrypted chunks from other clients (homomorphic addition)
            for client_idx in range(1, num_clients):
                sum_encrypted += encrypted_weights_list[client_idx][layer_idx]['encrypted'][chunk_idx]
            
            aggregated_chunks.append(sum_encrypted)
        
        # Store with metadata
        aggregated.append({
            'encrypted': aggregated_chunks,
            'shape': encrypted_weights_list[0][layer_idx]['shape'],
            'scale': encrypted_weights_list[0][layer_idx]['scale'] * num_clients,  # Adjust scale for averaging
            'size': encrypted_weights_list[0][layer_idx]['size']
        })
    
    print("[Server] Aggregation complete (weights remain encrypted)")
    
    return aggregated


# ========================
# Evaluation
# ========================

def evaluate_model(model: keras.Model, test_data: Tuple, label: str = "Model") -> float:
    """
    Evaluate model on test data
    Returns: accuracy
    """
    x_test, y_test = test_data
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"[{label}] Test Accuracy: {accuracy*100:.2f}%")
    return accuracy


# ========================
# Main Federated Learning Pipeline
# ========================

def main():
    print("="*60)
    print("Privacy-Preserving Federated Learning with Homomorphic Encryption")
    print("="*60)
    
    # Configuration
    NUM_CLIENTS = 3
    EPOCHS = 3
    
    # Step 1: Load and split data
    print("\n[Step 1] Loading and splitting MNIST data...")
    client_datasets, test_data = load_and_split_data(NUM_CLIENTS)
    print(f"Data split among {NUM_CLIENTS} clients")
    
    # Step 2: Initialize homomorphic encryption
    print("\n[Step 2] Setting up homomorphic encryption...")
    HE = setup_homomorphic_encryption()
    print("Encryption context created")
    
    # Step 3: Train clients locally
    print("\n[Step 3] Training local models...")
    client_models = []
    client_weights = []
    
    for client_id in range(NUM_CLIENTS):
        model, weights = train_client(client_id, client_datasets[client_id], epochs=EPOCHS)
        client_models.append(model)
        client_weights.append(weights)
        
        # Evaluate local model
        evaluate_model(model, test_data, label=f"Client {client_id}")
    
    # Step 4: Encrypt client weights
    print("\n[Step 4] Encrypting client weights...")
    encrypted_weights_list = []
    for client_id in range(NUM_CLIENTS):
        print(f"[Client {client_id}] Encrypting weights...")
        enc_weights = encrypt_weights(client_weights[client_id], HE)
        encrypted_weights_list.append(enc_weights)
    print("All weights encrypted")
    
    # Step 5: Server aggregates encrypted weights
    print("\n[Step 5] Server-side aggregation...")
    aggregated_encrypted = aggregate_encrypted_weights(encrypted_weights_list, HE)
    
    # Step 6: Decrypt aggregated weights
    print("\n[Step 6] Decrypting aggregated weights...")
    global_weights = decrypt_weights(aggregated_encrypted, HE)
    print("Global model weights decrypted")
    
    # Step 7: Create and evaluate global model
    print("\n[Step 7] Evaluating global federated model...")
    global_model = create_model()
    global_model.set_weights(global_weights)
    
    global_accuracy = evaluate_model(global_model, test_data, label="Global Federated Model")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Number of clients: {NUM_CLIENTS}")
    print(f"Training epochs per client: {EPOCHS}")
    print(f"\nLocal Model Accuracies:")
    for client_id in range(NUM_CLIENTS):
        acc = evaluate_model(client_models[client_id], test_data, label=f"  Client {client_id}")
    print(f"\nGlobal Federated Model Accuracy: {global_accuracy*100:.2f}%")
    print("\n✓ Privacy-preserving federated learning completed successfully!")
    print("  Weights were aggregated while encrypted using homomorphic encryption.")
    print("="*60)


if __name__ == "__main__":
    main()
