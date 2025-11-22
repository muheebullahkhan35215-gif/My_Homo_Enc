import time
import json
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Import functions and objects from your main file.
# Adjust the module name if your file is named differently.
import medical_fl_final as m

# Ensure deterministic seeds
import random, os as _os
_os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Ensure CPU only for reproducible timing on Codespaces
_os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Experiment settings
EPOCH_LIST = [2, 5, 10]
NUM_CLIENTS = 3
SAMPLES_PER_CLIENT = 100  # matches your current small simulation
IMG_SIZE = (64, 64)
BATCH_SIZE = 8

OUT_DIR = "epoch_sweep_results"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

results = []

def run_one_experiment(local_epochs):
    # Load data in the same way your script does.
    client_datasets, test_data = m.create_optimized_medical_data(num_clients=NUM_CLIENTS, samples_per_client=SAMPLES_PER_CLIENT, img_size=IMG_SIZE)
    
    # Initialize HE
    HE = m.setup_homomorphic_encryption()
    
    client_models = []
    client_weights = []
    enc_times = []
    train_start = time.time()
    
    # Train each client locally with given local_epochs
    for cid in range(NUM_CLIENTS):
        t0 = time.time()
        model, weights = m.train_client_memory_optimized(cid, client_datasets[cid], epochs=local_epochs)
        t1 = time.time()
        train_time = t1 - t0
        client_models.append(model)
        client_weights.append(weights)
    
    train_end = time.time()
    total_training_time = train_end - train_start

    # Plaintext aggregation and evaluation
    t0 = time.time()
    plaintext_global = m.aggregate_plaintext_weights(client_weights)
    plain_model = m.create_lightweight_medical_cnn()
    plain_model.set_weights(plaintext_global)
    plain_loss, plain_acc = plain_model.evaluate(test_data[0], test_data[1], verbose=0)
    t1 = time.time()
    plaintext_time = t1 - t0

    # Encrypted aggregation
    encrypted_weights_list = []
    enc_start = time.time()
    for cid in range(NUM_CLIENTS):
        t0c = time.time()
        enc = m.encrypt_weights(client_weights[cid], HE)
        t1c = time.time()
        enc_times.append(t1c - t0c)
        encrypted_weights_list.append(enc)
    enc_end = time.time()
    total_encryption_time = enc_end - enc_start

    # Server aggregation (encrypted)
    aggr_start = time.time()
    aggregated_encrypted = m.aggregate_encrypted_weights(encrypted_weights_list, HE)
    aggr_end = time.time()
    aggregation_time = aggr_end - aggr_start

    # Decrypt aggregated weights
    dec_start = time.time()
    # Your decrypt_weights divides by 3 inside. It expects 3 clients. That matches NUM_CLIENTS.
    global_weights = m.decrypt_weights(aggregated_encrypted, HE)
    dec_end = time.time()
    decryption_time = dec_end - dec_start

    # Evaluate decrypted global model
    global_model = m.create_lightweight_medical_cnn()
    global_model.set_weights(global_weights)
    g_loss, g_acc = global_model.evaluate(test_data[0], test_data[1], verbose=0)

    # Pack metrics
    run_metrics = {
        "local_epochs": local_epochs,
        "num_clients": NUM_CLIENTS,
        "samples_per_client": SAMPLES_PER_CLIENT,
        "total_training_time": total_training_time,
        "plaintext_aggregation_time": plaintext_time,
        "encryption_total_time": total_encryption_time,
        "encryption_times_per_client": enc_times,
        "aggregation_time": aggregation_time,
        "decryption_time": decryption_time,
        "plaintext_accuracy": float(plain_acc),
        "encrypted_accuracy": float(g_acc),
        "average_local_accuracy": float(np.mean([m.evaluate_model(cm, test_data, verbose=False) for cm in client_models])),
        "timestamp": time.time()
    }
    return run_metrics

# Run experiments
for e in EPOCH_LIST:
    print(f"Running epoch sweep point: {e} epochs per client")
    metrics = run_one_experiment(e)
    results.append(metrics)
    print(f" Done. plaintext acc {metrics['plaintext_accuracy']:.4f}, encrypted acc {metrics['encrypted_accuracy']:.4f}, train time {metrics['total_training_time']:.2f}s")

# Save results as JSON and CSV
json_path = os.path.join(OUT_DIR, "epoch_sweep_results.json")
with open(json_path, "w") as jf:
    json.dump(results, jf, indent=2)

csv_path = os.path.join(OUT_DIR, "epoch_sweep_results.csv")
with open(csv_path, "w", newline='') as cf:
    writer = csv.writer(cf)
    header = ["local_epochs", "plaintext_accuracy", "encrypted_accuracy", "average_local_accuracy", "total_training_time", "encryption_total_time", "aggregation_time", "decryption_time"]
    writer.writerow(header)
    for r in results:
        writer.writerow([r["local_epochs"], r["plaintext_accuracy"], r["encrypted_accuracy"], r["average_local_accuracy"], r["total_training_time"], r["encryption_total_time"], r["aggregation_time"], r["decryption_time"]])

# Plotting
epochs = [r["local_epochs"] for r in results]
plain_acc = [r["plaintext_accuracy"] for r in results]
enc_acc = [r["encrypted_accuracy"] for r in results]
times = [r["total_training_time"] for r in results]

plt.figure()
plt.plot(epochs, plain_acc, marker="o", label="Plaintext global")
plt.plot(epochs, enc_acc, marker="o", label="Encrypted global")
plt.xlabel("Epochs per client")
plt.ylabel("Test accuracy")
plt.legend()
plt.title("Epochs vs Accuracy")
plt.grid(True)
plt.savefig(os.path.join(OUT_DIR, "epochs_vs_accuracy.png"))
plt.close()

plt.figure()
plt.plot(epochs, times, marker="o")
plt.xlabel("Epochs per client")
plt.ylabel("Total local training time (s)")
plt.title("Epochs vs Training Time")
plt.grid(True)
plt.savefig(os.path.join(OUT_DIR, "epochs_vs_time.png"))
plt.close()

print("Saved results to", OUT_DIR)
print("Files:")
for f in os.listdir(OUT_DIR):
    print(" ", f)