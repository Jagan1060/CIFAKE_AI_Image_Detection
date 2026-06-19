import matplotlib.pyplot as plt

train_loss = [0.3344,0.2238,0.1826,0.1513,0.1289,
              0.1073,0.0899,0.0790,0.0690,0.0615]

val_loss = [0.2288,0.2301,0.2057,0.1741,0.1961,
            0.1865,0.2367,0.2084,0.2448,0.2420]

epochs = range(1,11)

plt.figure(figsize=(8,5))
plt.plot(epochs, train_loss, label="Training Loss")
plt.plot(epochs, val_loss, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()

plt.savefig("outputs/loss_graph.png")
plt.show()