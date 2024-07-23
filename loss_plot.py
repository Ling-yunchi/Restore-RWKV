import matplotlib.pyplot as plt

from data.common import dataIO

io = dataIO()

running_loss_path = "experiment/Restore_RWKV/running_loss.bin"
eval_loss_path = "experiment/Restore_RWKV/eval_loss.bin"

running_loss = io.load(running_loss_path)
eval_loss = io.load(eval_loss_path)

val_loss = eval_loss["val_loss"]
val_accuracy = eval_loss["val_accuracy"]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))
ax1.plot([i for i in range(len(running_loss))], running_loss)
ax1.set_title('Running Loss')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Loss')

ax2.plot([i * 20 for i in range(len(val_loss))], val_loss)
ax2.set_title('Validation Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')

ax3.plot([i * 20 for i in range(len(val_accuracy))], val_accuracy)
ax3.set_title('Validation Accuracy')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Accuracy')

plt.show()
