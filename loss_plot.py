import matplotlib.pyplot as plt

from data.common import dataIO

io = dataIO()

running_loss_path = "experiment/Restore_RWKV/running_loss.bin"
eval_loss_path = "experiment/Restore_RWKV/eval_loss.bin"

running_loss = io.load(running_loss_path)
eval_loss = io.load(eval_loss_path)

train_loss = running_loss["train_loss"]
train_accuracy = running_loss["train_accuracy"]

val_loss = eval_loss["val_loss"]
val_accuracy = eval_loss["val_accuracy"]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
ax1.plot([i for i in range(len(train_loss))], train_loss)
ax1.set_title('Training Loss')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Loss')

ax2.plot([i for i in range(len(train_accuracy))], train_accuracy)
ax2.set_title('Training Accuracy')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Loss')

ax3.plot([i * 20 for i in range(len(val_loss))], val_loss)
ax3.set_title('Validation Loss')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss')

ax4.plot([i * 20 for i in range(len(val_accuracy))], val_accuracy)
ax4.set_title('Validation Accuracy')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Accuracy')

plt.show()
plt.savefig("loss_plot.png")
