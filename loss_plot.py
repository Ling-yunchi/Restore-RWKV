import matplotlib.pyplot as plt

from data.common import dataIO

io = dataIO()

eval_loss_path = "experiment/Restore_RWKV/evaluationLoss.bin"
eval_loss = io.load(eval_loss_path)

val_loss = eval_loss["val_loss"]
val_accuracy = eval_loss["val_accuracy"]

# epoch is idx*20 , val_loss[idx] is loss, val_accuracy[idx] is accuracy, in one figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.plot([i * 20 for i in range(len(val_loss))], val_loss)
ax1.set_title('Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')

ax2.plot([i * 20 for i in range(len(val_accuracy))], val_accuracy)
ax2.set_title('Validation Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')

plt.show()
