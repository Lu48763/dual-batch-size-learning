import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

DPI = 300

bs = np.arange(1, 501)
dir_path = '../scp'

pt_epoch_true_val = np.load(f'./{dir_path}/pt_train_time_1_501_1.npy')
tf_epoch_true_val = np.load(f'./{dir_path}/tf_train_time_1_501_1.npy')

pt_batch_true_val = pt_epoch_true_val / np.ceil(50000/bs)
tf_batch_true_val = tf_epoch_true_val / np.ceil(50000/bs)

pt_reg_model = LinearRegression().fit(bs.reshape(-1,1), pt_batch_true_val)
tf_reg_model = LinearRegression().fit(bs.reshape(-1,1), tf_batch_true_val)

pt_batch_predict_val = pt_reg_model.predict(bs.reshape(-1,1))
tf_batch_predict_val = tf_reg_model.predict(bs.reshape(-1,1))

pt_epoch_predict_val = pt_batch_predict_val * np.ceil(50000/bs)
tf_epoch_predict_val = tf_batch_predict_val * np.ceil(50000/bs)

plt.figure(dpi=DPI)
plt.plot(
    bs,
    pt_epoch_true_val,
    label='PyTorch, measurement',
    linewidth=1,
)
plt.plot(
    bs,
    pt_epoch_predict_val,
    label='PyTorch, prediction',
    linestyle='--',
    linewidth=1,
)
plt.plot(
    bs,
    tf_epoch_true_val,
    label='TensorFlow, measurement',
    linewidth=1,
)
plt.plot(
    bs,
    tf_epoch_predict_val,
    label='TensorFlow, prediction',
    linestyle='--',
    linewidth=1,
)
#plt.title('Training Time an Epoch')
plt.xlabel('Batch Size')
plt.ylabel('Time (sec)')
plt.legend()
plt.savefig('train_time_an_epoch.png', transparent=True)

plt.figure(dpi=DPI)
plt.plot(
    bs,
    pt_batch_true_val,
    label='PyTorch, measurement',
    linewidth=1,
)
plt.plot(
    bs,
    pt_batch_predict_val,
    label='PyTorch, prediction',
    linestyle='--',
    linewidth=1,
)
plt.plot(
    bs,
    tf_batch_true_val,
    label='TensorFlow, measurement',
    linewidth=1,
)
plt.plot(
    bs,
    tf_batch_predict_val,
    label='TensorFlow, prediction',
    linestyle='--',
    linewidth=1,
)
#plt.title('Training Time a Batch')
plt.xlabel('Batch Size')
plt.ylabel('Time (sec)')
plt.legend()
plt.savefig('train_time_a_batch.png', transparent=True)
