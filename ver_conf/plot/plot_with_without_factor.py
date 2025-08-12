import numpy as np
import matplotlib.pyplot as plt

DPI = 300

def restruct_array(content):
    # concatenate all np.ndarray to an array
    time = np.reshape((content.item()['push_time']-content.item()['start_time'])[1:, :], (1, -1))
    train_loss = np.reshape(content.item()['train_loss'][1:, :], (1, -1))
    train_acc = np.reshape(content.item()['train_acc'][1:, :], (1, -1))
    test_loss = np.reshape(content.item()['test_loss'][1:, :], (1, -1))
    test_acc = np.reshape(content.item()['test_acc'][1:, :], (1, -1))
    alles = np.concatenate((time, train_loss, train_acc, test_loss, test_acc))
    # sort the array by push time
    index = np.argsort(alles[0])
    alles = alles[:, index]
    # remove items unused
    index2 = alles[0, :] > 0
    alles = alles[:, index2]
    return alles

dir_path = '../scp/tf_npy'
base = '500'
extra = '1.1'
ls = [1, 2, 3]
small = [38, 87, 127]
with_w = []
with_sqrt_w = []
without_w = []
for num1, num2 in zip(ls, small):
    with_w.append(restruct_array(
        np.load(f'./{dir_path}/tf_extra{extra}_{num1}s_{num2}_{base}.npy', allow_pickle=True)
    ))
    with_sqrt_w.append(restruct_array(
        np.load(f'./{dir_path}/tf_sqrt_extra{extra}_{num1}s_{num2}_{base}.npy', allow_pickle=True)
    ))
    without_w.append(restruct_array(
        np.load(f'./{dir_path}/tf_nw_extra{extra}_{num1}s_{num2}_{base}.npy', allow_pickle=True)
    ))

with_label = []
with_sqrt_label = []
without_label = []
for num1, num2 in zip(ls, small):
    with_label.append(f'{num1} small, $d_S / d_L$')
    with_sqrt_label.append((f'{num1} small,' + '$\sqrt{d_S / d_L}$'))
    without_label.append(f'{num1} small, -')

## testing loss
plt.figure(dpi=DPI)
for i in range(len(with_label)):
    plt.plot(np.arange(1, 561)/4, with_w[i][3], label=with_label[i], linewidth=1)
    plt.plot(np.arange(1, 561)/4, with_sqrt_w[i][3], '--', label=with_sqrt_label[i], linewidth=1)
    plt.plot(np.arange(1, 561)/4, without_w[i][3], ':', label=without_label[i], linewidth=1)
plt.ylim(1, 5)
#plt.title('The Testing Loss versus the Training Epoch')
plt.xlabel('Epoch')
plt.ylabel('Testing Loss')
plt.legend()
plt.savefig('loss_MUfactor.png', transparent=True)

## testing accuracy
plt.figure(dpi=DPI)
for i in range(len(with_label)):
    plt.plot(np.arange(1, 561)/4, with_w[i][4], label=with_label[i], linewidth=1)
    plt.plot(np.arange(1, 561)/4, with_sqrt_w[i][4], '--', label=with_sqrt_label[i], linewidth=1)
    plt.plot(np.arange(1, 561)/4, without_w[i][4], ':', label=without_label[i], linewidth=1)
#plt.title('The Testing Accuracy versus the Training Epoch')
plt.xlabel('Epoch')
plt.ylabel('Testing Accuracy')
plt.legend()
plt.savefig('acc_MUfactor.png', transparent=True)
