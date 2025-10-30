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
ls = [1, 2, 3, 4]
small = [38, 87, 127, 160]
dlr = []
dlr.append(restruct_array(np.load(f'./{dir_path}/tf_4GPU_{base}.npy', allow_pickle=True)))
for num1, num2 in zip(ls, small):
    dlr.append(restruct_array(np.load(f'./{dir_path}/tf_extra{extra}_{num1}s_{num2}_{base}.npy',
                                      allow_pickle=True)))

label = []
label.append(f'0 small, $B_L={base}$')
for num1, num2 in zip(ls, small):
    label.append(f'{num1} small, $B_S={num2}$')

## testing loss
plt.figure(dpi=DPI)
for i in range(len(label)):
    plt.plot(np.arange(1, 561)/4, dlr[i][3], label=label[i], linewidth=1)
plt.ylim(1, 5)
#plt.title('The Testing Loss versus the Training Epoch')
plt.xlabel('Epoch')
plt.ylabel('Testing Loss')
plt.legend()
plt.savefig('loss_110.png', transparent=True)

## testing accuracy
plt.figure(dpi=DPI)
for i in range(len(label)):
    plt.plot(np.arange(1, 561)/4, dlr[i][4], label=label[i], linewidth=1)
#plt.title('The Testing Accuracy versus the Training Epoch')
plt.xlabel('Epoch')
plt.ylabel('Testing Accuracy')
plt.legend()
plt.savefig('acc_110.png', transparent=True)
