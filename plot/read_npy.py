import sys
import numpy as np

def print_results(key, npy):
    best = np.argmax(npy['val_acc'])
    print(
        f'{key},',
        f'time: {npy["commit_time"][-1]: .1f},',
        f'loss: {npy["val_loss"][-1]: .3f},',
        f'acc: {round(npy["val_acc"][-1] * 100, 1)}%,',
        f'index: {best},',
        f'best_loss: {npy["val_loss"][best]: .3f},',
        f'best_acc: {round(npy["val_acc"][best] * 100, 1)}%'
    )

if len(sys.argv) < 2:
    print("read_logs: error: no input files")
    sys.exit()
if len(sys.argv) > 2:
    print("read_logs: error: too many input files")
    sys.exit()
if '.npy' not in sys.argv[1]:
    print('read_logs: error: logs is not a ".npy" file')

logs = np.load(sys.argv[1], allow_pickle=True).item()

print_results(sys.argv[1], logs)
