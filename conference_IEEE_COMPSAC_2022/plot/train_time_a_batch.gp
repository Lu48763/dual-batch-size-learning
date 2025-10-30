set term pngcairo transparent
set output 'train_time_a_batch.png'
set datafile separator ','
set xlabel 'Batch Size'
set ylabel 'Time (sec)'
#set title 'Training Time a Batch'
set key top left Left
set xrange [-10:510]

plot './csv/train_time_a_batch.csv' using 1:2 with lines title 'PyTorch, measurement', \
     './csv/train_time_a_batch.csv' using 1:3 with lines title 'PyTorch, prediction' dashtype 2, \
     './csv/train_time_a_batch.csv' using 1:4 with lines title 'TensorFlow, measurement', \
     './csv/train_time_a_batch.csv' using 1:5 with lines title 'TensorFlow, prediction' dashtype 2
