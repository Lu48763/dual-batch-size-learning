set term pngcairo transparent
set output 'acc_105.png'
set datafile separator ','
set xlabel 'Epoch'
set ylabel 'Testing Accuracy'
#set title 'Accuracy of k=1.05'
set key bottom right Left
set xrange [-5:145]

plot './csv/acc_105.csv' using 1:2 with lines title '0 small, B_L=500', \
     './csv/acc_105.csv' using 1:3 with lines title '1 small, B_S=83', \
     './csv/acc_105.csv' using 1:4 with lines title '2 small, B_S=154', \
     './csv/acc_105.csv' using 1:5 with lines title '3 small, B_S=205', \
     './csv/acc_105.csv' using 1:6 with lines title '4 small, B_S=242'
