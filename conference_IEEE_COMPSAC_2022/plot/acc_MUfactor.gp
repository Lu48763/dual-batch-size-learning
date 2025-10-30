set term pngcairo transparent
set output 'acc_MUfactor.png'
set datafile separator ','
set xlabel 'Epoch'
set ylabel 'Testing Accuracy'
#set title 'Accuracy of Training With/without Model-Update Factor, k=1.1'
set key bottom right Left
set xrange [-5:145]

plot './csv/acc_MUfactor.csv' using 1:2 with lines title '1 small, d_S/d_L', \
     './csv/acc_MUfactor.csv' using 1:3 with lines title '1 small, √(d_S/d_L)' dashtype 2, \
     './csv/acc_MUfactor.csv' using 1:4 with lines title '1 small, -' dashtype 3, \
     './csv/acc_MUfactor.csv' using 1:5 with lines title '2 small, d_S/d_L', \
     './csv/acc_MUfactor.csv' using 1:6 with lines title '2 small, √(d_S/d_L)' dashtype 2, \
     './csv/acc_MUfactor.csv' using 1:7 with lines title '2 small, -' dashtype 3, \
     './csv/acc_MUfactor.csv' using 1:8 with lines title '3 small, d_S/d_L', \
     './csv/acc_MUfactor.csv' using 1:9 with lines title '3 small, √(d_S/d_L)' dashtype 2, \
     './csv/acc_MUfactor.csv' using 1:10 with lines title '3 small, -' dashtype 3
