set term pngcairo transparent
set output 'loss_MUfactor.png'
set datafile separator ','
set xlabel 'Epoch'
set ylabel 'Testing Loss'
#set title 'Loss of Training With/without Model-Update Factor, k=1.1'
set key top right Left
set xrange [-5:145]
set yrange [1:5]

plot './csv/loss_MUfactor.csv' using 1:2 with lines title '1 small, d_S/d_L', \
     './csv/loss_MUfactor.csv' using 1:3 with lines title '1 small, √(d_S/d_L)' dashtype 2, \
     './csv/loss_MUfactor.csv' using 1:4 with lines title '1 small, -' dashtype 3, \
     './csv/loss_MUfactor.csv' using 1:5 with lines title '2 small, d_S/d_L', \
     './csv/loss_MUfactor.csv' using 1:6 with lines title '2 small, √(d_S/d_L)' dashtype 2, \
     './csv/loss_MUfactor.csv' using 1:7 with lines title '2 small, -' dashtype 3, \
     './csv/loss_MUfactor.csv' using 1:8 with lines title '3 small, d_S/d_L', \
     './csv/loss_MUfactor.csv' using 1:9 with lines title '3 small, √(d_S/d_L)' dashtype 2, \
     './csv/loss_MUfactor.csv' using 1:10 with lines title '3 small, -' dashtype 3
