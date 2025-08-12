PYTHON = python3
GNU = gnuplot
PY_DIR = pyplot_figure
GNU_DIR = gnuplot_figure
all: pyplot gnuplot
pyplot:
	$(PYTHON) plot_pt_tf_train_time_compare.py
	$(PYTHON) plot_extra_105.py
	$(PYTHON) plot_extra_110.py
	$(PYTHON) plot_with_without_factor.py
	-mkdir $(PY_DIR)
	mv *.png $(PY_DIR)
gnuplot:
	$(GNU) train_time_an_epoch.gp
	$(GNU) train_time_a_batch.gp
	$(GNU) loss_105.gp
	$(GNU) acc_105.gp
	$(GNU) loss_110.gp
	$(GNU) acc_110.gp
	$(GNU) loss_MUfactor.gp
	$(GNU) acc_MUfactor.gp
	-mkdir $(GNU_DIR)
	mv *.png $(GNU_DIR)
pngclean:
	-rm *.png
dirclean:
	-rm -r $(PY_DIR) $(GNU_DIR)
clean: pngclean dirclean
