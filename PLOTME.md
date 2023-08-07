### `cosndalpha` plot

Set `ALPHA=1` in `quanting.py`.
Select a target value for `alpha` and run the following:
```sh
python plot_toy.py --output_folder toytarget/ --xmax 3 --error --alpha 1.25 --npoints 16
```

### `uquark2d` plot

Set `ALPHA=1` in `quanting.py` and run the following:
```sh
python plot_uquark2d.py --nshots_predictions 1000000 --n_predictions 100 --n_points 20 --output_folder uquark2d_0.001to0.7_lbfgs/
```


### `uquark` plot

Set `ALPHA=0.92` in `quanting.py` and run the following:
```sh
python main.py --load plotscripts/uquark_alpha0.92/best_p.npy --layers 4 --nqubits 1 --ndim 1 --ansatz qpdf_iqm5q --maxiter 0 --target uquark --nshots 1000 --nruns 1 -o uquark1d -j 1 --mitigation true --npoints_plot 20
```