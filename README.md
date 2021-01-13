# DyLoc: Dynamic Localization for Massive MIMO Using Predictive Recurrent Neural Networks

## Matlab Codes

In order to generates datasets required for training and testing DyLoc, we have to sart with matlab codes,. First of all, you should generate scattring environments using DeepMIMO. 
The codes for DeepMIMO and instruction for using it can be found in [DeepMIMO](http://www.deepmimo.net/). After you create your environment based on the TABLE I of the paper (or any other arbitrary parameters), you should store the resulted struct array in "DeepMIMO_dataset". Next go the "Matlab Codes" folder. "Dataset_gen.m" generates a dataset (pairs of CSI-Location) for traning DCNN, and "Dataset_gen_FPL.m" generates 1000 testframes for evaluating Dyloc. "Moving_Dataset_gen.m" generates the moving datasets for training and testing PredRNN. All of these three codes save their outout in CSV files.

## Convert CSV to NPZ

Next, go to "mat2npz" folder, copy the output CSVs of Matlab codes in the "Data" folder and then run "run.py" codes to convert CSVs to one consolidated npz file. 

## Training DCNNs

Next, go to DCNN training folder, copy the output npz files to "Data" folder and run "run.py". The output weights will be stored in "weights" folder.

## Training and Testing DCNN+WKNN

Next, go to KNNCNN folder. First copy the output datasets (npz files) to "Data" folder. Then run "MakeADPgrid.py" to form and save the grid in "ADP" folder. Then run "train.py" to train the classification DCNN. The weights will be saved in "KNNCNNW1" folder. Finally, you can test the performance of DCNN+WKNN in the three mentioned scenarios in the paper by running "integration.py". The RSE (root square error) of each experiment will saved in "Results" folder.

## Training PredRNN 

Next, go to PredRNN training folder, copy the output npz files to "Data" folder and run "run.py". The output weights will be stored in "checkpoints/mnist_predrnn" folder.

## Testing DyLoc

Finally, everything is available to test DyLoc. Copy PredRNN weights to "PredRNN Models" folder. Copy DCNN weights to "DCNNweights" folder. Copy npz files to "Data" folder. Next run "run.py". The code will measure the performance of DyLoc and DCNN in the three mentioned scenarios in the paper. The RSE of each experiment will saved in "Results" folder.

(for Farzam: change the name of CSV that containes results in the code)

## Get Rid of Matlab Codes

In case you dont want to bother yourself running matlab codes. All reqired .npz files can be found in [Datasets](https://drive.google.com/drive/folders/1zXTY_Kx6ODgQFKLPeeEJ-ax2rfIZyAxR?usp=sharing). You need to copy and paste them in Data folder for doing the correspong job! Having this foler there is no need for running any Matlab codes or conversion from CSV to NPZ.

## Get Rid of Training Neural Networks

If you dont want to bother training and testing networks, we left weights files that can be used by the networks, so just download the "Data" folder and run the "run.py" and enjoy! 

## Improving Results

To improve the results significantly specifically for both O1 and I3 environments, you only need to increase bandwdith or the number ULA elements.


## Questions

If you have any question regarding the codes, dont hesitate to contact me at farzam.hejazi@gmail.com.

## Comments for Farzam 

Add CSV to npz conversion for Moving datasets (both Test and Train) 

Add comments within the codes 

Remove .ipynb folder from KNNCNN folder

Change the name of CSVs that store results in the code

