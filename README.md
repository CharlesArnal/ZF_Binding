# Zinc fingers binding and position weight matrices prediction

A slightly modified version of the code from https://github.com/OrensteinLab/DeepZF/, with a few additional files.

## Prerequisites
The code requires [protein-bert](https://github.com/nadavbra/protein_bert), and works on python 3.6.8 with keras 2.6.0 and tensorflow 2.6.2

## Binding predictions
Assemble the pretrained model model.p for BindZFpredictor by concatenating its splits in the folder BindZF_predictor/code/

`cat x?? > model.p`

Call BindZFpredictor on a csv file containing (a description of) the zinc fingers (and their context) and get the predictions in a tsv file with the following commands (the first one for the human ZFs, the second one for the mouse ZFs) :

`python3  BindZF_predictor/code/main_bindzfpredictor_predict.py -in Zkscan3_exps/Zkscan3_ZFs_df_human_for_binding_predictions.csv -out Zkscan3_exps/Zkscan3_binding_preds_human.tsv -m BindZF_predictor/code/model.p -e BindZF_predictor/code/encoder.p -r 1` 
`python3  BindZF_predictor/code/main_bindzfpredictor_predict.py -in Zkscan3_exps/Zkscan3_ZFs_df_mouse_for_binding_predictions.csv -out Zkscan3_exps/Zkscan3_binding_preds_mouse.tsv -m BindZF_predictor/code/model.p -e BindZF_predictor/code/encoder.p -r 1`

The file `turn_sequence_into_csv_for_binding_pred.py` can help you create csv files in the correct format.

## Position weight matrix predictions
Predict the PWM from csv files containing descriptions of the zinc fingers in the right format (first line for the human ZFs, second one for the mouse ZFs):

`python3.6 PWMpredictor/code/main_PWMprecictor.py -in Zkscan3_exps/Zkscan3_ZFs_df_human_for_PWM.csv -out Zkscan3_exps/predictions_Zkscan3_human_156_model.txt -m Zkscan3_exps/transfer_model_final.h5`
`python3.6 PWMpredictor/code/main_PWMprecictor.py -in Zkscan3_exps/Zkscan3_ZFs_df_mouse_for_PWM.csv -out Zkscan3_exps/predictions_Zkscan3_mouse_156_model.txt -m Zkscan3_exps/transfer_model_final.h5`

The file `prepare_sequence_for_PWM_prediction.py` can help you create the csv files in the correct format.
