import numpy as np
import os

local_path = "/home/charles/Desktop/ZF_Binding/DeepZF-main"

type = "mouse"

file_path = os.path.join(local_path,f"predictions_Zkscan3_{type}_156_model_numpy.npy")

preds = np.load(file_path)

if len(np.shape(preds)) == 1:
	print("Resizing")
	preds_resized = []
	for i in range(int(len(preds)/12)):
		preds_resized.append(preds[i*12:(i+1)*12])
	preds = np.array(preds_resized)


np.savetxt(os.path.join(local_path,f"predictions_Zkscan3_{type}_156_model.txt"),preds)

#preds = preds[0:7]

preds_resized = []
for bloc_of_pred in preds :
	preds_resized.append([bloc_of_pred[0:4],bloc_of_pred[4:8],bloc_of_pred[8:12]])

preds_resized = np.array(preds_resized)

print(str(preds_resized))
with open(os.path.join(local_path,f"predictions_Zkscan3_{type}_156_model_readable.txt"), "w") as f:
	f.write(str(preds_resized))



