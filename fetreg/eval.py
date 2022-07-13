import numpy as np
import cv2
import os, sys
from sklearn.metrics import f1_score

threshold = int(sys.argv[1])


fns = sorted(os.listdir('./data/preds'))
dice_scores = []
f1_scores = []
for fn in fns:
	gt = np.max(cv2.imread(os.path.join('./data/gts/', fn)), axis=2).astype(np.float32) / 255
	pred = np.max(cv2.imread(os.path.join('./data/preds/', fn)), axis=2).astype(np.float32) / 255
	scale = 255 / np.max(np.load(os.path.join('./data/preds_raw/', fn + '.npy'))[:,:,1])
	pred_raw = np.expand_dims(np.load(os.path.join('./data/preds_raw/', fn + '.npy'))[:,:,1], axis=2) * scale
	#print(np.min(pred_raw), np.max(pred_raw))
	#pred_raw *= (pred_raw > threshold)
	#pred_raw = pred_raw.astype(bool) * 1
	f1 = f1_score(gt.flatten(), pred.flatten())
	f1_scores.append(f1)

	pred_raw *= np.expand_dims(pred, axis=2)

	cv2.imwrite(os.path.join('./data/preds_raw_img/', fn), pred_raw)

	dice = np.sum(pred_raw[gt==1]) * 2.0 / (np.sum(pred_raw) + np.sum(gt))
	#dice = np.sum(pred[gt==1]) * 2.0 / (np.sum(pred) + np.sum(gt))

	dice_scores.append(dice)


print(threshold, 'Dice:', np.mean(np.array(dice_scores)))
print('F1:', np.mean(np.array(f1_scores)))

      
