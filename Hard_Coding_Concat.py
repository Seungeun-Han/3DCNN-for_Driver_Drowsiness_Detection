from DataTransform import *
from C3D_Preprocess import *
import os
import numpy as np
import time
from PIL import Image

'''================================================================================================='''
input_dir = "../../Dataset/f16_ovp8_gc_righteye_npy/C3D_5_26_f16_ovp8_gc_righteye/"
label_dir = "../../Dataset/f16_ovp8_gc_righteye_npy/C3D_5_26_f16_ovp8_gc_righteye_label/"
saveInput_dir = '../../Dataset/'
saveLabel_dir = '../../Dataset/'
'''================================================================================================='''

inputs_dir = []
labels_dir = []
for dir in os.listdir(input_dir):
    d = os.path.join(input_dir, dir)
    inputs_dir.append(d)
    print(dir)

for dir in os.listdir(label_dir):
    d = os.path.join(label_dir, dir)
    labels_dir.append(d)
    print(dir)

print('\n')
input_list = []
label_list = []
length = len(inputs_dir)
for i in range(length):
    npyInput = np.load(inputs_dir[i])
    input_list.append(npyInput)
    npyLabel = np.load(labels_dir[i])
    label_list.append(npyLabel)
    print(i+1, 'load')
print('\n')
inputs = input_list[0]
labels = label_list[0]
for i in range(len(input_list)-1):
    p = np.concatenate((inputs, input_list[i + 1]))
    inputs = p
    print(i+1, 'input concat')
print('\n')
for i in range(len(input_list)-1):
    t = np.concatenate((labels, label_list[i + 1]))
    labels = t
    print(i+1, 'label concat')

print('\n')
print(inputs.shape)
print(labels.shape)

saveInput_dir += 'Inputs_f16p8_gc_righteye.npy'
saveLabel_dir += 'Labels_f16p8_gc_righteye.npy'
np.save(saveInput_dir, inputs)
np.save(saveLabel_dir, labels)
print('Loading again ...\n')
# CHECK
x = np.load(saveInput_dir)
y = np.load(saveLabel_dir)


