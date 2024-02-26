import os
import numpy as np
import cv2

class C3D_Preprocessing():
    def __init__(self, path):
        self.path = path

    def Preprocess_Data(self, input, label, depth, overlap, step, show=False):
        '''
        :param input: input: (N, 64, 64, 3) 원본 numpy array
        :param label: label: (N, 2) 원본 numpy array
        :param depth: 차원 추가 할 프레임 수, <=1 sec, <= 30 frames.
        :param overlap: apply 되는 오버랩핑 프레임 수. 0= no overlap. overlap < frame.
        :param step: data preprocessing 하는데 빠르기 (비례하지는 않음). Normally, 10~100
        :param show: 어떻게 프레임들이 나뉘어 지는지 보기 (== True)
        :return: input(N2, 64, frames, 64, 3), label (N2, 2)
        '''
        slide = depth - overlap
        size = input.shape[0]
        total_size = size - (size - depth) % slide
        if show == True: print('total_size ', total_size)
        travel = int((size - depth) / slide)
        iter = int((travel - (travel % step)) / step)
        rest = travel % step

        start = 0
        end = depth
        endt = depth - 1
        all_inputs = np.reshape(input[start:end, :, :, :], (-1, depth, 64, 64, 3))
        all_labels = np.reshape(label[endt, :], (-1, 2))
        if show == True: print('[0]:\n', f'Input= [{start}: {end}]', f'label= [{endt}]\n')
        for n in range(iter):
            if show == True: print('[%d]\n' % (n + 1))
            for i in range(step):
                start = end - overlap
                end = start + depth
                endt = end - 1
                tmp = np.reshape(input[start:end, :, :, :], (-1, depth, 64, 64, 3))
                tml = np.reshape(label[endt, :], (-1, 2))
                d = np.concatenate((all_inputs, tmp))
                dt = np.concatenate((all_labels, tml))
                all_inputs = d
                all_labels = dt
                if show == True: print(f', Input= [{start}: {end}]', f'label= [{endt}]', end=' ')
            if show == True: print('\n')
        for i in range(rest):
            if show == True: print('[%d]\n' % (i + iter + 1))
            start = end - overlap
            end = start + depth
            endt = end - 1
            tmp = np.reshape(input[start:end, :, :, :], (-1, depth, 64, 64, 3))
            tml = np.reshape(label[endt, :], (-1, 2))
            d = np.concatenate((all_inputs, tmp))
            dt = np.concatenate((all_labels, tml))
            all_inputs = d
            all_labels = dt
            if show == True: print(f', Input= [{start}: {end}]', f'label= [{endt}]', end=' ')
            if show == True: print('\n')
        if show == True: print(all_inputs.shape)
        if show == True: print(all_labels.shape)
        return all_inputs, all_labels


    def Preprocess_OnlyLabel(self, label, depth, overlap, step, show=False):
        '''
        :param label: label: (N, 2) 원본 numpy array
        :param depth: 차원 추가 할 프레임 수, <=1 sec, <= 30 frames.
        :param overlap: apply 되는 오버랩핑 프레임 수. 0= no overlap. overlap < frame.
        :param step: data preprocessing 하는데 빠르기 (비례하지는 않음). Normally, 10~100
        :param show: 어떻게 프레임들이 나뉘어 지는지 보기 (== True)
        :return: label (N2, 2)
        '''
        '''
        f= 16, ovp = 8
        [0 0 0 0 1 0 0 0 0 0 0 1 0 1 1 1] ==> [0 0 0 0 1 0 0 0 | 0 0 0 1 0 1 1 1] ==> 0 0 0 0 1 0 0 0 < 0 0 0 1 0 1 1 1 ==> [0, 1]
        
        '''

        # slide = 한 영상에서 overlap 만큼 훑고 지나가는 프레임 이동 단위
        slide = depth - overlap
        # size = 한 영상에서 전체 프레임 수
        size = label.shape[0]
        iter = int(size/slide) -1
        half = int(depth/2)

        target_value = []
        if show == True: print(f'shape = ({iter}, 2)', end='\n')
        for s in range(iter):

            k = s * slide
            index = '%04d' % s
            start = '%04d' % k
            end = '%04d' % ((depth-1)+k)
            if show == True: print(f'[{index}: {start}~{end}]= ', end='')

            tmp = []
            n = 0
            for frame in range(depth):
                if (frame+k) < size:
                    tmp.append(label[frame+k])
                    n+=1
            if show == True: print(f'{n} elements,', end=' ')
            t = np.array(tmp)
            front = t[0:half, 1].sum()
            back = t[half:depth, 1].sum()
            summation = '%02d'%(t[:, 1].sum())
            if show == True: print(f'front: {front}, back: {back}, sum: {summation},', end=' ')

            if back >= front and back > 0:
                target_value.append([0, 1])
            else:
                target_value.append([1, 0])
            if show == True: print(f'fixed label: {target_value[s]}')

        all_labels = np.reshape(target_value, (-1, 2))


        return all_labels

    def Preprocess_OnlyInput(self, input, depth, overlap, step, show=False):
        '''
        :param input: input: (N, 64, 64, 3) 원본 numpy array
        :param depth: 차원 추가 할 프레임 수, <=1 sec, <= 30 frames.
        :param overlap: apply 되는 오버랩핑 프레임 수. 0= no overlap. overlap < frame.
        :param step: data preprocessing 하는데 빠르기 (비례하지는 않음). Normally, 10~100
        :param show: 어떻게 프레임들이 나뉘어 지는지 보기 (== True)
        :return: input(N2, 64, frames, 64, 3)
        '''
        slide = depth - overlap
        size = input.shape[0]
        total_size = size - (size - depth) % slide
        if show == True: print('total_size ', total_size)
        travel = int((size - depth) / slide)
        iter = int((travel - (travel % step)) / step)
        rest = travel % step

        start = 0
        end = depth
        all_inputs = np.reshape(input[start:end, :, :, :], (-1, depth, 64, 64, 3))

        if show == True: print('[0]:\n', f'Input= [{start}: {end}]\n')
        for n in range(iter):
            if show == True: print('[%d]\n' % (n + 1))
            for i in range(step):
                start = end - overlap
                end = start + depth
                tmp = np.reshape(input[start:end, :, :, :], (-1, depth, 64, 64, 3))
                d = np.concatenate((all_inputs, tmp))
                all_inputs = d
                if show == True: print(f', Input= [{start}: {end}]', end=' ')
            if show == True: print('\n')
        for i in range(rest):
            if show == True: print('[%d]\n' % (i + iter + 1))
            start = end - overlap
            end = start + depth
            tmp = np.reshape(input[start:end, :, :, :], (-1, depth, 64, 64, 3))
            d = np.concatenate((all_inputs, tmp))
            all_inputs = d
            if show == True: print(f', Input= [{start}: {end}]', end=' ')
            if show == True: print('\n')
        if show == True: print(all_inputs.shape)
        return all_inputs


    def MakeDir_CZ3D(self, text_dir, depth, overlap, type):
        path = self.path
        '''
        :param text_dir: 파일 이름 및 경로
        :param depth: 차원 추가 할 프레임 수
        :param overlap: 오버랩 프레임 수
        :return: directory (string)
        '''
        directory = path + text_dir + '_f' + str(depth) + '_ovp' + str(overlap) + '_' + type + '/'
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error: Creating directory. ' + directory)

        return directory



