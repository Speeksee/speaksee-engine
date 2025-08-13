import time, os, sys, subprocess
import numpy as np
import cv2
import torch
from torchvision import transforms
from .nets import S3FDNet
from .box_utils import nms_,nms

PATH_WEIGHT = 'model/faceDetector/s3fd/sfd_face.pth'
if os.path.isfile(PATH_WEIGHT) == False:
    Link = "1KafnHz7ccT-3IyddBsL5yi2xGtxAKypt"
    cmd = "gdown --id %s -O %s"%(Link, PATH_WEIGHT)
    subprocess.call(cmd, shell=True, stdout=None)
img_mean = np.array([104., 117., 123.])[:, np.newaxis, np.newaxis].astype('float32')


class S3FD():

    def __init__(self, device='cuda'):

        tstamp = time.time()
        self.device = device

        # print('[S3FD] loading with', self.device)
        self.net = S3FDNet(device=self.device).to(self.device)
        PATH = os.path.join(os.getcwd(), PATH_WEIGHT)
        state_dict = torch.load(PATH, map_location=self.device)

        for key, value in state_dict.items():
            print(f"{key}: {value.device}")  # 첫 번째 파라미터만 출력
            break

        self.net.load_state_dict(state_dict)
        self.net.to(self.device)
        self.net.eval()

        # print('[S3FD] finished loading (%.4f sec)' % (time.time() - tstamp))
    
    def detect_faces(self, image, conf_th=0.8, scales=[1]):

        w, h = image.shape[1], image.shape[0]

        bboxes = torch.empty((0, 5), device=self.device)

        with torch.no_grad():
            for s in scales:
                scaled_img = cv2.resize(image, dsize=(0, 0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR)

                scaled_img = np.swapaxes(scaled_img, 1, 2)
                scaled_img = np.swapaxes(scaled_img, 1, 0)
                scaled_img = scaled_img[[2, 1, 0], :, :]
                scaled_img = scaled_img.astype('float32')
                scaled_img -= img_mean
                scaled_img = scaled_img[[2, 1, 0], :, :]
                
                # Move image to the appropriate device (GPU if available)
                x = torch.from_numpy(scaled_img).unsqueeze(0).to(self.device)

                # Forward pass
                y = self.net(x)

                detections = y.data
                scale = torch.Tensor([w, h, w, h]).to(self.device)  # Ensure scale is on the correct device

                for i in range(detections.size(1)):
                    j = 0
                    while detections[0, i, j, 0] > conf_th:
                        score = detections[0, i, j, 0]
                        pt = (detections[0, i, j, 1:] * scale)  # 텐서로 계산
                        bbox = torch.cat([pt, score.unsqueeze(0)], dim=0)  # score를 함께 붙여서 하나의 텐서로 합침
                        
                        # 텐서 결합
                        bboxes = torch.cat([bboxes, bbox.unsqueeze(0)], dim=0)

                        j += 1

            # print(type(bbox))  # bbox의 타입을 출력
            # print(type(bboxes))  # bboxes의 타입을 출력

            # Move bboxes to GPU before calling nms_
            bboxes = torch.Tensor(bboxes).to(self.device)
            keep = nms_(bboxes, 0.1)

            # Ensure that the output is moved back to the CPU before returning
            bboxes = bboxes[keep].cpu().numpy()

        return bboxes

