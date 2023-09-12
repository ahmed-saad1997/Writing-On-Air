import cv2
from OCR import *
from Mediapipe_Model import *
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', default=None ,help='out put video path')
args=parser.parse_args()

mediapipe_model = Mediapipe_Model()
ocr_model = OCRmodel()
ocr_model.to(device)
pretrained_weights=torch.load('ocr_weights(VGG16).pt',map_location=device)
ocr_model.load_state_dict(pretrained_weights)

cap = cv2.VideoCapture(0)
mask1 = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),3)
                , dtype=np.uint8)
mask = np.ones((32,128), dtype=np.uint8)*255

videowriter = cv2.VideoWriter_fourcc(*'mp4v') if args.path is not None else None
out=cv2.VideoWriter(args.path, videowriter, 30, (1280, 480)) if args.path is not None else  None


def draw_line(prev_pt,curr_pt,img,color,thick):
    if prev_pt==(0,0):
        prev_pt=curr_pt
    pt0=(int(prev_pt[0]*img.shape[1]),int(prev_pt[1]*img.shape[0]))
    pt1 = (int(curr_pt[0] * img.shape[1]), int(curr_pt[1] * img.shape[0]))
    cv2.line(img, pt0, pt1, color, thick, lineType=cv2.LINE_AA)


prev_pt0=(0,0)
text = ''

while True:
    r,f=cap.read()
    f = cv2.flip(f, 1)
    img = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    index_position,action = mediapipe_model(img)
    if index_position is not None:
        cv2.circle(img,(int(index_position.x*img.shape[1]),int(index_position.y*img.shape[0])),
                   10,(0,255,0),2)
    if action=='write':
        draw_line(prev_pt0,(index_position.x,index_position.y),mask,(0,0,0),1)
        draw_line(prev_pt0, (index_position.x, index_position.y), mask1, (255, 255, 255),5)
        prev_pt0 = (index_position.x, index_position.y)
    elif action == 'break':
        prev_pt0 = (0, 0)
    elif action == 'predict':
        pr = ocr_model.predict(mask)
        if len(text.split('\n')[-1])+len(pr)>30:
            text += '\n'
        if len(pr)>0:
            text += ' '+pr
        mask = np.ones((32, 128)
                      , dtype=np.uint8) * 255
        mask1 = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3)
                         , dtype=np.uint8)
    elif action=='clear':
        prev_pt0=(0,0)
        text=''
        mask = np.ones((32, 128)
                      , dtype=np.uint8) * 255
        mask1 = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3)
                         , dtype=np.uint8)
    alpha = 0.8
    beta = 1.0 - alpha
    gamma = 0.0
    bottom_left_corner = (50, int(mask1.shape[0]) - 75)
    font_scale = 1
    font_color = (255, 255, 255)
    line_type = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    texts = text.split('\n')
    for i, x in enumerate(texts):
        bottom_left_corner = (50, (int(mask1.shape[0]) - 100) + i * 30)
        cv2.putText(mask1, x, bottom_left_corner, font, font_scale, font_color, line_type)
    if len(text) > 90:
        text = ''
    output = cv2.addWeighted(img, alpha, mask1, beta, gamma)
    side_by_side = cv2.hconcat([output, mask1])
    cv2.imshow('im', cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))
    if videowriter is not None:
        out.write(cv2.cvtColor(side_by_side, cv2.COLOR_BGR2RGB))
    if cv2.waitKey(1) == ord('q'):
        break
if out is not None:
    out.release()


