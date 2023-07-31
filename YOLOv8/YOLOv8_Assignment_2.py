from ultralytics import YOLO
import cv2
from pytube import YouTube
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
FILE_PATH = './Results/'
if not os.path.isdir(FILE_PATH): os.mkdir(FILE_PATH)


###### Code for YOLOv8 (n, s, m and l model) on your own image. ######
# model_types = ['n', 's', 'm', 'l']
# for type in model_types:
#     model = YOLO('yolov8' + type + '-seg.pt')
#     img = model(['https://user-images.githubusercontent.com/34116562/51802668-4c1b8980-2272-11e9-9994-327650ca7b52.jpg'])[0].plot()
#     cv2.imshow("result", img)
#     cv2.imwrite(FILE_PATH + 'yolov8' + type + '-seg_img.jpg', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


###### Code for YOLOv8s on a video. ######
# class VideoObjectDetection:
#     def __init__(self, url: str) -> None:
#         self.downloadVideo(url)
#         self.video, self.model = cv2.VideoCapture(FILE_PATH + self.video_title + '.mp4'), YOLO('yolov8n-seg.pt')
    
#     def downloadVideo(self, url: str) -> None:
#         yt = YouTube(url)
#         if not os.path.isfile(FILE_PATH + yt.title + '.mp4'):
#             stream = yt.streams.filter(res='720p', file_extension='mp4').get_highest_resolution()
#             stream.download(FILE_PATH)
#         self.video_title = yt.title
        
#     def __call__(self) -> None:
#         frame_arr = []
#         while self.video.isOpened() :
#             ret, frame = self.video.read()
#             if ret :
#                 img = self.model(frame)[0].plot()
#                 cv2.imshow("Results", img)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#             frame_arr.append(img)
#         out = cv2.VideoWriter(FILE_PATH + 'Result_2_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (img.shape[1], img.shape[0]))
#         for f in frame_arr: out.write(f)
#         out.release()
#         cv2.destroyAllWindows()
#         self.video.release()

# video = VideoObjectDetection('https://www.youtube.com/shorts/XmkP6YYOHRA')
# video()