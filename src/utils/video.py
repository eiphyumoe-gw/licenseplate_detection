import cv2
import os
from tqdm import tqdm

class Video:
    def __init__(self, path, output_path):
        self.__video_path = path
        self.__frame_width = 0
        self.__frame_height = 0
        self.__output_path = os.path.join(output_path, os.path.basename(self.__video_path))
        self.__fps = 0
        self.video_reader()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_output = cv2.VideoWriter(self.__output_path, fourcc, self.__fps, (self.__frame_width, self.__frame_height))
    
    def video_reader(self):
        self.cap = cv2.VideoCapture(self.__video_path)
        self.__fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.__frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.__frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        
    def process_video(self):
        with tqdm(total=self.frame_count) as pbar:
           while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                yield frame
                pbar.update(1)
    
    
    def video_writer(self, frame):
        self.video_output.write(frame)
    
    def close(self):
        self.video_output.release()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exec_type, exec_value, traceback):
        return self.cap.release()
    
    # def save_output(self):
    #     video_output = self.video_writer()
    #     while self.cap.isOpened():
    #         success, frame = self.cap.read()
            
    #         if not success:
    #             print("Ignoring empty frame.")
    #             break
            
    #         cv2.imshow("Test", frame)
    #         video_output.write(frame)
            
    #         if cv2.waitKey(5) & 0xFF == 27:
    #             break
    #     print("Finish")
    #     self.cap.release()
    #     video_output.release()
    #     cv2.destroyAllWindows()
              

    def get_output(self):
        return self.__output_path
    
    def get_vd_path(self):
        return self.__video_path
    