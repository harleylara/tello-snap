import jetson.inference
import jetson.utils
import time
#load the recognition network
#net = jetson.inference.detectNet("ssd-mobilenet-v2")

class DetectNetConnector:
    

    def __init__(self):
        print("init")

    def RunInference(self,img,net):
        """
            Run detectNet 

            :param img: image to be detected

            :return myDetections : list of detected objects
        """


        # classify the image

        myDetections =[]
        detections = net.Detect(img)


        for detection in detections:
                myDetections.append([net.GetClassDesc(detection.ClassID),detection.Confidence,round(detection.Top),round(detection.Bottom),round(detection.Left),round(detection.Right)])
        
       
        return myDetections
