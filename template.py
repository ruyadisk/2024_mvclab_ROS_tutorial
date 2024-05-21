#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from ultralytics import YOLO

class YoloInference:
    def __init__(self):
        self.sub = None # Subscriber code here
        self.pub = None # Publisher code here
        self.model = YOLO('weight/yolov8n.pt')
        self.bridge = CvBridge()
        self.rate = rospy.Rate(10)
        
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, '8UC3')
        except CvBridgeError as e:
            rospy.logerr(e)
            return
        
        results = self.model(cv_image)
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                cls = box.cls
                conf = box.conf
                cv2.rectangle(cv_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(cv_image, f'{self.model.names[int(cls.item())]} {conf.item():.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        ret = self.bridge.cv2_to_imgmsg() # transform code here
        self.pub.publish() # you should publish a image_topic in imgmsg format.
        self.rate.sleep()
        
    def listener(self):
        # ros spin code here
        
if __name__ == '__main__':
    rospy.init_node('YoloInference',anonymous=False)
    obj = YoloInference()
    try:
        obj.listener()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
