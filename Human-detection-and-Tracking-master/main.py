# importing libraries
import cv2
import glob
import os
import time
import sys
import numpy as np
from PIL import Image
import imutils
import argparse
from imutils.object_detection import non_max_suppression
import xml.etree.ElementTree as ET

# initialising variables
subject_label = 1
total_count = 0
subject_six_count = 0
rec_count = 0
ppl_dt_tp = 0


font = cv2.FONT_HERSHEY_SIMPLEX
list_of_videos = []
cascade_path = "./face_cascades/haarcascade_profileface.xml"
#cascade_path = "./face_cascades/haarcascade_frontalface_default.xml"
#cascade_path = "./face_cascades/face.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
recognizer = cv2.face.createLBPHFaceRecognizer()


def get_bbox_xml(xml_fname, framecount, resol):

    root = ET.parse(xml_fname).getroot()
    #img = root[2][framecount]
    #print root[2][framecount].attrib["left"] #prints the frame count
    intresol= int(resol)
    bbox = []
    x1 = int(root[2][framecount][0].attrib['left']) *intresol/1280
    y1 = int(root[2][framecount][0].attrib['top']) *intresol/1280
    x2 = x1+((int(root[2][framecount][0].attrib['width'])) *intresol/1280)
    y2 = y1+((int(root[2][framecount][0].attrib['height'])) *intresol/1280)
    bbox.append(x1)
    bbox.append(y1)
    bbox.append(x2)
    bbox.append(y2) 
    return bbox


class people:

    def detect_people(self, frame, framecount, bbox):
        """
        Detect Humans using HOG descriptor
        Args:
                frame: input frame
                framecount: framecount
                bbox: array of ground truth bounding box coordinates [x1,y1,x2,y2]

        Returns:
                processed frame
                increments total rectangle count (tp+fp)
                increments total true positive count
                increments pplDetectedFrame for recall rate = pplDetected Frame / total frame count
        """
        xb1 = bbox[0]
        yb1 = bbox[1]
        xb2 = bbox[2]
        yb2 = bbox[3]
        (rects, weights) = hog.detectMultiScale(
            frame, winStride=(4, 4), padding=(16, 16), scale=1.06)
        rects = non_max_suppression(rects, probs=None, overlapThresh=0.65)
        for (x, y, w, h) in rects:
            global rec_count
            global ppl_dt_tp
            rec_count = rec_count +1
            x2=x+w
            y2=y+h

            #Calculating Accuracy of ppl detect
            x_overlap = max(0,min(xb2,x2) - max(xb1,x))
            y_overlap = max(0,min(yb2,y2) - max(yb1,y))
            intersectionArea = x_overlap * y_overlap
            unionArea = (abs(xb1 - xb2) * abs(yb1 - yb2)) + (abs(x - x2) * abs(y - y2)) - intersectionArea
            overlapArea = float(intersectionArea)/float(unionArea)
            if (overlapArea >= 0.5):
                ppl_dt_tp+=1


            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        return frame

    def detect_face(self, frame):
        """
        detect human faces in image using haar-cascade
        Args:
                frame:

        Returns:
        coordinates of detected faces
        """
        faces = face_cascade.detectMultiScale(frame)
        return faces

    def recognize_face(self, frame_orginal_grayscale, framecount, resol):
        """
        recognize human faces using LBPH features
        Args:
                frame_orginal:
                faces:

        Returns:
                label of predicted person
        """
        #predict_label = []
        predict_conf = []

        #for x, y, w, h in faces:
            #reduce frame_orginal resolution to compute accuracy of FR
            #x = x*260/360
            #y = y*260/360
            #w = w*260/360
            #h = h*260/360

        #    frame_orginal_grayscale = cv2.cvtColor(
        #        frame_orginal[y: y + h, x: x + w], cv2.COLOR_BGR2GRAY)
        #cv2.imshow("cropped", frame_orginal_grayscale)
        predict_tuple = recognizer.predict(frame_orginal_grayscale)
        #a, b = predict_tuple
        a = predict_tuple
        b = predict_tuple
        #predict_label.append(a)
        predict_conf.append(b)
        print(predict_tuple)
        #cv2.imwrite(os.path.join('./Video/results/facelabel'+str(resol), str(predict_tuple)+"_"+str(framecount)+".jpg"), frame_orginal_grayscale)
        #return predict_label
        return a

# face drawing and putting labels


class make:

    def draw_faces(self, frame, faces):
        """
        draw rectangle around detected faces
        Args:
                frame:
                faces:

        Returns:
        face drawn processed frame
        """
        for (x, y, w, h) in faces:
            xA = x
            yA = y
            xB = x + w
            yB = y + h
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
        return frame

    def put_label_on_face(self, frame, faces, labels):
        """
        draw label on faces
        Args:
                frame:
                faces:
                labels:

        Returns:
                processed frame
        """
        i = 0
        for x, y, w, h in faces:
            cv2.putText(frame, str(labels[i]),
                        (x, y), font, 1, (255, 255, 255), 2)
            i += 1
        return frame



if __name__ == '__main__':
    """
    main function
    """
    ppl = people()
    makke = make()

    #Parsing Arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--videos", required=True,
                    help="path to videos directory")
    ap.add_argument("-r", "--resolution", required=True,
                    help="define resolution")
    args = vars(ap.parse_args())
    path = args["videos"]
    resol = args["resolution"]
    #make directories
    if not os.path.exists("./Video/results/frames"+str(resol)):
        os.makedirs("./Video/results/frames"+str(resol))
    if not os.path.exists("./Video/results/facelabel"+str(resol)):
        os.makedirs("./Video/results/facelabel"+str(resol))


    for f in os.listdir(path):
        list_of_videos = glob.glob(os.path.join(os.path.abspath(path), f))
        print(list_of_videos)
        if os.path.exists("cont.yaml"):
            recognizer.load("cont.yaml")
            for video in list_of_videos:
                camera = cv2.VideoCapture(video)
                framecount = -1
                recogframecount = 1
                total_time_ppldetect = 0
                total_time_facedetect =0
                total_time_facerecog =0
                total_time_this_video = 0
                while True:
                    predict_label = []     #resets the array of face labels for EVERY frame
                    framecount = framecount + 1
                    print('framecount%d'%framecount)
                    bbox_gt = get_bbox_xml('./bboxset.xml', framecount, resol)
                    starttime = time.time()
                    grabbed, frame = camera.read()
                    if not grabbed:
                        print("not grabbed any frame")
                        break
                    before_img = time.time()
                    frame_orginal = imutils.resize(
                        frame, width=min(int(resol), frame.shape[1]))
                    #frame_FR = imutils.resize(  ####################################### get rid of it
                    #    frame, width=min(260, frame.shape[1]))
                    frame_orginal1 = cv2.cvtColor(
                        frame_orginal, cv2.COLOR_BGR2GRAY)
                    after_img = time.time()
                    #cv2.imwrite(os.path.join('./Video/results/frames'+str(resol), "orginal"+str(framecount)+".jpg"), frame_orginal)

                    before_ppldetect = time.time()
                    frame_processed = ppl.detect_people(frame_orginal1, framecount, bbox_gt)
                    after_ppldetect = time.time()
                    total_time_ppldetect=total_time_ppldetect + (after_ppldetect-before_ppldetect)

                    before_facedetect = time.time()
                    faces = ppl.detect_face(frame_orginal)
                    after_facedetect = time.time()
                    total_time_facedetect = total_time_facedetect + (after_facedetect - before_facedetect)

                    if len(faces) == 0:
                        print("did not grab any face")
                    if len(faces) > 0:
                        print("face grabbed")
                        recogframecount = recogframecount + 1
                        before_facerecog = time.time()
                        frame_processed = makke.draw_faces(frame_processed, faces)

                        ##### This code has been modified to send ONLY the face pixels to FR. 
                        for x, y, w, h in faces:
                            frame_orginal_grayscale = cv2.cvtColor(frame_orginal[y: y + h, x: x + w], cv2.COLOR_BGR2GRAY)
                            label = ppl.recognize_face(frame_orginal_grayscale, framecount, resol)
                            predict_label.append(label)

                        #label = ppl.recognize_face(frame_orginal,faces, framecount, resol) ##########change to frame orginal
                        #cv2.imwrite(os.path.join('./Video/results/face'+str(resol), str(framecount)+".jpg"), faces)
                        frame_processed = makke.put_label_on_face(
                            frame_processed, faces, predict_label)
                        for i in predict_label:
                            total_count += 1
                            if i == 6:
                                subject_six_count += 1 
                            if i == 7:
                                subject_six_count += 1

                        after_facerecog = time.time()
                        print("face recog %s seconds" % (after_facerecog - before_facerecog))
                        total_time_facerecog = total_time_facerecog + (after_facerecog - before_facerecog)
                    endtime = time.time()
                    #cv2.imshow("window", frame_processed)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    total_time_this_video = total_time_this_video + (endtime - starttime)
                    print("img processing %s seconds" % (after_img - before_img))
                    print("ppl detect %s seconds" % (after_ppldetect - before_ppldetect))
                    print("face detect %s seconds" % (after_facedetect - before_facedetect))
                    print("total %s seconds\n" % (endtime - starttime))
                camera.release()
                cv2.destroyAllWindows()
                #subject_array = [subject_one_count, subject_two_count, subject_three_count, subject_four_count, subject_five_count, subject_six_count]
                print('Total # of face = %s\n' % total_count)
                print('Enddy(subject 6,7) = %s\n'% subject_six_count)
                
                '''
                txt_file = open("./Video/results/%s.txt" %resol, "w")
                txt_file.write(video)
                txt_file.write('\ntotal frame =%s\n'% framecount)
                txt_file.write('rec_count =%s\n'% rec_count)
                txt_file.write('People Detect TP=%s\n' % ppl_dt_tp)
                txt_file.write('img processing %s seconds\n' % (after_img - before_img))
                txt_file.write('average ppldetect = %s\n' % (total_time_ppldetect/framecount))
                txt_file.write('average facedetect = %s\n' % (total_time_facedetect/framecount))
                txt_file.write('average facerecog = %s\n' % (total_time_facerecog/recogframecount))
                txt_file.write('average time/frame = %s\n' % (total_time_this_video/framecount))
                txt_file.write('Total # of face = %s\n' % total_count)
                txt_file.write('Enddy(subject 6,7) = %s\n'% subject_six_count)
                '''
                exit()

        else:
            print("model file not found")
        list_of_videos = []
