import sys
from glob import glob
import itertools as it
import cv2
import re
import string
import xml.etree.ElementTree as ET


def get_bbox_xml(xml_fname, framecount):


	root = ET.parse(xml_fname).getroot()
	#img = root[2][framecount]
	#print root[2][framecount].attrib["left"] #prints the frame count

	bbox = []
	x1 = int(root[2][framecount][0].attrib['left'])
	y1 = int(root[2][framecount][0].attrib['top'])
	x2 = x1 + int(root[2][framecount][0].attrib['width'])
	y2 = y1 + int(root[2][framecount][0].attrib['height'])
	bbox.append(x1)
	bbox.append(y1)
	bbox.append(x2)
	bbox.append(y2) 
	return bbox

if __name__ == '__main__':
	bbox_gt = get_bbox_xml('./bboxset.xml', 300)
	#print(bbox_gt)