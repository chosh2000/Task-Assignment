import os
import sys
path = "/home/sanghyun/Downloads/tud-pedestrians"
i = 1
for file1 in os.listdir(path):
    os.rename(os.path.join(path, file1), os.path.join(
        #path, "subject25"+"."+str(i)+".jpg"))
        #path, "25" + "." + str(i) + ".jpg"))
		path, str(i)+".png"))
    i = i + 1