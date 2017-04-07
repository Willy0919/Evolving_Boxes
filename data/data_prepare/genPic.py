__author__ = "willy"
import cv2
import os
#generate training pictures for our method, the ignored regions are replaced by black boxes.
if __name__ == "__main__":
    root_dir = r'/datacenter/1/DETRAC'
    imgpath = os.path.join(root_dir, 'Insight-MVT_Annotation_Train')
    ignorepath = os.path.join(root_dir, 'anno')
    img_ignore = os.path.join(root_dir, 'img_ignore')
    for img_dir in os.listdir(imgpath):
        if img_dir[0] != '.':
            ignore_dir = os.path.join(img_ignore, img_dir)
            if not os.path.exists(ignore_dir):
                os.mkdir(ignore_dir)

            ignorefile = os.path.join(ignorepath, img_dir, 'ignored_region')
            image = os.path.join(imgpath, img_dir)

            f = open(ignorefile, 'r')
            lines = f.readlines()
            f.close()

            line_sz = len(lines)

            for img in os.listdir(image):
                img_name = os.path.join(image, img)
                im = cv2.imread(img_name)
                for i in range(2, line_sz):
                    rec = lines[i].strip().split(' ')
                    for x in range(int(float(rec[0])),int(float(rec[0])+float(rec[2]))):
                        for y in range(int(float(rec[1])), int(float(rec[1]) + float(rec[3]))):
                            im[y,x] = [0,0,0]
#                    cv2.rectangle(im,(int(float(rec[0])), int(float(rec[1]))), (int(float(rec[0])+float(rec[2])), int(float(rec[1]) + float(rec[3]))), (0,0,0),-1)
                cv2.imwrite(os.path.join(root_dir, 'img_ignore',img_dir, img), im)
                #cv2.imshow('win',im)

        #cv2.waitKey()
