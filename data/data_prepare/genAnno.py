__author__ = "willy"
#generate training annotational data for our method


import os
import xml.etree.cElementTree as ET


root_dir = r'/media/wl/tv16exp01/detrac'
imgpath = os.path.join(root_dir, 'Insight-MVT_Annotation_Train')
dir_dic = {}
#make the corresponding to training and xml directory
for i in os.listdir(imgpath):
    dir_dic[i.split('_')[-1]] = i

if __name__ == '__main__':

    xmlpath = os.path.join(root_dir, 'DETRAC-Train-Annotations-XML')
    annopath = os.path.join(root_dir, 'anno')

    for i in os.listdir(xmlpath):
        annofile = os.path.join(annopath, dir_dic[i.split('.')[0].split('_')[-1]])
        xmlfile = os.path.join(xmlpath, i)
        os.mkdir(annofile)
        tree = ET.parse(xmlfile)
        root = tree.getroot()#sequence
        #ignore region
        for ignore in root.findall('ignored_region'):
            ignorefile = os.path.join(annofile, 'ignored_region')
            fi = open(ignorefile, 'w')
            fi.write('ignored_region\n')
            fi.write(str(len(ignore.findall('box'))) + '\n')
            for box in ignore.findall('box'):
                str1 = box.get('left') + ' ' + box.get('top') + ' ' + box.get('width') + ' ' + box.get('height') + '\n'
                fi.write(str1)
        fi.close()
        #each frame generate an annotation file
        for frame in root.findall('frame'):
            framefile = os.path.join(annofile, 'img'+"%05d"%int(frame.get('num')))
            ff = open(framefile, 'w')
            ff.write('img'+"%05d"%int(frame.get('num')) + '\n')
            ff.write(str(len(frame[0].findall('target'))) + '\n')
            #print frame[0][0][0].get('left')
            for target in frame[0].findall('target'):
                #print target[0]
                str2 = target[0].get('left') + ' ' + target[0].get('top') + ' ' + target[0].get('width') + ' ' + target[0].get('height') + '\n'
                ff.write(str2)

            ff.close()



