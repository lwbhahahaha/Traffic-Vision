import os
import csv
import json
from json_flatten import flatten
import random
dict=[]
stat={}
org={}
def rename_keys(curr):
    curr['filename'] = curr['name']
    del curr['name']
    if ('category' in curr):
        curr['class'] = curr['category']
        del curr['category']
    if ('x1' in curr):
        curr['xmin'] = curr['x1']
        del curr['x1']
        curr['ymin'] = curr['y1']
        del curr['y1']
        curr['xmax'] = curr['x2']
        del curr['x2']
        curr['ymax'] = curr['y2']
        del curr['y2']
    del curr['video_name']
    del curr['index']
    if ('id' in curr):
        del curr['id']
    if ('Occluded' in curr):
        del curr['Occluded']
    if ('Truncated' in curr):
        del curr['Truncated']
    if ('Crowd' in curr):
        del curr['Crowd']
    return curr



def json2csv(jsonPath,filename,TestOrTrain):
    """
    df=pandas.read_json(jsonPath+'.json')
    df.to_csv()
    df.to_csv(r'output/'+str(jsonIndex)+'.csv', index=False)
    """
    currJson=json.load(open(jsonPath+'.json'))
    path = 'output/' + filename + '.csv'
    if not os.path.exists(path):
        with open(path, "w", newline='', encoding='utf-8') as csvfile:  # newline='' 去除空白行
            writer = csv.DictWriter(csvfile, fieldnames=dict)  # 写字典的方法
            writer.writeheader()
    for j in currJson:
        curr = flatten(j) #get flattened dict
        curr['width'] = 1280
        curr['height'] = 720
        curr=rename_keys(curr)
        curr['filename']='C:/Users/Administrator.WIN-2EPKD7D6018/Desktop/bdd100k-non/bdd100k/images/track/'+TestOrTrain+'/'+curr['filename'].rsplit('-', 2)[0]+'-'+curr['filename'].rsplit('-', 2)[1]+'/'+curr['filename']
        with open(path, "a", newline='', encoding='utf-8') as csvfile:  # newline='' 一定要写，否则写入数据有空白行
            writer = csv.DictWriter(csvfile, fieldnames=dict)
            writer.writerow(curr)  # 按行写入数据
    print(filename+" write success")

def saveAllTrainToSeperateFile():
    # train convert json into one csv
    filePath = 'C:/Users/Administrator.WIN-2EPKD7D6018/Desktop/bdd100k-non/bdd100k/labels-20/box-track/train/'
    ct = 0
    for i, j, k in os.walk(filePath):
        # k is ####.json
        for path in k:
            json2csv(filePath + path.rsplit('.', 1)[0], "train/train_"+str(ct),"train")
            ct += 1

def saveAllTrainToOneFile():
    filePath = 'C:/Users/Administrator.WIN-2EPKD7D6018/Desktop/bdd100k-non/bdd100k/labels-20/box-track/train/'
    path = 'output/train/train_all.csv'
    csvfile = open(path, "w", newline='', encoding='utf-8')
    writer = csv.DictWriter(csvfile, fieldnames=dict)  # 写字典的方法
    writer.writeheader()

    ct=0
    for i, j, k in os.walk(filePath):
        # k is ####.json
        for path in k:
            jsonPath=filePath + path.rsplit('.', 1)[0]
            currJson = json.load(open(jsonPath + '.json'))
            for j in currJson:
                curr = flatten(j)  # get flattened dict
                curr['width'] = 1280
                curr['height'] = 720
                curr = rename_keys(curr)
                curr['filename'] = 'C:/Users/Administrator.WIN-2EPKD7D6018/Desktop/bdd100k-non/bdd100k/images/track/train/' + \
                                  curr['filename'].rsplit('-', 2)[0] + '-' + curr['filename'].rsplit('-', 2)[1] + '/' + \
                                  curr['filename']

                writer = csv.DictWriter(csvfile, fieldnames=dict)
                writer.writerow(curr)  # 按行写入数据
            print(str(ct)+" write success")
            ct+=1
    print("All write success")

def saveAllTestToOneFile():
    filePath = 'C:/Users/Administrator.WIN-2EPKD7D6018/Desktop/bdd100k-non/bdd100k/labels-20/box-track/val/'
    path = 'output/test/test_all.csv'
    csvfile = open(path, "w", newline='', encoding='utf-8')
    writer = csv.DictWriter(csvfile, fieldnames=dict)  # 写字典的方法
    writer.writeheader()

    ct=0
    for i, j, k in os.walk(filePath):
        # k is ####.json
        for path in k:
            jsonPath=filePath + path.rsplit('.', 1)[0]
            currJson = json.load(open(jsonPath + '.json'))
            for j in currJson:
                curr = flatten(j)  # get flattened dict
                curr['width'] = 1280
                curr['height'] = 720
                curr = rename_keys(curr)
                curr['filename'] = 'C:/Users/Administrator.WIN-2EPKD7D6018/Desktop/bdd100k-non/bdd100k/images/track/test/' + \
                                  curr['filename'].rsplit('-', 2)[0] + '-' + curr['filename'].rsplit('-', 2)[1] + '/' + \
                                  curr['filename']
                writer = csv.DictWriter(csvfile, fieldnames=dict)
                writer.writerow(curr)  # 按行写入数据
            print(str(ct)+" write success")
            ct+=1
    print("All write success")

def saveDetectoin2020Test():
    jsonPath = 'C:/Users/Administrator.WIN-2EPKD7D6018/Desktop/bdd100k-non/bdd100k/detection20/det_v2_val_release.json'
    path = 'outputForDetection2020/Test_Detection2020.csv'
    csvfile = open(path, "w", newline='', encoding='utf-8')
    writer = csv.DictWriter(csvfile, fieldnames=dict)  # 写字典的方法
    writer.writeheader()
    currJson = json.load(open(jsonPath))
    ct=0
    for j in currJson:
        #表头只能有[, 'xmin', 'ymin', 'xmax', 'ymax']
        currName=j['name']
        if j['labels'] is not None:
            for lable in j['labels']:
                if((lable['category'] != 'other person') and (lable['category'] != 'other vehicle') and (lable['category'] != 'trailer') and (lable['category'] != 'train')):
                    currRow = {}
                    currRow['filename']='C:/Users/Administrator.WIN-2EPKD7D6018/Desktop/bdd100k-non/bdd100k/images/100k/val/'+currName
                    currRow['width'] = 1280
                    currRow['height'] = 720
                    currRow['xmin'] = int(lable['box2d']['x1'])
                    currRow['ymin'] = int(lable['box2d']['y1'])
                    currRow['xmax'] = int(lable['box2d']['x2'])
                    currRow['ymax'] = int(lable['box2d']['y2'])
                    if (lable['category'] == 'traffic light'):
                        # G:green, R:red, Y:yellow
                        if (lable['attributes']['trafficLightColor'][1] == 'G'):
                            currRow['class'] = 'Go'
                        elif (lable['attributes']['trafficLightColor'][1] == 'R'):
                            currRow['class'] = 'Stop'
                        elif (lable['attributes']['trafficLightColor'][1] == 'Y'):
                            currRow['class'] = 'Slow Down'
                        else:
                            currRow['class'] = 'traffic light'
                    else:
                        currRow['class'] = lable['category']
                    if (currRow['class'] in stat):
                        stat[currRow['class']]+=1
                    else:
                        stat[currRow['class']] = 1
                    writer = csv.DictWriter(csvfile, fieldnames=dict)
                    writer.writerow(currRow)
                    print(str(ct) + " write success")
                    ct+=1
    print("All write success")

def saveDetectoin2020Train():
    jsonPath = 'C:/Users/Administrator.WIN-2EPKD7D6018/Desktop/bdd100k-non/bdd100k/detection20/det_v2_train_release.json'
    path = 'outputForDetection2020/Train_Detection2020.csv'
    csvfile = open(path, "w", newline='', encoding='utf-8')
    writer = csv.DictWriter(csvfile, fieldnames=dict)  # 写字典的方法
    writer.writeheader()
    currJson = json.load(open(jsonPath))
    ct=0
    for j in currJson:
        #表头只能有[, 'xmin', 'ymin', 'xmax', 'ymax']
        currName=j['name']
        if j['labels'] is not None:
            for lable in j['labels']:
                if (lable['category'] in org):
                    org[lable['category']] += 1
                else:
                    org[lable['category']] = 1
                if(lable['category']!='other person' and lable['category']!='other vehicle' and lable['category']!='trailer' and lable['category']!='train'):
                    currRow = {}
                    currRow['filename']='C:/Users/Administrator.WIN-2EPKD7D6018/Desktop/bdd100k-non/bdd100k/images/100k/train/'+currName
                    currRow['width'] = 1280
                    currRow['height'] = 720
                    currRow['xmin'] = int(lable['box2d']['x1'])
                    currRow['ymin'] = int(lable['box2d']['y1'])
                    currRow['xmax'] = int(lable['box2d']['x2'])
                    currRow['ymax'] = int(lable['box2d']['y2'])
                    """
                    if (lable['attributes']['trafficLightColor'][1] == 'G'):
                        currRow['class'] = 'Go'
                    elif (lable['attributes']['trafficLightColor'][1] == 'R'):
                        currRow['class'] = 'Stop'
                    elif (lable['attributes']['trafficLightColor'][1] == 'Y'):
                        currRow['class'] = 'Slow Down'
                    else:
                        currRow['class'] = lable['category']
                    writer = csv.DictWriter(csvfile, fieldnames=dict)
                    writer.writerow(currRow)
                    print(str(ct) + " write success")
                    ct+=1
                    if ('class' in currRow):
                        if (currRow['class'] in stat):
                            stat[currRow['class']] += 1
                        else:
                            stat[currRow['class']] = 1
                    """
                    if (lable['category'] == 'bicycle'):
                        currRow['class'] = 'bicycle'
                        for i in range(98):
                            writer = csv.DictWriter(csvfile, fieldnames=dict)
                            writer.writerow(currRow)
                            print(str(ct) + " write success")
                            ct += 1
                            if ('class' in currRow):
                                if (currRow['class'] in stat):
                                    stat[currRow['class']] += 1
                                else:
                                    stat[currRow['class']] = 1
                        if (random.randint(0,100) <= 35):
                            writer = csv.DictWriter(csvfile, fieldnames=dict)
                            writer.writerow(currRow)
                            print(str(ct) + " write success")
                            ct += 1
                            if ('class' in currRow):
                                if (currRow['class'] in stat):
                                    stat[currRow['class']] += 1
                                else:
                                    stat[currRow['class']] = 1
                    elif (lable['category'] == 'bus'):
                        currRow['class'] = 'bus'
                        for i in range(58):
                            writer = csv.DictWriter(csvfile, fieldnames=dict)
                            writer.writerow(currRow)
                            print(str(ct) + " write success")
                            ct += 1
                            if ('class' in currRow):
                                if (currRow['class'] in stat):
                                    stat[currRow['class']] += 1
                                else:
                                    stat[currRow['class']] = 1
                        if (random.randint(0,100) <= 50):
                            writer = csv.DictWriter(csvfile, fieldnames=dict)
                            writer.writerow(currRow)
                            print(str(ct) + " write success")
                            ct += 1
                            if ('class' in currRow):
                                if (currRow['class'] in stat):
                                    stat[currRow['class']] += 1
                                else:
                                    stat[currRow['class']] = 1
                    elif (lable['category'] == 'car'):
                        currRow['class'] = 'car'
                        for i in range(1):
                            writer = csv.DictWriter(csvfile, fieldnames=dict)
                            writer.writerow(currRow)
                            print(str(ct) + " write success")
                            ct += 1
                            if ('class' in currRow):
                                if (currRow['class'] in stat):
                                    stat[currRow['class']] += 1
                                else:
                                    stat[currRow['class']] = 1
                        if (random.randint(0,1000) <= -1):
                            writer = csv.DictWriter(csvfile, fieldnames=dict)
                            writer.writerow(currRow)
                            print(str(ct) + " write success")
                            ct += 1
                            if ('class' in currRow):
                                if (currRow['class'] in stat):
                                    stat[currRow['class']] += 1
                                else:
                                    stat[currRow['class']] = 1
                    elif (lable['category'] == 'motorcycle'):
                        currRow['class'] = 'motorcycle'
                        for i in range(231):
                            writer = csv.DictWriter(csvfile, fieldnames=dict)
                            writer.writerow(currRow)
                            print(str(ct) + " write success")
                            ct += 1
                            if ('class' in currRow):
                                if (currRow['class'] in stat):
                                    stat[currRow['class']] += 1
                                else:
                                    stat[currRow['class']] = 1
                        if (random.randint(0,100) <= 79):
                            writer = csv.DictWriter(csvfile, fieldnames=dict)
                            writer.writerow(currRow)
                            print(str(ct) + " write success")
                            ct += 1
                            if ('class' in currRow):
                                if (currRow['class'] in stat):
                                    stat[currRow['class']] += 1
                                else:
                                    stat[currRow['class']] = 1
                    elif (lable['category'] == 'pedestrian'):
                        currRow['class'] = 'pedestrian'
                        for i in range(7):
                            writer = csv.DictWriter(csvfile, fieldnames=dict)
                            writer.writerow(currRow)
                            print(str(ct) + " write success")
                            ct += 1
                            if ('class' in currRow):
                                if (currRow['class'] in stat):
                                    stat[currRow['class']] += 1
                                else:
                                    stat[currRow['class']] = 1
                        if (random.randint(0,100) <= 60):
                            writer = csv.DictWriter(csvfile, fieldnames=dict)
                            writer.writerow(currRow)
                            print(str(ct) + " write success")
                            ct += 1
                            if ('class' in currRow):
                                if (currRow['class'] in stat):
                                    stat[currRow['class']] += 1
                                else:
                                    stat[currRow['class']] = 1
                    elif (lable['category'] == 'rider'):
                        currRow['class'] = 'rider'
                        for i in range(153):
                            writer = csv.DictWriter(csvfile, fieldnames=dict)
                            writer.writerow(currRow)
                            print(str(ct) + " write success")
                            ct += 1
                            if ('class' in currRow):
                                if (currRow['class'] in stat):
                                    stat[currRow['class']] += 1
                                else:
                                    stat[currRow['class']] = 1
                        if (random.randint(0,100) <= 66):
                            writer = csv.DictWriter(csvfile, fieldnames=dict)
                            writer.writerow(currRow)
                            print(str(ct) + " write success")
                            ct += 1
                            if ('class' in currRow):
                                if (currRow['class'] in stat):
                                    stat[currRow['class']] += 1
                                else:
                                    stat[currRow['class']] = 1
                    elif (lable['category'] == 'truck'):
                        currRow['class'] = 'truck'
                        for i in range(25):
                            writer = csv.DictWriter(csvfile, fieldnames=dict)
                            writer.writerow(currRow)
                            print(str(ct) + " write success")
                            ct += 1
                            if ('class' in currRow):
                                if (currRow['class'] in stat):
                                    stat[currRow['class']] += 1
                                else:
                                    stat[currRow['class']] = 1
                        if (random.randint(0,100) <= 12):
                            writer = csv.DictWriter(csvfile, fieldnames=dict)
                            writer.writerow(currRow)
                            print(str(ct) + " write success")
                            ct += 1
                            if ('class' in currRow):
                                if (currRow['class'] in stat):
                                    stat[currRow['class']] += 1
                                else:
                                    stat[currRow['class']] = 1
                    elif (lable['category'] == 'traffic light'):
                        if (lable['attributes']['trafficLightColor'][1] == 'G'):
                            currRow['class'] = 'Go'
                            for i in range(8):
                                writer = csv.DictWriter(csvfile, fieldnames=dict)
                                writer.writerow(currRow)
                                print(str(ct) + " write success")
                                ct += 1
                                if ('class' in currRow):
                                    if (currRow['class'] in stat):
                                        stat[currRow['class']] += 1
                                    else:
                                        stat[currRow['class']] = 1
                            if (random.randint(0,100) <= 68):
                                writer = csv.DictWriter(csvfile, fieldnames=dict)
                                writer.writerow(currRow)
                                print(str(ct) + " write success")
                                ct += 1
                                if ('class' in currRow):
                                    if (currRow['class'] in stat):
                                        stat[currRow['class']] += 1
                                    else:
                                        stat[currRow['class']] = 1
                        elif (lable['attributes']['trafficLightColor'][1] == 'R'):
                            currRow['class'] = 'Stop'
                            for i in range(14):
                                writer = csv.DictWriter(csvfile, fieldnames=dict)
                                writer.writerow(currRow)
                                print(str(ct) + " write success")
                                ct += 1
                                if ('class' in currRow):
                                    if (currRow['class'] in stat):
                                        stat[currRow['class']] += 1
                                    else:
                                        stat[currRow['class']] = 1
                            if (random.randint(0,100) <= 85):
                                writer = csv.DictWriter(csvfile, fieldnames=dict)
                                writer.writerow(currRow)
                                print(str(ct) + " write success")
                                ct += 1
                                if ('class' in currRow):
                                    if (currRow['class'] in stat):
                                        stat[currRow['class']] += 1
                                    else:
                                        stat[currRow['class']] = 1
                        elif (lable['attributes']['trafficLightColor'][1] == 'Y'):
                            currRow['class'] = 'Slow Down'
                            for i in range(220):
                                writer = csv.DictWriter(csvfile, fieldnames=dict)
                                writer.writerow(currRow)
                                print(str(ct) + " write success")
                                ct += 1
                                if ('class' in currRow):
                                    if (currRow['class'] in stat):
                                        stat[currRow['class']] += 1
                                    else:
                                        stat[currRow['class']] = 1
                            if (random.randint(0,100) <= 44):
                                writer = csv.DictWriter(csvfile, fieldnames=dict)
                                writer.writerow(currRow)
                                print(str(ct) + " write success")
                                ct += 1
                                if ('class' in currRow):
                                    if (currRow['class'] in stat):
                                        stat[currRow['class']] += 1
                                    else:
                                        stat[currRow['class']] = 1
                        else:
                            currRow['class'] = 'traffic light'
                            for i in range(1):
                                writer = csv.DictWriter(csvfile, fieldnames=dict)
                                writer.writerow(currRow)
                                print(str(ct) + " write success")
                                ct += 1
                                if ('class' in currRow):
                                    if (currRow['class'] in stat):
                                        stat[currRow['class']] += 1
                                    else:
                                        stat[currRow['class']] = 1
                            if (random.randint(0,100) <= 20):
                                writer = csv.DictWriter(csvfile, fieldnames=dict)
                                writer.writerow(currRow)
                                print(str(ct) + " write success")
                                ct += 1
                                if ('class' in currRow):
                                    if (currRow['class'] in stat):
                                        stat[currRow['class']] += 1
                                    else:
                                        stat[currRow['class']] = 1
                    elif (lable['category'] == 'traffic sign'):
                        currRow['class'] = 'traffic sign'
                        for i in range(2):
                            writer = csv.DictWriter(csvfile, fieldnames=dict)
                            writer.writerow(currRow)
                            print(str(ct) + " write success")
                            ct += 1
                            if ('class' in currRow):
                                if (currRow['class'] in stat):
                                    stat[currRow['class']] += 1
                                else:
                                    stat[currRow['class']] = 1
                        if (random.randint(0,100) <= 94):
                            writer = csv.DictWriter(csvfile, fieldnames=dict)
                            writer.writerow(currRow)
                            print(str(ct) + " write success")
                            ct += 1
                            if ('class' in currRow):
                                if (currRow['class'] in stat):
                                    stat[currRow['class']] += 1
                                else:
                                    stat[currRow['class']] = 1


    print("All write success")

def saveAllTestToSeperateFile():
    # train convert json into one csv
    filePath = 'C:/Users/Administrator.WIN-2EPKD7D6018/Desktop/bdd100k-non/bdd100k/labels-20/box-track/val/'
    ct = 0
    for i, j, k in os.walk(filePath):
        # k is ####.json
        for path in k:
            json2csv(filePath + path.rsplit('.', 1)[0], "test/test_"+str(ct),"test")
            ct += 1

if __name__ == '__main__':
    dict = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    stat = {}
    org= {}
    #python main.py
    #saveAllTrainToSeperateFile()
    #saveAllTestToSeperateFile()
    #saveAllTrainToOneFile()
    #saveAllTestToOneFile()()
    saveDetectoin2020Train()
    #saveDetectoin2020Test()
    print(stat)
    print(org)
