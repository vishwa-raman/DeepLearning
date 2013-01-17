# DataStream.py

"""
This is a smaller version of the DataStream module containing only (part of) the 
function to read synchronized_data_streams.xml files

Example:
d = read_xml('/data01/processed_for_annotation/CESAR_May-Tue-29-18-55-16-2012/
synchronized_data_streams.xml')
print d['gaze'][0] #(video_filename, frame_number, age_of_sample)
"""

import sys, os
import unicodedata

from subprocess import call
from xml.dom.minidom import *

sector = {'Driver Window': 1, 'Left of Center': 2, 'Center': 3, 'Right of Center': 4,
          'Passenger Window':5, 'Copilot':6, 'OverRightShoulder':7, 'OverLeftShoulder':8,
          'Other': 9};
status = {'stationary-Ignore': 1, 'stationary-Parked': 2, 'stationary-AtIntersection': 3,
          'moving-AtInterSection': 4, 'moving-InCarPark': 5, 'moving-OnStreet': 6,
          'Other': 7};
intersection = {'4way': 1, 'Tjunction': 2, 'CarParkExit': 3, 'other': 4};

#sector = {'driverWindow': 1, 'leftOfCenter': 2, 'center': 3, 'rightOfCenter': 4,
#          'passengerWindow':5, 'coPilot':6, 'overRightShoulder':7, 'overLeftShoulder':8,
#          'Other': 9};
#status = {'sIgnore': 1, 'sParked': 2, 'sAtIntersection': 3,
#          'mAtInterSection': 4, 'mCarPark': 5, 'mOnStreet': 6,
#          'Other': 7};
#intersection = {'4way': 1, 'Tjunction': 2, 'carParkExit': 3, 'other': 4};

annotations = {}

def read_xml(xml_filename):
    """
    Read in a synchronized_data_streams.xml file,

    Return a dict of dicts. Return_value[stream_name][t] = 
    (video_filename, frame_number, age)
    """
    print 'reading xml DataStream'
    doc=parse(xml_filename)
    header = doc.childNodes[0].childNodes[1]
    sync_points = [n for n in doc.childNodes[0].childNodes if n.nodeName == 'sync_point']

    data_types = dict()
    paced_time_data_maps = dict()
    ages = dict()
    stream_identifiers = [n for n in header.childNodes if n.nodeName == 'stream']
    stream_names = []

    for s in stream_identifiers:
        name = s.getAttribute('sensor_name')
        data_type = s.getAttribute('data_type')
        if not data_type == 'video_file_reference:frame_number':
            continue
        stream_names.append(name)
        data_types[name] = data_type
        paced_time_data_maps[name] = dict()
        ages[name] = dict()
    print 'stream names:', stream_names

    for p in sync_points:
        t = float(p.getAttribute('timestamp'))
        for sample in [n for n in p.childNodes if n.nodeName == 'sample']:
            stream_name = sample.getAttribute('stream')
            if not stream_name in stream_names:
                continue
            data = sample.getAttribute('data')
            #print 'DATA:', data
            video_filename = data.split(':')[0]
            frame_number = int(data.split(':')[1])
            age = sample.getAttribute('age')
            paced_time_data_maps[stream_name][t] = (video_filename, frame_number, age)

    ans = dict()
    for s in stream_names:
        ans[s] = paced_time_data_maps[s]

    return ans

def find(lb, ub, lIndex, rIndex, stream, keys):
    key = keys[lIndex]
    value = stream[key]
    key = key + float(unicode.decode(value[2]))
    if (key >= lb and key <= ub):
        return lIndex
    key = keys[rIndex]
    value = stream[key]
    key = key + float(unicode.decode(value[2]))
    if (key >= lb and key <= ub):
        return rIndex
    if (lIndex == rIndex):
        return None
    index = (lIndex + rIndex) / 2
    key = keys[index]
    value = stream[key]
    key = key + float(unicode.decode(value[2]))
    if (key >= lb and key <= ub):
        return index
    if (key < lb):
        return find(lb, ub, index + 1, rIndex, stream, keys)
    return find(lb, ub, lIndex, index - 1, stream, keys)

def insertAnnotation(key, value, ann):
#    print '%f = %s'%(key, value)
    aType, aVal = ann
    f, frame, delta = value
    fileName = unicode.decode(f)
    annotationKey = '%s_%0*d'%(fileName, 4, frame)
    if annotationKey in annotations:
        annotations[annotationKey][aType] = aVal
    else:
        annotations[annotationKey] = {}
        annotations[annotationKey]['timestamp'] = key
        annotations[annotationKey]['sector'] = 9
        annotations[annotationKey]['status'] = 7
        annotations[annotationKey]['intersection'] = 4
        annotations[annotationKey][aType] = aVal

# Returns a list of frames that are included in a given time range. Since the
# data stream is sorted by timestamp. The data stream keys are timestamps that
# may or may not fall within the range passed to this function. We search through 
# the sequence of gaze data using binary search

def getIncludedFrames(lb, ub, ann, stream, keys):
    length = len(keys)
    # find the first key in stream that falls within the range requested
    index = find(lb, ub, 0, length - 1, stream, keys)
    if index is not None:
        key = keys[index]
        value = stream[key]
        key = key + float(unicode.decode(value[2]))
        insertAnnotation(key, value, ann)
        i = index
        while i > 0:
            i = i - 1
            key = keys[i]
            value = stream[key]
            key = key + float(unicode.decode(value[2]))
            if key >= lb and key <= ub:
                insertAnnotation(key, value, ann)
            else:
                break
        i = index
        while i < length - 1:
            i = i + 1
            key = keys[i]
            value = stream[key]
            key = key + float(unicode.decode(value[2]))
            if key >= lb and key <= ub:
                insertAnnotation(key, value, ann)
            else:
                break

def getAnnotation(token):
    try:
        key, value = 'sector', sector[token]
    except KeyError:
        try:
            key, value = 'status', status[token]
        except KeyError:
            key, value = 'intersection', intersection[token]
    return key, value

def read_vcode(fileName):
    f = open(fileName, 'r')
    # ignore the first four lines
    for i in range(4):
        line = f.readline()
    line = f.readline()
    data = {}
    while line:
        tokens = line.split(',')
        lb = float(tokens[0])
        rb = lb + float(tokens[1])
        try:
            data[lb] = (rb, getAnnotation(tokens[2]))
        except (KeyError):
            pass
        line = f.readline()
    return data

def blowUpVideo(fileName):
    call(["/bin/mkdir", "-p", "_temp"])
    call(["/bin/rm", "-f", "_temp/*.bmp"])
    call(["/bin/rm", "-f", "/run/shm/gaze*"])
    imageNames = '_temp/frame_%d.bmp'
    avibz2 = '/run/shm/gaze.avi.bz2'
    call(["/bin/cp", "-f", fileName + '.bz2', avibz2])
    call(["bunzip2", "-f", avibz2])
    call(["ffmpeg", "-i", "/run/shm/gaze.avi", "-sameq", imageNames])

def buildAnnotations(keys, datasetDir, prefix, outputDir):
    destFrame = 1
    currentFileName = ""
    call(["mkdir", "-p", outputDir])
    annotationsFileName = outputDir + '/annotations.xml'
    annotationsFile = open(annotationsFileName, 'w')
    annotationsFile.write('<?xml version="1.0"?>\n')
    annotationsFile.write('<annotations dir="_temp" center="320,240">\n')
    lenOfKeys = len(keys)
    for i in range(lenOfKeys):
        aKey = keys[i]
        value = annotations[aKey]
        # ignore the first 10 and last 10 frames for sector 3. The problem
        # is that we have a disproportionate number of sector 3 frames.
        # The end points of these sections of the video have frames that
        # are labelled either 2 or 4 in the annotations, which generate
        # a large number of badly labelled sector 2 and 4 frames, not to
        # mention incorrect section 3 labellings. By ignoring the first 10
        # and last 10 frames we hope to have better labelled frames
        if (value['sector'] == 3 and i < 10 and i > (lenOfKeys - 10)):
            continue;
        fileName, frameStr = aKey.split('avi_')
        sourceFrame = int(frameStr)
        fileName = datasetDir + '/' + prefix + '/' + fileName + 'avi'
        tokens = aKey.split('-')
#        number = int(tokens[-1].split('.')[0])
        if (fileName != currentFileName):
            print 'building annotations for %s'%aKey
            currentFileName = fileName
            blowUpVideo(currentFileName)
        sourceFileName = '_temp/frame_%d.bmp'%(sourceFrame + 1)
        destFileName = outputDir + ('/frame_%d.png'%destFrame)
        call(["convert", sourceFileName, destFileName])
        annotationsFile.write('  <frame>\n')
        annotationsFile.write('    <frameNumber>%d</frameNumber>\n'%destFrame)
        annotationsFile.write('    <face>0,0</face>\n')
        annotationsFile.write('    <zone>%d</zone>\n'%value['sector'])
        annotationsFile.write('    <status>%d</status>\n'%value['status'])
        annotationsFile.write('    <intersection>%d</intersection>\n'%value['intersection'])
        annotationsFile.write('  </frame>\n')
        destFrame = destFrame + 1
    annotationsFile.write('</annotations>\n')
    annotationsFile.close()

if __name__ == '__main__':
    if (len(sys.argv) < 3):
        print 'Usage: python DataStream.py <vcodeFileName> <datasetsDir> <outputDir>'
        sys.exit()

    vcodeFileName = sys.argv[1]
    datasetDir = sys.argv[2]
    outputDir = sys.argv[3]

    vcodeData = read_vcode(vcodeFileName)

    prefix = vcodeFileName[:vcodeFileName.index('-stitched')]
    summaryFileName = datasetDir + '/' + prefix + '/synchronized_data_streams.xml'

    stream = read_xml(summaryFileName)
    keys = stream['gaze'].keys()
    keys.sort()

    for key, value in sorted(vcodeData.iteritems()):
#        print ('%f = %s'%(key, value))
        getIncludedFrames(key, value[0], value[1], stream['gaze'], keys)

    keys = annotations.keys()
    keys.sort()

    buildAnnotations(keys, datasetDir, prefix, outputDir)
