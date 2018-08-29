#common import
import os
import logging
from logging.handlers import RotatingFileHandler
import random
import string
import colorsys
import numpy as np
import time
import cv2


#privite import
import alg_common

#deep_sort's import
from deep_sort_tracker.deep_sort import nn_matching
from deep_sort_tracker.deep_sort.detection import Detection
from deep_sort_tracker.deep_sort.tracker import Tracker
# from deep_sort_tracker.deep_sort.track import Track
from deep_sort_tracker.application_util import preprocessing
from deep_sort_tracker.application_util.image_viewer import ImageViewer
from deep_sort_tracker.tools.generate_detections import create_box_encoder

#dark_net's import
import darknet_yolo.darknet as dn


class tracker_alg():
    def __init__(self, gpu_index = 0, disable_tracker = False):
        '''tracker init function

        :param gpu_index: int
            tell tracker which gpu to use
        :param disable_tracker: bool
            if it is True, traker will be disable,
            do_frame_track will return yolo's result instaed
        '''
        self.logger = logging.getLogger('')
        self.logger.debug("========tracker_alg init=========")
        self.display = False
        self.viewer = None #to save mp4 file
        self.disable_tracker = disable_tracker

        #dark net
        self.logger.debug("start darknet init")
        dn.set_gpu(gpu_index)
        self.net = dn.load_net( alg_common.net_cfg_path.encode( "utf-8" ),
                                alg_common.net_weights_path.encode( "utf-8" ), 0 )
        self.meta = dn.load_meta( alg_common.net_data_path.encode( "utf-8" ) )
        self.logger.debug("tracker_alg init done")
        if disable_tracker:
            #if tracker is not needed, codes below is not needed
            return

        # deep_sort init
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", alg_common.max_cosine_distance, alg_common.nn_budget )
        self.tracker = Tracker( metric )
        self.feature_encoder = create_box_encoder( alg_common.feature_model_file,
                                                   batch_size=alg_common.feature_batch_size )
                                                   
        self.annotate_x = 0
        self.annotate_y = 0
        self.annotate_text = ''
        self.annotate = False
        
        self.line_a_points = None
        self.label_a = "Line_A"
        self.line_b_points = None
        self.label_b = "Line_B"

    def config_display( self, display = False, out_file = 'out.mp4', update_ms = 40,
                        display_shape=(640, 480), caption="deep_traker"):
        self.display = display
        self.update_ms = update_ms
        self.display_shape = display_shape
        self.caption = caption
        random_name = ''.join( random.sample( string.ascii_letters + string.digits, 8 ) )
        self.out_file = random_name + '_' + out_file
        if os.path.isdir('./video'):
            self.out_file = os.path.join('./video', self.out_file)
        self.logger.debug('out_file:{}'.format(self.out_file))

        self.viewer = ImageViewer(
            self.update_ms, self.display_shape, self.caption )
        self.viewer.thickness = 2
        if self.display:
            self.viewer.enable_videowriter( self.out_file )
        else:
            self.viewer.disable_videowriter()

        return alg_common.retcode_succ

    def _do_display( self, image, detections, tracks):
        if not self.display:
            return
        self.viewer.image = image.copy()

        self.viewer.thickness = 2
        self.viewer.color = 0, 0, 255
        for i, detection in enumerate( detections ):
            self.viewer.rectangle( *detection.tlwh )

        self.viewer.thickness = 2
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            self.viewer.color = self._create_unique_color( track.track_id )
            self.viewer.rectangle(
                *track.to_tlwh().astype( np.int ), label=str( track.track_id ) )
        self.logger.info( '_do_display' )
        if self.annotate:
            self.logger.info( '_do_display:annotate' )
            self.viewer.annotate(self.annotate_x, self.annotate_y, self.annotate_text)
        
        if self.line_a_points != None :
            self.viewer.line(self.line_a_points, self.label_a)
        if self.line_b_points != None :
            self.viewer.line(self.line_b_points, self.label_b)
        
        self.viewer.do_write()

        return alg_common.retcode_succ

    def _create_unique_color( self,  tag, hue_step=0.41):
        h, v = (tag * hue_step) % 1, 1. - (int( tag * hue_step ) % 4) / 5.
        r, g, b = colorsys.hsv_to_rgb( h, 1., v )
        return int( 255 * r ), int( 255 * g ), int( 255 * b )
        
    def do_frame_detect( self, img):
        '''detect object in img

        :param img: np_array
        :return: retcode, frame_tracked_object list
            frame_tracked_object like [frame_tracked_object, ...]
            for this function, all the frame_tracked_object.track_id is always -1
        '''
        self.logger.info( 'do_frame_detect normal in' )
        retcode, object_detect_list = self._dark_net_detect( img, alg_common.min_confidence,
                                                             alg_common.min_detection_height )
        if retcode != alg_common.retcode_succ:
            self.logger.error( "_object_detect failed, retcode:{}".format( retcode ) )
            return retcode, None
        if len( object_detect_list ) == 0:
            self.logger.error( "_object_detect get 0 object" )
            return alg_common.retcode_faild, None
        self.logger.debug( "_object_detect get {} objects".format( len( object_detect_list ) ) )

        retcode, bbox_list = self._dark_net_result_2_bbox_list(object_detect_list)
        if retcode != alg_common.retcode_succ:
            self.logger.error( "_dark_net_result_2_object_list failed, retcode:{}".format( retcode ) )
            return retcode, None
        if len(bbox_list) == 0:
            self.logger.error( "_dark_net_result_2_bbox_list get 0 object" )
            return alg_common.retcode_faild, None

        self.logger.info( 'do_frame_detect normal out' )
        return alg_common.retcode_succ, bbox_list
    def set_annotate(self, x, y, text, enable=False):
        self.annotate_x = x
        self.annotate_y = y
        self.annotate_text = text
        self.annotate = enable
    def set_lines(self, line_a_points, line_b_points, label_a="Line_A", label_b="Line_B"):
        self.line_a_points = line_a_points
        self.label_a = label_a
        self.line_b_points = line_b_points
        self.label_b = label_b
       
    def do_frame_track( self, img):
        '''track all the object in the img

        :param img: numpy array
            image need to track
        :return: retcode, frame_tracked_object list
            frame_tracked_object like [frame_tracked_object, ...]
        '''
        self.logger.info( 'do_frame_track normal in' )
        if self.disable_tracker:
            self.logger.error( 'do_frame_track while disable_tracker is True' )
            return alg_common.retcode_faild, None

        retcode, object_detect_list = self._dark_net_detect( img, alg_common.min_confidence,
                                                             alg_common.min_detection_height )
        if retcode != alg_common.retcode_succ :
            self.logger.error("_object_detect failed, retcode:{}".format(retcode))
            return retcode, None
        if len(object_detect_list) == 0:
            self.logger.info( "_object_detect get 0 object" )
            return alg_common.retcode_succ, []
        self.logger.debug( "_object_detect get {} objects".format(len(object_detect_list)) )

        retcode, deep_sort_detections = self._create_deep_sort_detections( img, object_detect_list )
        if retcode != alg_common.retcode_succ :
            self.logger.error("_create_detections failed, retcode:{}".format(retcode))
            return retcode, None
        self.logger.debug( "deep_sort_detections get {} objects".format(len(deep_sort_detections)) )

        boxes = np.array( [d.tlwh for d in deep_sort_detections] )
        scores = np.array( [d.confidence for d in deep_sort_detections] )
        indices = preprocessing.non_max_suppression(
            boxes, alg_common.nms_max_overlap, scores )
        detections = [deep_sort_detections[i] for i in indices]
        self.logger.debug( "detections len is {}".format(len(detections)) )

        # Update tracker.
        self.tracker.predict()
        self.tracker.update( detections )
        if self.display:
            self._do_display(img, detections, self.tracker.tracks)

        retcode, obj_list = self._deep_sort_result_2_tracked_object_list( self.tracker.tracks )
        if retcode != alg_common.retcode_succ :
            self.logger.error("_track_result_2_list failed, retcode:{}".format(retcode))
            return retcode, None
        if len(obj_list) == 0:
            self.logger.error( "_track_result_2_list get 0 object" )
            return alg_common.retcode_faild, None

        #self.logger.debug('obj_list:{}'.format(obj_list))
        self.logger.info('do_frame_track normal out')
        return alg_common.retcode_succ, obj_list

    def _dark_net_detect( self, img, min_confidence, min_detection_height, class_name ='person' ):
        '''get all the objects box in images by dark net(yoloV3)

        :param img: a image's numpy array
        :param min_confidence: Disregard all detections that lower than this value
        :param min_detection_height: Disregard all detections that lower than this value.
        :return: retcode, object_detect_result
            detect result like -[(class_name, score, (x, y, w, h))]
            example-[(b'chair', 0.98, (34.4, 29.4, 99.9, 89.1)),  ....,]
        '''
        # call dark_net&yoloV3 to detect all person in image
        random_name = ''.join( random.sample( string.ascii_letters + string.digits, 8 ) )
        random_path = os.path.join('/tmp', random_name+'.jpg')
        cv2.imwrite(random_path, img)
        result_tuple = dn.detect(self.net, self.meta, random_path.encode("utf-8"))
        os.remove(random_path)
        if len(result_tuple) == 0:
            self.logger.error('dark net detect failed:get 0 result')
            return alg_common.retcode_faild, None

        # self.logger.debug( 'dark net detect result_list:{}'.format(result_list) )
        # print('min_confidence:{}-{}, min_detection_height:{}-{}, class_name:{}-{}'.format(
        #     min_confidence, result_list[0][1], min_detection_height, result_list[0][2][3],
        #     class_name, result_list[0][0]
        # ))
        # use min_confidence/min_detection_height/class_name to filter the results
        result_tuple = [d for d in result_tuple if ((d[1] >= min_confidence) and
                                                  (d[2][3] > min_detection_height) and
                                                  (d[0] == bytes(class_name, 'utf-8')))]
        # fix offset of darknet's bbox.
        result_list = []
        for index in range(len(result_tuple)):
            name = result_tuple[index][0]
            confidence = result_tuple[index][1]
            tlwh = list(result_tuple[index][2])
            if tlwh[0] >= (tlwh[2]/2):
                tlwh[0] -= tlwh[2]/2
            if tlwh[1] >= (tlwh[3]/2):
                tlwh[1] -= tlwh[3]/2
            result_list.append([name, confidence, tlwh])
        self.logger.debug( 'after filt, dark net detect result_list:{}'.format( result_tuple ) )
        return alg_common.retcode_succ, result_list


    def _create_deep_sort_detections( self, img, object_detect_result ):
        '''convert yolo's detct result to deep_sort's format

        :param object_detect_result: detector's output result
        :return: retcode, detection_list
            detect list needed by deep_sort，it's a list of Detection(bbox, confidence, feature)
        '''
        if len(object_detect_result) == 0:
            self.logger.error('_create_detections: input a empty list')
            return alg_common.retcode_faild, None

        #get feature from deep feature_encoder
        tlwh_list = [d[2] for d in object_detect_result]
        np_detect = np.array(tlwh_list, dtype=float)
        
        np_features = self.feature_encoder( img, np_detect.copy() )
        if np_features.shape[0] != len(object_detect_result):
            self.logger.error('np_features.shape[0]-{} != len(object_detect_result)-{}'.format(
                np_features.shape[0], len( object_detect_result )
            ))
            return alg_common.retcode_faild, None

        detection_list = []
        for index, obj_detect in enumerate(object_detect_result):
            confidence_score = obj_detect[1]
            bbox = list(obj_detect[2])
            detection_list.append(Detection(bbox, confidence_score, np_features[index]))

        return alg_common.retcode_succ, detection_list

    def _deep_sort_result_2_tracked_object_list( self, track_result ):
        '''convert deep_sort's result to frame_tracked_object list

        :param track_result: deep_sort's Track list
        :return: retcode, frame_tracked_object list
            returned list like [frame_tracked_object, ...]
        '''
        if len(track_result) == 0:
            self.logger.error('_track_result_2_list: input a empty list')
            return alg_common.retcode_faild, None

        deep_track_list = []
        for track in track_result:
            deep_track = alg_common.frame_tracked_object( track.track_id,
                                                          track.to_tlwh(), track.state )
            deep_track_list.append(deep_track)

        return alg_common.retcode_succ, deep_track_list

    def _dark_net_result_2_bbox_list( self, detection_list ):
        '''convert yolo's detections to object_bbox

        :param detection_list:
            detect result like -[(class_name, score, (x, y, w, h))]
            example-[(b'chair', 0.98, (34.4, 29.4, 99.9, 89.1)),  ....,]
        :return:retcode, object_bbox
        '''
        if len(detection_list) == 0:
            self.logger.error("invalid praram, len(detection_list) == 0")
            return alg_common.retcode_faild, None

        bbox_list = []
        for det in detection_list:
            bbox_list.append(alg_common.object_bbox(det[2]))

        return alg_common.retcode_succ, bbox_list

def _set_log():
    ROOT_PATH = os.getcwd()
    LOG_PATH = os.path.join(ROOT_PATH, "log/log.txt") 

    rthandler = RotatingFileHandler(LOG_PATH, maxBytes=50 * 1024 * 1024, backupCount=2)
    # Rthandler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s')
    rthandler.setFormatter(formatter)
    logger = logging.getLogger('')
    logger.addHandler(rthandler)
    logger.setLevel(logging.DEBUG)

def pic_dir_test():
    tracker = tracker_alg()
    #tracker.config_display(True, display_shape=(1920, 1080))
    tracker.config_display( True )
    images_path = '/home/user/MOT16/test/MOT16-07/img1'

    start_time = time.time()
    for index, img in enumerate(os.listdir(images_path)):
        img_path = os.path.join(images_path, img)
        print( "index-{}, img:{}".format( index, img_path ) )
        img_bgr = cv2.imread(img_path)
        print('img_bgr.shape:{}'.format(img_bgr.shape))
        ret, detects = tracker.do_frame_track(img_bgr)
        for det in detects:
            print('track_id:{}, xywh:({},{},{},{}), state:{}'.format(det.track_id,
                                                                     det.x, det.y, det.w,
                                                                     det.h, det.status))

        # if index >= 5:
        #     break
    time_used = time.time() - start_time
    print('time_used:{}'.format(time_used))

def video_dir_test():
    tracker = tracker_alg()
    #tracker.config_display(True, display_shape=(1920, 1080))
    tracker.config_display( True )
    video_path = 'video\haha.mp4'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print( "open video file failed:{}".format( video_path ) )
        return -1

    start_time = time.time()
    index = -1
    while (cap.isOpened()):
        index += 1
        ret, frame = cap.read()
        if True != ret:
            print( "read failed for {}th frame".format( index ) )
            return -1

        print('img_bgr.shape:{}'.format(frame.shape))
        ret, detects = tracker.do_frame_track(frame)
        for det in detects:
            print('{} track_id:{}, xywh:({},{},{},{}), state:{}'.format(index, det.track_id,
                                                                     det.x, det.y, det.w,
                                                                     det.h, det.status))

    time_used = time.time() - start_time
    print('time_used:{}'.format(time_used))

def video_uarl_test():
    tracker = tracker_alg()
    interval = 6
    tracker.config_display( True , update_ms = int(1000./interval))

    cap = cv2.VideoCapture(
    'rtsp://222.190.43.243:554/010150010500940505.sdp?guid=010150010500940505&streamid=1&userid=jsqxyyt&userip=58.223.251.12&cuid=019999300259920011&cutype=1&playtype=5&ifcharge=0&time=20180412092418+08&life=3600&clienttype=1&ifpricereqsnd=0&usersessionid=0&vcdnid=&boid=001&cuorpunode=1&cuunit=&puunit=popunit00000000010&network=1&pussid=ss0000000001000000&cryptmode=0502&crypt=EC07D86E9FC5D2F1CE7E9204A66C5F4F' )

    for index in range(1000):
        ret, frame = cap.read()
        if (index % interval) != 0:
            continue

        print( 'index:{}'.format(index) )
        ret, detects = tracker.do_frame_track( frame )
        if ret == alg_common.retcode_succ:
            for det in detects:
                print( 'track_id:{}, xywh:({},{},{},{}), state:{}'.format( det.track_id,
                                                                           det.x, det.y, det.w,
                                                                           det.h, det.status ) )
def only_detector_test():
    tracker = tracker_alg(disable_tracker=True)
    # tracker.config_display(True, display_shape=(1920, 1080))
    images_path = '/home/user/MOT16/test/MOT16-07/img1'

    start_time = time.time()
    for index, img in enumerate( os.listdir( images_path ) ):
        img_path = os.path.join( images_path, img )
        print( "index-{}, img:{}".format( index, img_path ) )
        img_bgr = cv2.imread( img_path )
        print( 'img_bgr.shape:{}'.format( img_bgr.shape ) )
        ret, bbox_list = tracker.do_frame_detect( img_bgr )
        for bbox in bbox_list:
             print( 'xywh:({},{},{},{})'.format( bbox.x, bbox.y, bbox.w, bbox.h ))


        # if index >= 5:
        #     break
    time_used = time.time() - start_time
    print( 'time_used:{}'.format( time_used ) )

if __name__ == '__main__':
    _set_log()
    #pic_dir_test()
    video_dir_test()
    #only_detector_test()
    #video_uarl_test()


