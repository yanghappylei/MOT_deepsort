
#param for dark net
net_cfg_path = "darknet_yolo/weights/yolov3.cfg"
net_weights_path = "darknet_yolo\weights\yolov3.weights"
net_data_path = "darknet_yolo\weights\coco.data"
darknet_so_path = "darknet_yolo\libdarknet.so"

#param for deep_sort
# TODO：这些超参的具体含义还没研究透，需要看论文后结合项目修改
min_confidence = 0.5
nms_max_overlap = 1.0
min_detection_height = 10
max_cosine_distance = 0.2
nn_budget = None
feature_model_file = 'deep_sort_tracker\networks\mars-small128.pb'
feature_batch_size = 32
# TODO:使用入参修改这个比例
per_process_gpu_memory_fraction = 0.05


# return code
retcode_succ = 0
retcode_faild = -1

# frame

class track_state:
    Tentative = 1
    Confirmed = 2
    Deleted = 3
class object_bbox():
    x = 0.0
    y = 0.0
    w = 0.0
    h = 0.0
    def __init__(self, bbox):
        self.x = bbox[0]
        self.y = bbox[1]
        self.w = bbox[2]
        self.h = bbox[3]

class frame_tracked_object( object_bbox ):
    track_id = -1
    status = track_state.Deleted
    def __init__(self, track_id, bbox, status):
        object_bbox.__init__( self, bbox )
        self.track_id = track_id
        self.status = status
    def to_str(self):
        return "track_id-{}, status-{}, bbbox-{},{},{},{}".format(self.track_id,self.status, self.x, self.y, self.w, self.h)

class object_point():
    x = 0.0
    y = 0.0
    def __init__( self, x = 0.0, y = 0.0 ):
        self.x = x
        self.y = y
    def is_new_point( self ):
        if 0.0 == self.x and 0.0 == self.y:
            return True
        else:
            return False
    def set( self, x, y ):
        self.x = x
        self.y = y
# in default, every camera has 4 channel
max_channels_num=4

# 计数层和接口层通信的队列长度
max_msg_queue_size=5

# 更新计数最多连续失败的次数，超过则退出
max_update_failed_num = 5

# 保活信号提前的秒数
alive_update_ahead_sec = 15