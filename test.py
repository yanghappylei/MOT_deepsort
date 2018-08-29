import cv2
import time
import os

'''
cap = cv2.VideoCapture('http://10.11.20.104:8554/jsdx')

while (cap.isOpened()):
    #start_time = time.time()
    ret, frame = cap.read()
    if ret == True:
        cv2.imshow( 'iframe', frame )

    key_ret = cv2.waitKey( 30 )
    #print( "key_ret = {}, time used:{} sec".format( key_ret, round( end_time - start_time, 3 ) ) )
    if (key_ret == 0):  # if delete key is pressed
        break
'''
import os
import argparse

parser = argparse.ArgumentParser()


parser.add_argument("--enable_debug", default=0,
                    help="if 1, enable debug mode, else, debug function is disabled",
                    type=int)
args = parser.parse_args()
os.environ["enable_debug"] = str(args.enable_debug)

os.environ["test"] = '-1'
os.putenv('test123', '123')

print("{}".format(os.getenv('enable_debug')))

