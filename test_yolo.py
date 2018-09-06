import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import time

def detect_img(yolo):
    '''加载模型后，持续检测输入的图像
    @paras:yolo是配置好的YOLO类的实例
    '''
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            tic=time.time()
            r_image = yolo.detect_image(image)
            toc=time.time()
            print("cost time:",toc-tic)
            r_image.show()
    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument('--model', type=str, default='logs/004/trained_weights_stage_1.h5',
                        help='path to model weight file, default ' + YOLO.get_defaults("model_path"))
    parser.add_argument('--anchors', type=str,
                        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path"))
    parser.add_argument('--classes', type=str, default='model_data/nwpu_classes.txt',
                        help='path to class definitions, default ' + YOLO.get_defaults("classes_path"))
    parser.add_argument('--gpu_num', type=int, default=1,
                        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num")))
    parser.add_argument('--image', default=True, action="store_true",
                        help='Image detection mode, will ignore all positional arguments')
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument("--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path")
    parser.add_argument("--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path")

    FLAGS = parser.parse_args()

    print("debug:",FLAGS.image)
    if FLAGS.image: #如果image设置为True,则检测图像
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))

    elif "input" in FLAGS: #检测视频
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")