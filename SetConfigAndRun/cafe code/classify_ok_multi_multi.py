#!/usr/bin/env python
"""
classify.py is an out-of-the-box image classifer callable from the command line.

By default it configures and runs the Caffe reference ImageNet model.
"""
import numpy as np
import pandas as pd
import os
import sys
import argparse
import glob
import time
from skimage.color import rgb2gray

import caffe


def main(argv):
    pycaffe_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "input_file",
        help="Input image, directory, or npy."
    )
    parser.add_argument(
        "output_file",
        help="Output npy filename."
    )
    # Optional arguments.
    parser.add_argument(
        "--model_def",
        default=os.path.join(pycaffe_dir,
                "python/lenet.prototxt"),
        help="Model definition file."
    )
    parser.add_argument(
        "--pretrained_model",
        default=os.path.join(pycaffe_dir,
                "python/lenet_iter_5000.caffemodel"),
        help="Trained model weights file."
    )
    parser.add_argument(
        "--gpu",
        action='store_true',
        help="Switch for gpu computation."
    )
    parser.add_argument(
        "--center_only",
        action='store_true',
        help="Switch for prediction from center crop alone instead of " +
             "averaging predictions across crops (default)."
    )
    parser.add_argument(
        "--images_dim",
        default='28,28',
        help="Canonical 'height,width' dimensions of input images."
    )
    parser.add_argument(
        "--mean_file",
        default=os.path.join(pycaffe_dir,
                             'caffe/imagenet/ilsvrc_2012_mean.npy'),
        help="Data set image mean of H x W x K dimensions (numpy array). " +
             "Set to '' for no mean subtraction."
    )
    parser.add_argument(
        "--input_scale",
        type=float,
        help="Multiply input features by this scale before input to net"
    )
    parser.add_argument(
        "--channel_swap",
        default='2,1,0',
        help="Order to permute input channels. The default converts " +
             "RGB -> BGR since BGR is the Caffe default by way of OpenCV."

    )
    parser.add_argument(
        "--ext",
        default='jpg',
        help="Image file extension to take as input when a directory " +
             "is given as the input file."
    )
    parser.add_argument(
        "--labels_file",
        default=os.path.join(pycaffe_dir,
                "mnist_words.txt"),
        help="Readable label definition file."
    )
    parser.add_argument(
        "--print_results",
        action='store_true',
        help="Write output text to stdout rather than serializing to a file."
    )
    parser.add_argument(
        "--force_grayscale",
        action='store_true',
        help="Converts RGB images down to single-channel grayscale versions," +
             "useful for single-channel networks like MNIST."
    )
    args = parser.parse_args()

    image_dims = [int(s) for s in args.images_dim.split(',')]
    if args.force_grayscale:
      channel_swap = None
      mean = None
    else:
      channel_swap = [int(s) for s in args.channel_swap.split(',')]
      mean = np.load(args.mean_file)

    # Make classifier.
    classifier = caffe.Classifier(args.model_def, args.pretrained_model,
            image_dims=image_dims,  mean=mean,
            input_scale=args.input_scale)  #, channel_swap=channel_swap

    if args.gpu:
        print 'GPU mode'
    print 'Get in while'

    with open("D:\\Skywawtch\\Working\\caffe\\python\\classify_config.txt", "r") as f:
        content = f.readlines()
    
    total_counter = 1
    sub_counter = 0
    total_file_number = int(content[0])
    sub_file_number = int(content[1])
    my_list_prob = []
    my_list_label = []
    output_directory = "D:\\Skywawtch\\Working\\caffe\\prediction\\prediction_file%s" % (content[3])
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_name = "D:\\Skywawtch\\Working\\caffe\\prediction\\prediction_file%s\\prediction_file%s.txt" % (content[3],int(content[2]))
    new_a_file = open(output_name, 'w')
    new_a_file.close()
    file = open(output_name, "r+")
    while total_counter <= total_file_number:
        my_list_prob = []
        my_list_label = []
        sub_counter = 0
        while sub_counter <= sub_file_number:
            print 'total_count: '+'%d' %total_counter
            # Load numpy array (.npy), directory glob (*.jpg), or image file.
            FilePath = args.input_file + "test (%d)_%d.jpg" % (total_counter , sub_counter)
            print FilePath
            FilePath = os.path.expanduser(FilePath)
            if FilePath.endswith('npy'):
                inputs = np.load(FilePath)
            elif os.path.isdir(FilePath):
                inputs =[caffe.io.load_image(im_f)
                         for im_f in glob.glob(FilePath + '/*.' + args.ext)]
            else:
                inputs = [caffe.io.load_image(FilePath)]

            if args.force_grayscale:
              inputs = [rgb2gray(input) for input in inputs];
              #print inputs
              inputs = np.asarray(inputs)
              inputs = np.reshape(inputs, inputs.shape +(1,))
              print inputs.shape
            print "Classifying %d inputs." % len(inputs)
            print 'Get in Classify'
            # Classify.
            start = time.time()
            scores = classifier.predict(inputs, not args.center_only).flatten()
            print "Done in %.2f s." % (time.time() - start)

            if args.print_results:
                with open(args.labels_file) as f:
                  labels_df = pd.DataFrame([
                       {
                           'synset_id': l.strip().split(' ')[0],
                           'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
                       }
                       for l in f.readlines()
                    ])
                labels = labels_df.sort('synset_id')['name'].values

                indices = (-scores).argsort()[:5]
                predictions = labels[indices]

                meta = [
                           (p, '%.5f' % scores[i])
                           for i, p in zip(indices, predictions)
                       ]

                print meta

            # Save
            #np.save(args.output_file, predictions[0])
            if float(meta[0][1]) > 0.4:
              my_list_prob.append(meta[0][1])
              my_list_label.append(meta[0][0])
            if (sub_counter!=0) and (sub_counter % sub_file_number == 0):
              if my_list_prob:
                Max_num_index = int(my_list_prob.index(max(my_list_prob)))
              else:
                my_list_prob.append(meta[0][1])
                my_list_label.append(meta[0][0])
                Max_num_index = int(my_list_prob.index(max(my_list_prob)))
              print my_list_label[Max_num_index]
              save_str = "%d\t" %total_counter
              #file.write(save_str+ str(my_list_label[Max_num_index]) +"\r\n")
              if total_counter == total_file_number:
                file.write(str(my_list_label[Max_num_index]))
              else:
                file.write(str(my_list_label[Max_num_index]) +"\n")
            sub_counter += 1
        total_counter += 1
        print total_counter
    file.close()

if __name__ == '__main__':
    main(sys.argv)
