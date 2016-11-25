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
                "python/lenet_iter_10000.caffemodel"),
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

    # Load numpy array (.npy), directory glob (*.jpg), or image file.
    args.input_file = os.path.expanduser(args.input_file)
    if args.input_file.endswith('npy'):
        inputs = np.load(args.input_file)
    elif os.path.isdir(args.input_file):
        inputs =[caffe.io.load_image(im_f)
                 for im_f in glob.glob(args.input_file + '/*.' + args.ext)]
    else:
        inputs = [caffe.io.load_image(args.input_file)]

    if args.force_grayscale:
      inputs = [rgb2gray(input) for input in inputs];
      #print inputs
      inputs = np.asarray(inputs)
      inputs = np.reshape(inputs, inputs.shape +(1,))
      print inputs.shape
    print "Classifying %d inputs." % len(inputs)

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
    file = open("prediction/prediction_file.txt", "w")
    print meta[0][1]
    file.write(predictions[0] +" " + meta[0][1])
    file.close()

if __name__ == '__main__':
    main(sys.argv)
