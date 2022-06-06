import tensorflow as tf
import tensorflow.contrib.tensorrt as trt # must import this although we will not use it explicitly
from tensorflow.python.platform import gfile
from PIL import Image
import numpy as np
import time
from skimage.transform import resize
import pandas as pd

image_list = []
dataset_df = pd.read_csv("/media/ACE43D43E43D1156/test.csv")
x_path = ( dataset_df.sample(frac=1., random_state = 1 ))["Image Index"].as_matrix()
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])
for i in x_path :
    img1= Image.open( "/media/ACE43D43E43D1156/image_test/" + i )
    img1 = np.asarray(img1.convert("RGB") )
    img1 = img1 / 255
    input_img = np.concatenate([resize(img1, [224, 224, 3]).reshape( (1,224,224,3) )], axis=0)
    input_img = ( input_img - imagenet_mean ) / imagenet_std
    image_list.append(input_img)

'''
img1= Image.open("./test/covid_173.png")
img1 = np.asarray(img1)
input_img = np.concatenate([resize(img1, [224, 224, 3]).reshape( (1,224,224,3) )], axis=0)
'''

def read_plan_graph(model):
  with gfile.FastGFile(model,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def

TENSORRT_MODEL_PATH = './covidtensorRT/TensorRT_model.pb'

graph = tf.Graph()
with graph.as_default():
    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.50))) as sess:
        # read TensorRT model
        trt_graph = read_plan_graph(TENSORRT_MODEL_PATH)

        # obtain the corresponding input-output tensor
        tf.import_graph_def(trt_graph, name='')
        input = sess.graph.get_tensor_by_name('input_1:0')
        output = sess.graph.get_tensor_by_name('covid_19/Sigmoid:0')

        # in this case, it demonstrates to perform inference for 50 times
        total_time = 0; n_time_inference = len(x_path)
        out_pred = sess.run(output, feed_dict={input: image_list[0]})

        correct = 0
        for i in range(n_time_inference):
            t1 = time.time()
            out_pred = sess.run(output, feed_dict={input: image_list[i]})
            t2 = time.time()
            delta_time = t2 - t1
            total_time += delta_time
            if out_pred[0][0] <= out_pred[0][1] and x_path[i][0] == 'c' : correct = correct + 1
            if out_pred[0][0] > out_pred[0][1] and x_path[i][0] != 'c' : correct = correct + 1

            print("inference-" + str(i)  + " " + x_path[i] + " result: ", out_pred )
            print("needed time in inference-" + str(i) + ": ", delta_time)

        avg_time_tensorRT = total_time / n_time_inference
        print("average inference time: ", avg_time_tensorRT)
        print("ACC : " + str(correct/397) )


