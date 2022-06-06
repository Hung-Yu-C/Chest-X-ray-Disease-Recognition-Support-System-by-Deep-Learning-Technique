import numpy as np
import os
from configparser import ConfigParser
#from generator import AugmentedImageSequence
#from models.keras import ModelFactory
from sklearn.metrics import roc_auc_score
#from utility import get_sample_counts
import tensorflow as tf

# utility.py ------start------

import pandas as pd

'''#-------------------------------------------------------------------
def solve_cudnn_error():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
'''#-------------------------------------------------------------------
def get_sample_counts(output_dir, dataset, class_names):
		"""
		Get total and class-wise positive sample count of a dataset
		Arguments:
		output_dir - str, folder of dataset.csv
		dataset - str, train|dev|test
		class_names - list of str, target classes
		Returns:
		total_count - int
		class_positive_counts - dict of int, ex: {"Effusion": 300, "Infiltration": 500 ...}
		"""
		df = pd.read_csv(os.path.join(output_dir, '{}.csv'.format( dataset )))
		total_count = df.shape[0]
		labels = df[class_names].values
		positive_counts = np.sum(labels, axis=0)
		class_positive_counts = dict(zip(class_names, positive_counts))
		return total_count, class_positive_counts
# utility.py ------end------

# /models/keras.py ------start------
import importlib
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers.core import Dense
from tensorflow.keras.models import Model


class ModelFactory:
		"""
		Model facotry for Keras default models
		"""

		def __init__(self):
				self.models_ = dict(
						VGG16=dict(
								input_shape=(224, 224, 3),
								module_name="vgg16",
								last_conv_layer="block5_conv3",
						),
						VGG19=dict(
								input_shape=(224, 224, 3),
								module_name="vgg19",
								last_conv_layer="block5_conv4",
						),
						DenseNet121=dict(
								input_shape=(224, 224, 3),
								module_name="densenet",
								last_conv_layer="bn",
						),
						ResNet50=dict(
								input_shape=(224, 224, 3),
								module_name="resnet50",
								last_conv_layer="activation_49",
						),
						InceptionV3=dict(
								input_shape=(299, 299, 3),
								module_name="inception_v3",
								last_conv_layer="mixed10",
						),
						InceptionResNetV2=dict(
								input_shape=(299, 299, 3),
								module_name="inception_resnet_v2",
								last_conv_layer="conv_7b_ac",
						),
						NASNetMobile=dict(
								input_shape=(224, 224, 3),
								module_name="nasnet",
								last_conv_layer="activation_188",
						),
						NASNetLarge=dict(
								input_shape=(331, 331, 3),
								module_name="nasnet",
								last_conv_layer="activation_260",
						),
				)

		def get_last_conv_layer(self, model_name):
				return self.models_[model_name]["last_conv_layer"]

		def get_input_size(self, model_name):
				return self.models_[model_name]["input_shape"][:2]

		def get_model(self, class_names, model_name="DenseNet121", use_base_weights=True,
									weights_path=None, input_shape=None):

				if use_base_weights is True:
						base_weights = "imagenet"
				else:
						base_weights = None

				base_model_class = getattr(
						importlib.import_module(
								# f"keras.applications.{self.models_[model_name]['module_name']}"
								"tensorflow.keras.applications." + self.models_[model_name]["module_name"]  # try
						),
						model_name)

				if input_shape is None:
						input_shape = self.models_[model_name]["input_shape"]

				img_input = Input(shape=input_shape)

				base_model = base_model_class(
						include_top=False,
						input_tensor=img_input,
						input_shape=input_shape,
						weights=base_weights,
						pooling="avg")


				x = base_model.output
				temp = Dense(512, activation="sigmoid", name="temp")(x)
				predictions = Dense(len(class_names), activation="sigmoid", name="predictions")(temp)
				model = Model(inputs=img_input, outputs=predictions)

				if weights_path == "":
						weights_path = None

				if weights_path is not None:
						# print(f"load model weights_path: {weights_path}")
						print( 'load model weights_path: {}'.format( weights_path ) ) # try
						
						model.load_weights(weights_path,by_name=True)

				model.trainable = False
				model.summary()

				return model

# /models/keras.py ------end------



# generate.py ------start------
import pandas as pd
from tensorflow.keras.utils import Sequence
from PIL import Image
from skimage.transform import resize


class AugmentedImageSequence(Sequence):
		"""
		Thread-safe image generator with imgaug support
		For more information of imgaug see: https://github.com/aleju/imgaug
		"""

		def __init__(self, dataset_csv_file, class_names, source_image_dir, batch_size=2,
								 target_size=(224, 224), augmenter=None, verbose=0, steps=None,
								 shuffle_on_epoch_end=True, random_state=1):
				"""
				:param dataset_csv_file: str, path of dataset csv file
				:param class_names: list of str
				:param batch_size: int
				:param target_size: tuple(int, int)
				:param augmenter: imgaug object. Do not specify resize in augmenter.
													It will be done automatically according to input_shape of the model.
				:param verbose: int
				"""
				self.dataset_df = pd.read_csv(dataset_csv_file)
				self.source_image_dir = source_image_dir
				self.batch_size = batch_size
				self.target_size = target_size
				self.augmenter = augmenter
				self.verbose = verbose
				self.shuffle = shuffle_on_epoch_end
				self.random_state = random_state
				self.class_names = class_names
				self.prepare_dataset()
				if steps is None:
						self.steps = int(np.ceil(len(self.x_path) / float(self.batch_size)))
				else:
						self.steps = int(steps)

		def __bool__(self):
				return True

		def __len__(self):
				return self.steps

		def __getitem__(self, idx):
				batch_x_path = self.x_path[idx * self.batch_size:(idx + 1) * self.batch_size]
				batch_x = np.asarray([self.load_image(x_path) for x_path in batch_x_path])
				batch_x = self.transform_batch_images(batch_x)
				batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
				return batch_x, batch_y

		def load_image(self, image_file):
				image_path = os.path.join(self.source_image_dir, image_file)
				image = Image.open(image_path)
				image_array = np.asarray(image.convert("RGB"))
				image_array = image_array / 255.
				image_array = resize(image_array, self.target_size)
				return image_array

		def transform_batch_images(self, batch_x):
				batch_x = batch_x.astype(np.float32)

				if self.augmenter is not None:
						batch_x = self.augmenter.augment_images(batch_x)
				batch_x = batch_x.astype(np.float64)

				imagenet_mean = np.array([0.485, 0.456, 0.406])
				imagenet_std = np.array([0.229, 0.224, 0.225])
				batch_x = (batch_x - imagenet_mean) / imagenet_std
				return batch_x

		def get_y_true(self):
				"""
				Use this function to get y_true for predict_generator
				In order to get correct y, you have to set shuffle_on_epoch_end=False.
				"""
				if self.shuffle:
						raise ValueError("""
						You're trying run get_y_true() when generator option 'shuffle_on_epoch_end' is True.
						""")
				print( "num: " + str(self.steps*self.batch_size) )
				return self.y[:self.steps*self.batch_size, :]
		def get_Image_Index(self) :
				return self.x_path[:self.steps*self.batch_size]

		

		def prepare_dataset(self):
				df = self.dataset_df.sample(frac=1., random_state=self.random_state)
				self.x_path, self.y = df["Image Index"].values, df[self.class_names].values

		def on_epoch_end(self):
				if self.shuffle:
						self.random_state += 1
						self.prepare_dataset()
# generate.py ------end------
import csv

def getThres(prediction):
    total_thres = []
    for labelIndex in range (0, 14):
        sum = 0.0
        for index in range (0, len(prediction)):
            sum = sum + prediction[index][labelIndex]
        total_thres.append(sum/len(prediction))
    return total_thres

def findResult( predict, className, thres ):
    # print("Class: {}".format(className))
    print("Threshold={}".format(thres))
    result = []
    for i in range (0, len(predict)):
        comparison = []
        comparisonStrList = []
        print("Image {} potential disease:".format(i))
        for j in range (0, 14):
            # print("predict={}".format(predict[i]))					
            if ( predict[i][j] > thres[j] ):
                comparison.append(predict[i][j]-thres[j])
                comparisonStrList.append(className[j])
                print(className[j])
            else:
                comparison.append(0)
            result.append(' '.join(comparisonStrList))
    return result

def giveResultPerPatient(imageIndex, predict, className, thres):
    result = findResult(predict, className, thres)
    with open('./result/resultPerPatient_14.csv', 'w', newline='') as csvfile:
        # 建立 CSV 檔寫入器
        writer = csv.writer(csvfile)
        writer.writerow(['Image Index', 'Predict label'])
        for index in range (0, len(imageIndex)):
            writer.writerow([imageIndex[index], result[index]])

import keras
import time
def main():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
    #solve_cudnn_error()
    # parser config
    config_file = "./14/sample_config.ini"
    cp = ConfigParser()
    cp.read(config_file)

    # default config
    output_dir =  "./14/output_2020-09-01"
    base_model_name = cp["DEFAULT"].get("base_model_name")
    class_names = cp["DEFAULT"].get("class_names").split(",")
    image_source_dir = cp["DEFAULT"].get("image_source_dir")

    # train config
    image_dimension = cp["TRAIN"].getint("image_dimension")

    # test config
    batch_size = cp["TEST"].getint("batch_size")
    test_steps = cp["TEST"].get("test_steps")
    use_best_weights = cp["TEST"].getboolean("use_best_weights")

    # parse weights file path
    output_weights_name = cp["TRAIN"].get("output_weights_name")
    weights_path = os.path.join(output_dir, output_weights_name)
    best_weights_path = os.path.join(output_dir, 'best_{}'.format(output_weights_name) )

    # get test sample count
    test_counts, _ = get_sample_counts(output_dir, "test", class_names)

    # compute steps
    if test_steps == "auto":
        test_steps = int(test_counts / batch_size)
    else:
        try:
            test_steps = int(test_steps)
        except ValueError:
            raise ValueError("""
                test_steps: {} is invalid,
                please use 'auto' or integer.
                """.format(test_steps))
    print('** test_steps: {} **'.format(test_steps))

    print("** load model **")
    if use_best_weights:
        print("** use best weights **")
        model_weights_path = best_weights_path
    else:
        print("** use last weights **")
        model_weights_path = weights_path
    model_factory = ModelFactory()
    model = model_factory.get_model(
        class_names,
        model_name=base_model_name,
        use_base_weights=False,
        weights_path=model_weights_path)

    print("** load test generator **")
    test_sequence = AugmentedImageSequence(
        dataset_csv_file=os.path.join(output_dir, "test.csv"),
        class_names=class_names,
        source_image_dir=image_source_dir,
        batch_size=batch_size,
        target_size=(image_dimension, image_dimension),
        augmenter=None,
        steps=test_steps,
        shuffle_on_epoch_end=False,
    )

    print("** make prediction **")
    if not os.path.isdir("./result"):
        os.mkdir("./result")

    start_time = time.time()
    y_hat = model.predict_generator(test_sequence, verbose=1)
    y = test_sequence.get_y_true()
    image_index = test_sequence.get_Image_Index()
    end_time = time.time()
    print( "image index: " + str(len(image_index)) )
    print( "y_hat: " + str(len(y_hat)) )
    outCSV = pd.DataFrame( columns = class_names, index = image_index, data = y_hat )
    
    newCSVfile = open( './result/14_result_value.csv', 'w', newline = '\n' )
    outCSV.to_csv( newCSVfile, encoding='gbk' )
    print( outCSV )

    threshold = getThres(y_hat)
    giveResultPerPatient(image_index, y_hat, class_names, threshold)
    '''
    for i in range(len(image_index)):
        print( str(image_index[i]) + ": " + str(y_hat[i]) )
    
    
    for i in range(len(image_index)):
        image_index[i].append(y_hat[i])
    for i in range(len(image_index)):
        print( str(image_index[i]) )
    '''
    '''
    for i in range(len(y_hat)):
        print(test_sequence.x_path[i] + "\n\t Normal:" + str(y_hat[i][0]) + "\n\t Covid_19:" + str(y_hat[i][1]) )
        if y_hat[i][0] > y_hat[i][1] :
            print( "I think picture( " + test_sequence.x_path[i] + " ) is Normal.\n" )
        else :
            print( "I think picture( " + test_sequence.x_path[i] + " ) is Covid_19.\n" )
    '''
    '''
    test_log_path = os.path.join(output_dir, "test.log")
    print("** write log to {} **".format(test_log_path))
    aurocs = []
    with open(test_log_path, "w") as f:
        for i in range(len(class_names)):
            try:
                score = roc_auc_score(y[:, i], y_hat[:, i])
                aurocs.append(score)
            except ValueError:
                score = 0
            f.write("{}: {}\n".format(class_names[i],score))
        mean_auroc = np.mean(aurocs)
        f.write("-------------------------\n")
        f.write("mean auroc: {}\n".format(mean_auroc))
        print("mean auroc: {}".format(mean_auroc))
    '''
    print("test_14 run time: {}s".format(end_time-start_time))


if __name__ == "__main__":
    main()
