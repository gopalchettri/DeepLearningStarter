# import the necessary packages
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from keras import backend as K



class LeNet:

    @staticmethod
    def gpu_grow_memory():
        from tensorflow.compat.v1.keras.backend import set_session
        from distutils.version import looseversion
        import warnings
        config = tf.compat.v1.configproto()
        config.gpu_options.allow_growth = true  # dynamically grow the memory used on the gpu
        config.log_device_placement = true  # to log device placement (on which device the operation ran)
        sess = tf.compat.v1.session(config=config)
        set_session(sess)

        physical_devices = tf.config.experimental.list_physical_devices('gpu')
        assert len(physical_devices) > 0, "not enough gpu hardware devices available"
        tf.config.experimental.set_memory_growth(physical_devices[0], true)

        if not tf.test.gpu_device_name(): 
            warnings.warn('no gpu found')
        else: 
            print('default gpu device: {}' .format(tf.test.gpu_device_name()))
    
    @staticmethod
    def build(width, height, depth, classes):
        # Initialize the model
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(20, (5, 5), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # first (and only) set of Fully Connected(FC) layer  => RELU
        # coverting the multi-dimenstional representation to 1D list using Flatten
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # applying softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model