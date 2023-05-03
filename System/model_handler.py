from configuration import config

from keras.layers import Input, Dense, Subtract, Lambda
from keras.utils import plot_model
from keras.layers.merge import Add, Average
from keras.models import Model
import keras.optimizers
from keras.constraints import maxnorm
from keras import backend as K
from keras import metrics
import tensorflow as tf
import pandas as pd
import numpy as np
import logging
import os


collinearity_power_factor = 10

class ModelHandler:
    """
    Class to handle the creation and using of the Model
    """

    def __init__(self, num_of_drugs, starting_weights_path=None):
        """
        Initialize all the models
        :param num_of_drugs: Number of drugs that in the set
        :param starting_weights_path: file with starting weights, if None, initialize them
        """
        # Set Keras backend to float64
        K.set_floatx('float64')

        # Extract the model basic hyper parameters.
        self.batch_size = config.config_map['batch_size']
        self.input_size = config.config_map['input_size']
        self.latent_dim = config.config_map['latent_dim']
        self.models_folder = config.config_map['models_folder']
        self.print_model = config.config_map['print_model']
        self.max_weight = config.config_map['maximum_weight']
        self.num_of_drugs = num_of_drugs

        if not os.path.isdir(self.models_folder):
            os.makedirs(self.models_folder)

        self._create_full_vae()

        if starting_weights_path is not None:
            self.full_model.load_weights(starting_weights_path)

    def _show_and_save_model(self, model, output_filename):
        """
        Function to plot a model
        :param model:  model to plot.
        :param output_filename: output file name to save the model1
        """
        full_summary_path = os.path.join(self.models_folder, output_filename + '.txt')
        full_image_path = os.path.join(self.models_folder, output_filename + '.png')
        with open(full_summary_path, 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        if not os.path.isdir(self.models_folder):
            os.makedirs(self.models_folder)
        try:
            plot_model(model, full_image_path, show_shapes=True)
        except AssertionError:
            logging.error('Assertion error on plot_model in model handler, full_image_path: %s', full_image_path)

    def kl_loss(self):
        """
        Function for calculate the kl-loss.
        :return: calculated KL divergence
        """
        kl_loss_value = config.config_map['KL_loss_factor'] * (-0.5 / self.input_size) * K.mean(
            1. + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
        return kl_loss_value

    def logxy_loss(self, x_input, x_output):
        """
        Function for calculate the loss of the kl-divergence between the current distributaion (z_mean and z_var),
        and the binary_crossentropy between input and output.
        :param x_input: input layer
        :param x_output: ouptut layer
        :return: calculated cross entropy
        """
        x_input = K.flatten(x_input)
        x_output = K.flatten(x_output)

        xent_loss = config.config_map['log_xy_loss_factor'] * metrics.mean_squared_error(x_input, x_output)
        return xent_loss

    def vae_loss(self, x, x_decoded_mean):
        """
        Loss calculator for the classifier.
        :param x: real label
        :param x_decoded_mean: predicted label.
        :return: calculated loss
        """
        return K.mean(self.logxy_loss(x, x_decoded_mean) + self.kl_loss())

    def classifier_loss(self, y, y_predicted):
        """
        Loss calculator for the classifier.
        :param y: real label
        :param y_predicted: predicted label.
        :return: calculated loss
        """
        # Extract the original label.
        return self.classifier_loss_selector * 0.1 * metrics.categorical_crossentropy(y, y_predicted)

    def coliniarity_straight_loss(self, y, y_predicted):
        """
        Loss calculator for straight loss - just return y_predicted.
        :param y: labels, unused.
        :param y_predicted: predicted label.
        :return: calculated loss.
        """
        return self.coliniarity_loss_selector * y_predicted

    def distance_straight_loss(self, y, y_predicted):
        """
        Loss calculator for straight loss - just return y_predicted.
        :param y: labels, unused.
        :param y_predicted: predicted label.
        :return: calculated loss.
        """
        return self.distance_loss_selector * y_predicted

    def no_selectors_straight_loss(self, y, y_predicted):
        """
        Loss calculator for straight loss - just return y_predicted.
        :param y: labels, unused.
        :param y_predicted: predicted label.
        :return: calculated loss.
        """
        return y_predicted

    def _sampling_function(self, args):
        """
        Sample from mean and deviation, to create the latent space
        :param args: tuple of the arguments (this function will implement a layer, so we can pass it only one argument)
                    the tuple contains of:
                    input layer of the mean
                    input layer of the variance
        :return: the sampling's layer values
        """
        # Create random array of shape batch_size X latent_dim, each cell contains a
        # random number that generated with normal distribution, starting with
        # mean 0 and epsilon_str standard deviation
        input_mean, input_var = args
        epsilon = K.random_normal(shape=(self.batch_size, self.latent_dim),
                                  mean=0., stddev=config.config_map['vae_sampling_std'])

        # Return that latent space values - the generated array
        return input_mean + K.exp(input_var) * epsilon

    def _create_encoder(self):
        """
        Create encoder model
        """
        # 1. Create the inputs layers
        self.x_in = Input(name='x_input', batch_shape=(self.batch_size, self.input_size))
        self.start_time_control_in = Input(name='start_time_control_in',
                                           batch_shape=(self.batch_size, self.input_size))
        self.treated_ref_in = Input(name='treated_ref_in', batch_shape=(self.batch_size, self.input_size))
        self.start_time_control_ref_in = Input(name='start_time_control_ref_in',
                                               batch_shape=(self.batch_size, self.input_size))
        self.pert_time_control_in = Input(name='pert_time_control_in',
                                          batch_shape=(self.batch_size, self.input_size))
        self.pert_time_control_ref_in = Input(name='pert_time_control_ref_in',
                                              batch_shape=(self.batch_size, self.input_size))
        self.cloud_ref_in = Input(name='cloud_ref_in',
                                              batch_shape=(self.batch_size, self.input_size))

        # 2. Create current layers
        current_layers_mean = [self.x_in,
                               self.start_time_control_in,
                               self.treated_ref_in,
                               self.start_time_control_ref_in,
                               self.pert_time_control_in,
                               self.pert_time_control_ref_in,
                               self.cloud_ref_in]
        self.other_perts_ref_inputs = []
        for i in range(self.num_of_drugs - 1):
            treated_input = Input(name='treated_pert_{}_ref_in'.format(i),
                                  batch_shape=(self.batch_size, self.input_size))
            control_input = Input(name='control_pert_{}_ref_in'.format(i),
                                  batch_shape=(self.batch_size, self.input_size))
            self.other_perts_ref_inputs.extend([treated_input, control_input])
        current_layers_mean.extend(self.other_perts_ref_inputs)

        current_layer_var = self.x_in

        layers_dimensions = config.config_map['encoder_dim'].copy()
        layers_dimensions.append(self.latent_dim)

        for i in range(len(layers_dimensions)):
            mean_dense = Dense(layers_dimensions[i], name='encoder_mean_dense_' + str(i),
                               kernel_initializer=config.config_map['initialization_method'],
                               kernel_constraint=maxnorm(self.max_weight), use_bias=config.config_map['bias'])
            var_dense = Dense(layers_dimensions[i], name='encoder_var_dense_' + str(i),
                              kernel_initializer=config.config_map['initialization_method'],
                              kernel_constraint=maxnorm(self.max_weight), use_bias=config.config_map['bias'])

            mean_activation = config.config_map['activation'](name='encoder_mean_activation_' + str(i))
            var_activation = config.config_map['activation'](name='encoder_vae_activation_' + str(i))
            for j in range(len(current_layers_mean)):
                current_layers_mean[j] = mean_activation(mean_dense(current_layers_mean[j]))
            current_layer_var = var_activation(var_dense(current_layer_var))

        # 3. Create the layer that sample from the mean and std to latent space z
        self.z_mean = current_layers_mean[0]
        self.z_log_var = current_layer_var

        self.z = Lambda(self._sampling_function, name='encoder_sampling', output_shape=(self.latent_dim,))(
            [self.z_mean, self.z_log_var])

        # Set Encoder inputs and outputs
        self.encoder_outputs = current_layers_mean

        self.encoder_inputs = [self.x_in,
                               self.start_time_control_in,
                               self.treated_ref_in,
                               self.start_time_control_ref_in,
                               self.pert_time_control_in,
                               self.pert_time_control_ref_in,
                               self.cloud_ref_in]
        self.encoder_inputs.extend(self.other_perts_ref_inputs)

        self.encoder = Model(name='Encoder', inputs=self.x_in, outputs=self.z)
        if self.print_model:
            self._show_and_save_model(self.encoder, 'Encoder')

    def _create_decoder(self):
        """
        Create the decoder model
        """
        # 1. Create intermediate layers
        # List of intermediate layers of the decoder
        layers_dimensions = config.config_map['decoder_dim'].copy()

        # Reverse the order of intermediate layers, and add them the output layer
        layers_dimensions.reverse()
        layers_dimensions.append(self.input_size)

        self.decoder_input = Input(name='decoder_input', batch_shape=(self.batch_size, self.latent_dim))

        current_layer = self.decoder_input
        for i in range(len(layers_dimensions)):
            dense = Dense(layers_dimensions[i], name='decoder_dense_' + str(i),
                          kernel_initializer=config.config_map['initialization_method'],
                          kernel_constraint=maxnorm(self.max_weight, axis=[0, 1]), use_bias=config.config_map['bias'])
            activation = config.config_map['activation'](name='decoder_activation_' + str(i))
            current_layer = activation(dense(current_layer))

        # 2. Set output layer and the model
        self.decoder_output = current_layer

        self.vae_loss_selector = Input(name='vae_loss_selector', batch_shape=(self.batch_size, 1))
        self.decoder = Model(name='Decoder', inputs=[self.decoder_input, self.vae_loss_selector],
                             outputs=[self.decoder_output])
        if self.print_model:
            self._show_and_save_model(self.decoder, 'decoder')

    @staticmethod
    def normalize_vector_layer(x):
        return tf.nn.l2_normalize(x, axis=-1)

    @staticmethod
    def _same_directions_layer(x):
        v1, v2 = x
        all_v1_zero = tf.reduce_all(tf.less(tf.abs(v1), 1e-7), axis=-1)
        all_v2_zero = tf.reduce_all(tf.less(tf.abs(v2), 1e-7), axis=-1)
        one_vectors_zero = tf.logical_or(all_v1_zero, all_v2_zero)
        dot_product = K.sum(v1 * v2, axis=-1, keepdims=True)
        loss_tensor = tf.where(one_vectors_zero, tf.zeros_like(dot_product), 1 - dot_product)
        return loss_tensor

    @staticmethod
    def _different_directions_layer(x):
        v1, v2 = x
        all_v1_zero = tf.reduce_all(tf.less(tf.abs(v1), 1e-7), axis=-1)
        all_v2_zero = tf.reduce_all(tf.less(tf.abs(v2), 1e-7), axis=-1)
        one_vectors_zero = tf.logical_or(all_v1_zero, all_v2_zero)
        dot_product = K.sum(v1 * v2, axis=-1, keepdims=True)
        dot_product = (dot_product + 1) / 2
        absolute_bias = dot_product ** collinearity_power_factor
        loss_tensor = tf.where(one_vectors_zero, tf.zeros_like(absolute_bias), absolute_bias)
        return loss_tensor

    @staticmethod
    def _create_collinearity_net_same_direction(real_end, real_start, ref_end, ref_start, label_prefix):
        """
        Create collinearity loss network, based on 4 input layers - 2 points for real vector,
        and 2 points to reference vector.
        The more parallel the less loss
        :param real_end: layer contains the end point of real vector.
        :param real_start: layer contains the start point of real vector.
        :param ref_end: layer contains the end point of reference vector.
        :param ref_start: layer contains the start point of reference vector.
        :param label_prefix: label prefix for layers names.
        :return: output layer
        """
        vectors_1_layer = Subtract(name=label_prefix + '_sample_vector')([real_end, real_start])
        vectors_2_layer = Subtract(name=label_prefix + 'reference_vector')([ref_end, ref_start])

        normalize_vector_1_layer = Lambda(ModelHandler.normalize_vector_layer,
                                          name=label_prefix + '_sample_vector_normalize',
                                          output_shape=None)(vectors_1_layer)
        normalize_vectors_2_layer = Lambda(ModelHandler.normalize_vector_layer,
                                           name=label_prefix + '_reference_vector_normalize',
                                           output_shape=None)(vectors_2_layer)

        coliniarity_out = Lambda(ModelHandler._same_directions_layer,
                                 name=label_prefix + '_coliniarity',
                                 output_shape=None)([normalize_vector_1_layer, normalize_vectors_2_layer])
        return coliniarity_out, normalize_vector_1_layer

    @staticmethod
    def _create_collinearity_net_diff_direction(real_end, real_start, ref_end, ref_start, label_prefix):
        """
        Create collinearity loss network, based on 4 input layers - 2 points for real vector,
        and 2 points to reference vector.
        The more different dirstions - the less loss
        :param real_end: layer contains the end point of real vector.
        :param real_start: layer contains the start point of real vector.
        :param ref_end: layer contains the end point of reference vector.
        :param ref_start: layer contains the start point of reference vector.
        :param label_prefix: label prefix for layers names.
        :return: output layer
        """
        vectors_1_layer = Subtract(name=label_prefix + '_sample_vector')([real_end, real_start])
        vectors_2_layer = Subtract(name=label_prefix + 'reference_vector')([ref_end, ref_start])

        normalize_vector_1_layer = Lambda(ModelHandler.normalize_vector_layer,
                                          name=label_prefix + '_sample_vector_normalize',
                                          output_shape=None)(vectors_1_layer)
        normalize_vectors_2_layer = Lambda(ModelHandler.normalize_vector_layer,
                                           name=label_prefix + '_reference_vector_normalize',
                                           output_shape=None)(vectors_2_layer)

        coliniarity_out = Lambda(ModelHandler._different_directions_layer,
                                 name=label_prefix + '_coliniarity',
                                 output_shape=None)([normalize_vector_1_layer, normalize_vectors_2_layer])
        return coliniarity_out, normalize_vector_1_layer

    def _create_coliniarity_net(self):
        """
        Create the coliniarity network
        """
        # Create the 3 coliniarity networks (pert and time, pert only, time only)
        self.pert_and_time_loss_layer, pert_and_time_vector =\
            self._create_collinearity_net_same_direction(self.encoder_outputs[0],
                                                         self.encoder_outputs[1],
                                                         self.encoder_outputs[2],
                                                         self.encoder_outputs[3],
                                                         'Pert_and_time')

        self.pert_loss_layer, pert_vector = self._create_collinearity_net_same_direction(self.encoder_outputs[0],
                                                                                         self.encoder_outputs[4],
                                                                                         self.encoder_outputs[2],
                                                                                         self.encoder_outputs[5],
                                                                                         'Pert_only')
        other_perts_loss = []
        global collinearity_power_factor

        # Parallel loss layer
        collinearity_power_factor = config.config_map['parallel_vectors_power_factor']
        self.parallel_loss_layer = Lambda(ModelHandler._different_directions_layer, name='Parallel')\
            ([pert_and_time_vector, pert_vector])

        # Other drugs layer
        collinearity_power_factor = config.config_map['other_perts_power_factor']
        for i in range(7, len(self.encoder_outputs), 2):
            loss_layer, vector = self._create_collinearity_net_diff_direction(self.encoder_outputs[0],
                                                                              self.encoder_outputs[4],
                                                                              self.encoder_outputs[i],
                                                                              self.encoder_outputs[i+1],
                                                                              'Other_perts_layer_{}'.format((i-7)/2))
            other_perts_loss.append(loss_layer)

        if len(other_perts_loss) > 1:
            self.other_pert_loss_layer = Average(name='Other_perts_collinearity')(other_perts_loss)
        elif len(other_perts_loss) == 1:
            self.other_pert_loss_layer = other_perts_loss[0]
        else:
            self.other_pert_loss_layer = None

        # Loss selectors
        self.coliniarity_loss_selector = Input(name='collinearity_loss_selector', batch_shape=(self.batch_size, 1))

        # Inputs and outputs for the collinearity model
        self.collinearity_input = self.encoder_inputs
        self.collinearity_input.append(self.coliniarity_loss_selector)

        output_layers = [self.pert_and_time_loss_layer, self.pert_loss_layer, self.parallel_loss_layer]
        if self.other_pert_loss_layer is not None:
            output_layers.append(self.other_pert_loss_layer)
        self.collinearity_model = Model(name='Collinearity', inputs=self.collinearity_input, outputs=output_layers)

        if self.print_model:
            self._show_and_save_model(self.collinearity_model, 'collinearity')

    @staticmethod
    def safe_divide(dividend, divisor):
        """
        Tensorflow safe divide:
        if divisor <> 0-> return dividend/divisor
        else: return divisor (0)
        :param dividend: tensor
        :param divisor: tensor
        :return: tensor
        """
        safe_divisor = tf.where(tf.abs(divisor) > 1e-7, divisor, tf.ones_like(divisor))
        return tf.where(tf.abs(divisor) > 1e-7, dividend / safe_divisor, tf.zeros_like(dividend))

    @staticmethod
    def safe_std(tensor):
        """
        Calculate safe std - if all values in tensor are 0, return 0, otherwise return std(tensor)
        :param tensor: tensor to calculate on it std
        :return: std(tensor)
        """
        all_zero = tf.reduce_all(tf.less(tf.abs(tensor), 1e-7), axis=-1)
        safe_tensor = tf.where(all_zero, tf.ones_like(tensor), tensor)
        return K.std(safe_tensor)

    @staticmethod
    def safe_sqrt(tensor):
        """
        Calculate safe sqrt - if all values in tensor are 0, return 0, otherwise return sqrt(tensor)
        :param tensor: tensor to calculate on it sqrt
        :return: sqrt(tensor)
        """
        all_zero = tf.reduce_all(tf.less(tf.abs(tensor), 1e-7), axis=-1)
        safe_tensor = tf.where(all_zero, tf.zeros_like(tensor), tensor)
        return K.sqrt(safe_tensor)

    @staticmethod
    def distance_between_vectors(all_layers):
        """
        Create special layer that calculate the distance between 2 vectors.
        This is based on the article :
        The Minimum Distance Between Two Lines in n-Space. by Michael Bard, Denny Himel
        :param all_layers: list of input tensors
        :return: output tensor
        """
        treated, control_start_time, treated_ref, control_start_time_ref, control_pert_time, control_pert_time_ref,\
            cloud_ref = all_layers

        # Calculate vectors
        x0 = control_start_time
        y0 = control_pert_time
        x = treated_ref - control_start_time_ref
        y = treated_ref - control_pert_time_ref

        # calculate s and t
        A = tf.reduce_sum(tf.multiply(x, x), 1, keepdims=True)
        B = 2 * (tf.reduce_sum(tf.multiply(x, x0), 1, keepdims=True) - tf.reduce_sum(tf.multiply(x, y0), 1, keepdims=True))
        C = 2 * tf.reduce_sum(tf.multiply(x, y), 1, keepdims=True)
        D = 2 * (tf.reduce_sum(tf.multiply(y, y0), 1, keepdims=True) - tf.reduce_sum(tf.multiply(y, x0), 1, keepdims=True))
        E = tf.reduce_sum(tf.multiply(y, y), 1, keepdims=True)
        s_dividend = 2 * A * D + B * C
        s_divisor = C * C - 4 * A * E
        s = ModelHandler.safe_divide(s_dividend, s_divisor)
        t = ModelHandler.safe_divide(C*s - B, 2*A)

        # Closest points
        p1 = x0 + t*x
        p2 = y0 + s*y

        # Calculate distance between points, and normal by var
        distance_vector = p2 - p1
        square_distance = tf.reduce_sum(tf.multiply(distance_vector, distance_vector), 1, keepdims=True)
        # Less than 1e-10 is floating point error, just set to zero
        square_distance = tf.where(square_distance > 1e-10, square_distance, tf.zeros_like(square_distance))
        distance = ModelHandler.safe_sqrt(square_distance)
        std = ModelHandler.safe_std(treated)
        normal_distance = ModelHandler.safe_divide(distance, std)

        # Where treated == control_pert_time, we are not in treated sample, so set normal_distance to 0
        vector_t_to_cpt = treated - control_pert_time
        distance_t_to_cpt = tf.reduce_sum(tf.multiply(vector_t_to_cpt, vector_t_to_cpt), 1, keepdims=True)
        final_distance = tf.where(distance_t_to_cpt > 1e-7, normal_distance, tf.zeros_like(normal_distance))
        return final_distance

    @staticmethod
    def distance_from_reference_cloud(all_layers):
        """
        Create special layer that calculate the distance between x_in to cloud reference, in latent space
        :param all_layers: list of input tensors
        :return: output tensor
        """
        x, _, _, _, _, _, cloud_ref = all_layers
        distance_vector = x - cloud_ref
        square_distance = tf.reduce_sum(tf.multiply(distance_vector, distance_vector), 1, keepdims=True)
        distance = ModelHandler.safe_sqrt(square_distance)

        std = ModelHandler.safe_std(x)
        return ModelHandler.safe_divide(distance, std)

    def _create_distance_net(self):
        """
        Create network that calculate a point using 2 vectors - pert and pert + time, and calculate the distance
        between that point and the treated point (x_in)
        """
        self.distance_from_calculated = Lambda(ModelHandler.distance_between_vectors,
                                       name='Distance_between_vectors')(self.encoder_outputs[0:7])
        self.distance_from_reference_point = Lambda(ModelHandler.distance_from_reference_cloud,
                                       name='Distance_from_ref_point')(self.encoder_outputs[0:7])
        self.distance_loss_selector = Input(name='distance_loss_selector', batch_shape=(self.batch_size, 1))

    def _create_labels_classifier(self):
        """
        Create the classifier for the labels
        """
        # 1. Create intermediate layers
        # List of intermediate layers of the decoder
        layers_dimensions = config.config_map['classifier_intermediate_dim']

        current_layer = self.z_mean
        for i in range(len(layers_dimensions)):
            dense = Dense(layers_dimensions[i], name='classifier_dense_' + str(i),
                          kernel_initializer=config.config_map['initialization_method'])
            current_layer = dense(current_layer)
        self.classifier_output = Dense(config.config_map['num_classes'], name='Classifier', activation='softmax',
                                       kernel_initializer=config.config_map['initialization_method'])(current_layer)

        self.classifier_loss_selector = Input(name='classifier_loss_selector', batch_shape=(self.batch_size, 1))
        self.classifier = Model(name='Classifier', inputs=[self.x_in, self.classifier_loss_selector],
                                outputs=[self.classifier_output])
        if self.print_model:
            self._show_and_save_model(self.classifier, 'Classifier')

    def _create_full_vae(self):
        """
        Create the full VAE model
        """
        self._create_encoder()
        self._create_decoder()
        self._create_coliniarity_net()
        self._create_distance_net()
        self._create_labels_classifier()

        self.encoder_out = self.encoder(self.x_in)
        self.decoder_out = self.decoder([self.encoder_out, self.vae_loss_selector])

        # Create model to calculate vae loss
        self.only_vae = Model(name='Vae', inputs=[self.x_in, self.vae_loss_selector], outputs=[self.decoder_out, self.z_mean, self.z_log_var])
        self.to_latent_space = Model(name='ToLatentSpace', inputs=self.x_in, outputs=self.z_mean)
        if self.print_model:
            self._show_and_save_model(self.only_vae, 'vae_model')

        inputs = [self.x_in, self.start_time_control_in, self.treated_ref_in, self.start_time_control_ref_in,
                  self.pert_time_control_in, self.pert_time_control_ref_in, self.cloud_ref_in]
        outputs = [self.decoder_out, self.classifier_output, self.pert_and_time_loss_layer, self.pert_loss_layer,
                   self.parallel_loss_layer, self.distance_from_calculated, self.distance_from_reference_point]
        loss_functions = [self.vae_loss, self.classifier_loss, self.coliniarity_straight_loss,
                          self.coliniarity_straight_loss, self.coliniarity_straight_loss,
                          self.coliniarity_straight_loss, self.no_selectors_straight_loss]
        loss_factors = [config.config_map['vae_loss_factor'], config.config_map['classifier_loss_factor'],
                        config.config_map['coliniarity_pert_and_time_loss_factor'],
                        config.config_map['coliniarity_pert_loss_factor'],
                        config.config_map['parallel_vectors_loss_factor'],
                        config.config_map['distance_between_vectors_loss_factor'],
                        config.config_map['distance_from_reference_loss_factor']]

        if self.other_pert_loss_layer is not None:
            inputs.extend(self.other_perts_ref_inputs)
            outputs.append(self.other_pert_loss_layer)
            loss_functions.append(self.coliniarity_straight_loss)
            loss_factors.append(config.config_map['collinearity_other_perts_loss_factor'])
        inputs.extend([self.vae_loss_selector, self.classifier_loss_selector, self.coliniarity_loss_selector,
                       self.distance_loss_selector])

        self.full_model = Model(name='Full', inputs=inputs, outputs=outputs)

        if self.print_model:
            self._show_and_save_model(self.full_model, 'full_model')

        self.optimizer = keras.optimizers.Adam(lr=config.config_map['learning_rate'], decay=config.config_map['decay'])
        self.full_model.compile(optimizer=self.optimizer,
                                loss=loss_functions,
                                loss_weights=loss_factors)

    def recompile_model(self):
        """
        Recompile the model, base on current configuration values
        """
        loss_functions = [self.vae_loss, self.classifier_loss, self.coliniarity_straight_loss,
                          self.coliniarity_straight_loss, self.coliniarity_straight_loss,
                          self.coliniarity_straight_loss, self.no_selectors_straight_loss]
        loss_factors = [config.config_map['vae_loss_factor'], config.config_map['classifier_loss_factor'],
                        config.config_map['coliniarity_pert_and_time_loss_factor'],
                        config.config_map['coliniarity_pert_loss_factor'],
                        config.config_map['parallel_vectors_loss_factor'],
                        config.config_map['distance_between_vectors_loss_factor'],
                        config.config_map['distance_from_reference_loss_factor']]
        if self.other_pert_loss_layer is not None:
            loss_functions.append(self.coliniarity_straight_loss)
            loss_factors.append(config.config_map['collinearity_other_perts_loss_factor'])
        self.full_model.compile(optimizer=self.optimizer, loss=loss_functions, loss_weights=loss_factors)

    @staticmethod
    def divide_round_up(n, d):
        """
        Divide n by d, and round up the quotient
        :param n: the dividend
        :param d: the divisor
        :return: the quotient rounded up
        """
        return np.int((n + (d - 1)) / d)

    def predict_latent_space(self, samples_df):
        """
        Predict the latent space for DataFrame of samples.
        :param samples_df: DataFrame of the samples to predict
        :return: DataFrame with predicted latent space, with the same indexes as recieved
        """
        number_of_samples = samples_df.shape[0]
        number_of_batches = self.divide_round_up(number_of_samples, self.batch_size)
        samples_to_predict_np = np.zeros(shape=[
            number_of_batches * self.batch_size, samples_df.shape[1]])
        samples_to_predict_np[:number_of_samples, :] = samples_df.values

        predicted_np = self.to_latent_space.predict(samples_to_predict_np,
                                            batch_size=self.batch_size)
        latent_space_df = pd.DataFrame(predicted_np[:number_of_samples, :],
                                       index=samples_df.index, columns=range(self.latent_dim))
        return latent_space_df

    def predict_variational_loss(self, samples_df):
        """
        Predict x_output from x_in and y labels
        :param samples_df: data to predict
        :return: predicted values
        """
        number_of_samples = samples_df.shape[0]
        number_of_batches = self.divide_round_up(number_of_samples, self.batch_size)
        samples_to_predict_np = np.zeros(shape=[
            number_of_batches * self.batch_size, samples_df.shape[1]])
        samples_to_predict_np[:number_of_samples, :] = samples_df.values

        # Use mock selectors
        loss_selectors_np = np.zeros(shape=[number_of_batches * self.batch_size, 1])
        predicted_np, z_mean_np, z_log_var_np = self.only_vae.predict([samples_to_predict_np, loss_selectors_np],
                                                                      batch_size=self.batch_size)
        x_output_df = pd.DataFrame(predicted_np[:number_of_samples, :],
                                   index=samples_df.index, columns=samples_df.columns)
        z_mean_df = pd.DataFrame(z_mean_np[:number_of_samples, :], index=samples_df.index)
        z_log_var_df = pd.DataFrame(z_log_var_np[:number_of_samples, :], index=samples_df.index)

        return x_output_df, z_mean_df, z_log_var_df

    def predict_coliniarity_loss(self, samples_df, reference_points):
        """
        Predict the loss of each point in samples_np, based on the mean of untreated and the reference points
        :param samples_df: samples to predict their loss
        :param reference_points: tuple of 5 np arrays, they are the reference points for the samples
        :return: numpy array of the calculated losses
        """
        number_of_samples = samples_df.shape[0]
        number_of_batches = self.divide_round_up(number_of_samples, self.batch_size)

        rounded_up_np = list()
        rounded_up_np.append(np.zeros(shape=[number_of_batches * self.batch_size, samples_df.shape[1]]))
        rounded_up_np[0][:number_of_samples, :] = samples_df
        for i in range(len(reference_points)):
            rounded_up_np.append(np.zeros(shape=[number_of_batches * self.batch_size, samples_df.shape[1]]))
            rounded_up_np[i+1][:number_of_samples, :] = reference_points[i]

        # Create mock loss selectors, and append it 4 times, for each required selectors
        loss_selectors_np = np.zeros(shape=[number_of_batches * self.batch_size, 1])
        rounded_up_np.append(loss_selectors_np)
        rounded_up_np.append(loss_selectors_np)
        rounded_up_np.append(loss_selectors_np)
        rounded_up_np.append(loss_selectors_np)

        output = self.full_model.predict(rounded_up_np, batch_size=self.batch_size)
        pert_and_time_loss, pert_only_loss = output[2:4]
        collinearity_loss_np = pert_and_time_loss + pert_only_loss
        loss_df = pd.DataFrame(collinearity_loss_np[:number_of_samples, :],
                               index=samples_df.index,
                               columns=['loss'])
        return loss_df

    def predict_decoder(self, samples_df):
        """
        Predict x_output based on latent space only
        :param samples_df: DataFrame of values in latent space
        :return: predicted samples in real space
        """
        number_of_samples = samples_df.shape[0]
        number_of_batches = self.divide_round_up(number_of_samples, self.batch_size)
        samples_to_predict_np = np.zeros(shape=[
            number_of_batches * self.batch_size, samples_df.shape[1]])
        samples_to_predict_np[:number_of_samples, :] = samples_df.values

        # Create mock loss selector
        loss_selectors_np = np.zeros(shape=[number_of_batches * self.batch_size, 1])
        predicted_np = self.decoder.predict([samples_to_predict_np, loss_selectors_np], batch_size=self.batch_size)
        x_output_df = pd.DataFrame(predicted_np[:number_of_samples, :],
                                   index=samples_df.index, columns=range(self.input_size))
        return x_output_df

    def predict_classifier(self, samples_df):
        """
        Predict classifier, from x_in to classifier_output
        :param samples_df: samples to predict
        :return: DataFrame with classifier prediction to each samples
        """
        samples_np = samples_df.values
        number_of_samples = samples_np.shape[0]
        number_of_batches = self.divide_round_up(number_of_samples, self.batch_size)
        samples_to_predict_np = np.zeros(shape=[
            number_of_batches * self.batch_size, samples_np.shape[1]])
        samples_to_predict_np[:number_of_samples, :] = samples_np

        # Create mock loss selector
        loss_selectors_np = np.zeros(shape=[number_of_batches * self.batch_size, 1])
        classifier_predictions_np = self.classifier.predict([samples_to_predict_np, loss_selectors_np],
                                                            batch_size=self.batch_size)
        classifier_predictions_df = pd.DataFrame(classifier_predictions_np[:number_of_samples, :],
                                                 index=samples_df.index)
        return classifier_predictions_df

    def get_correctly_classified(self, data_df, info_df):
        """
        Set correctly_classified column in info_df.
        :param data_df: DataFrame with values
        :param info_df: DataFrame with info, the correct classification is in 'numeric_labels' column.
        """
        # Predict classifier and find where the samples it wrong at
        classifier_choices_df = self.predict_classifier(data_df)
        classifier_max_index_np = np.argmax(classifier_choices_df.values, axis=1)
        classifier_df = pd.DataFrame(index=classifier_choices_df.index, columns=['real', 'predicted'])

        classifier_df['real'] = info_df['numeric_labels']
        classifier_df['predicted'] = classifier_max_index_np
        info_df['correctly_classified'] = 'Failed'
        correct_indexes = classifier_df[classifier_df['real'] == classifier_df['predicted']].index
        info_df.loc[correct_indexes, 'correctly_classified'] = 'Correct'

    def save_network_weights(self, output_path=None):
        """
        Save model weights to file
        :param output_path: path to save the weights, if None - save to output model folder
        """
        if output_path is None:
            output_path = os.path.join(self.models_folder, 'model.h5')
        self.full_model.save_weights(output_path)
