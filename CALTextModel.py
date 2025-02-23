import tensorflow as tf
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.layers import Lambda
import numpy as np
import math
import utility

'''
DenseEncoder class implements dense encoder 
using bottleneck layers. Encoder contains three dense blocks. 
'''
class DenseEncoder(tf.keras.layers.Layer):
    def __init__(self, blocks,       # number of dense blocks
                 level,                     # number of levels in each blocks
                 growth_rate,               # growth rate in DenseNet paper: k
                 training,
                 dropout_rate=0.2,          # keep-rate of dropout layer
                 dense_channels=0,          # filter numbers of transition layer's input
                 transition=0.5,            # rate of compression
                 input_conv_filters=48,     # filter numbers of conv2d before dense blocks
                 input_conv_stride=2,       # stride of conv2d before dense blocks
                 input_conv_kernel=[7,7]):  # kernel size of conv2d before dense blocks
        super(DenseEncoder, self).__init__() #Initialize Keras Layer
        self.blocks = blocks
        self.growth_rate = growth_rate
        self.training = training
        self.dense_channels = dense_channels
        self.level = level
        self.dropout_rate = dropout_rate
        self.transition = transition
        self.input_conv_kernel = input_conv_kernel
        self.input_conv_stride = input_conv_stride
        self.input_conv_filters = input_conv_filters
        self.output_dim = self.estimate_output_dim()
    
    def estimate_output_dim(self):
        channels = self.input_conv_filters  # Initial conv layer filters
        for i in range(self.blocks):
            channels += self.level * self.growth_rate  # Growth inside block
            if i < self.blocks - 1:
                channels = int(channels * self.transition)  # Compression after block
        return channels

    def bound(self, nin, nout, kernel):
        fin = nin * kernel[0] * kernel[1]
        fout = nout * kernel[0] * kernel[1]
        return np.sqrt(6. / (fin + fout))
    
    def call(self, input_x, mask_x, training=False):
        self.dense_channels = 0
        
        if len(input_x.shape) == 3:
            input_x = tf.expand_dims(input=input_x, axis=3)
        
        x = input_x
    
        limit = self.bound(1, self.input_conv_filters, self.input_conv_kernel)
        x = tf.keras.layers.Conv2D(
            filters=self.input_conv_filters, 
            strides=self.input_conv_stride,
            kernel_size=self.input_conv_kernel, 
            padding='SAME', 
            use_bias=False,
            kernel_initializer=tf.keras.initializers.RandomUniform(-limit, limit)
        )(x)
    
        mask_x = mask_x[:, 0::2, 0::2]
        
        x = tf.keras.layers.LayerNormalization()(x)  # Replace BN with LN
        '''
        bn_layer = tf.keras.layers.BatchNormalization(
            momentum=0.9, scale=True,
            gamma_initializer=tf.keras.initializers.RandomUniform(
                -1.0 / math.sqrt(self.input_conv_filters),    
                1.0 / math.sqrt(self.input_conv_filters)
            )
        )
        x = bn_layer(x, training=training)  # <-- Pass training dynamically
        '''
    
        x = tf.nn.relu(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='SAME')(x)
        
        input_pre = x
        mask_x = mask_x[:, 0::2, 0::2]
        self.dense_channels += self.input_conv_filters
        dense_out = x
        
        for i in range(self.blocks):
            for j in range(self.level):
                limit = self.bound(self.dense_channels, 4 * self.growth_rate, [1, 1])
                x = tf.keras.layers.Conv2D(
                    filters=4 * self.growth_rate, 
                    kernel_size=[1, 1], 
                    strides=1, 
                    padding='VALID',
                    use_bias=False, 
                    kernel_initializer=tf.keras.initializers.RandomUniform(-limit, limit)
                )(x)
                
                x = tf.keras.layers.LayerNormalization()(x)  # Replace BN with LN
                '''
                bn_layer = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True,
                    gamma_initializer=tf.keras.initializers.RandomUniform(
                        -1.0 / math.sqrt(4 * self.growth_rate),
                        1.0 / math.sqrt(4 * self.growth_rate)
                    )
                )
                x = bn_layer(x, training=training)  # <-- Pass training dynamically
                '''
                
                x = tf.nn.relu(x)
                x = tf.keras.layers.Dropout(rate=self.dropout_rate)(x, training=training)  # <-- Pass training dynamically
                
                limit = self.bound(4 * self.growth_rate, self.growth_rate, [3, 3])
                x = tf.keras.layers.Conv2D(
                    filters=self.growth_rate, 
                    kernel_size=[3, 3], 
                    strides=1, 
                    padding='SAME',
                    use_bias=False, 
                    kernel_initializer=tf.keras.initializers.RandomUniform(-limit, limit)
                )(x)
                
                x = tf.keras.layers.LayerNormalization()(x)  # Replace BN with LN
                '''
                bn_layer = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True,
                    gamma_initializer=tf.keras.initializers.RandomUniform(
                        -1.0 / math.sqrt(self.growth_rate),
                        1.0 / math.sqrt(self.growth_rate)
                    )
                )
                x = bn_layer(x, training=training)  # <-- Pass training dynamically
                '''
                
                x = tf.nn.relu(x)
                x = tf.keras.layers.Dropout(rate=self.dropout_rate)(x, training=training)  # <-- Pass training dynamically
                
                dense_out = tf.concat([dense_out, x], axis=3)
                x = dense_out
                self.dense_channels += self.growth_rate
                
            if i < self.blocks - 1:
                compressed_channels = int(self.dense_channels * self.transition)
                self.dense_channels = compressed_channels
                
                limit = self.bound(self.dense_channels, compressed_channels, [1, 1])
                x = tf.keras.layers.Conv2D(
                    filters=compressed_channels, 
                    kernel_size=[1, 1], 
                    strides=1, 
                    padding='VALID',
                    use_bias=False, 
                    kernel_initializer=tf.keras.initializers.RandomUniform(-limit, limit)
                )(x)
                
                x = tf.keras.layers.LayerNormalization()(x)  # Replace BN with LN
                '''
                bn_layer = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True,
                    gamma_initializer=tf.keras.initializers.RandomUniform(
                        -1.0 / math.sqrt(self.dense_channels),
                        1.0 / math.sqrt(self.dense_channels)
                    )
                )
                x = bn_layer(x, training=training)  # <-- Pass training dynamically
                '''
                
                x = tf.nn.relu(x)
                x = tf.keras.layers.Dropout(rate=self.dropout_rate)(x, training=training)  # <-- Pass training dynamically
                
                x = tf.keras.layers.AveragePooling2D(pool_size=[2, 2], strides=2, padding='SAME')(x)
                dense_out = x
                mask_x = mask_x[:, 0::2, 0::2]
                
        self.output_dim = dense_out.shape[-1]
        
        return dense_out, mask_x

    '''
    def dense_net(self, input_x, mask_x):

        self.dense_channels = 0
        #### before flowing into dense blocks ####
        if len(input_x.shape)==3:
            input_x = tf.expand_dims(input=input_x, axis=3)
        x = input_x

        limit = self.bound(1, self.input_conv_filters, self.input_conv_kernel)
        x = tf.keras.layers.Conv2D(filters=self.input_conv_filters, strides=self.input_conv_stride,
                                   kernel_size=self.input_conv_kernel, padding='SAME', use_bias=False,
                                   kernel_initializer=tf.keras.initializers.RandomUniform(-limit, limit))(x)
        mask_x = mask_x[:, 0::2, 0::2]
        bn_layer = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True,
                                                      gamma_initializer=tf.keras.initializers.RandomUniform(
                                                          -1.0 / math.sqrt(self.input_conv_filters),
                                                          1.0 / math.sqrt(self.input_conv_filters)))
        x = bn_layer(x, training=self.training)
        #x = tf.keras.layers.BatchNormalization(training=self.training, momentum=0.9, scale=True,
        #                                       gamma_initializer=tf.keras.initializers.RandomUniform(
        #                                           -1.0 / math.sqrt(self.input_conv_filters),
        #                                           1.0 / math.sqrt(self.input_conv_filters)))(x)
        x = tf.nn.relu(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='SAME')(x)

        input_pre = x
        mask_x = mask_x[:, 0::2, 0::2]
        self.dense_channels += self.input_conv_filters
        dense_out = x

        #### flowing into dense blocks and transition_layer ####
        for i in range(self.blocks):
            for j in range(self.level):

                #### [1, 1] convolution part for bottleneck ####
                limit = self.bound(self.dense_channels, 4 * self.growth_rate, [1, 1])
                x = tf.keras.layers.Conv2D(filters=4 * self.growth_rate, kernel_size=[1, 1], strides=1, padding='VALID',
                                           use_bias=False, kernel_initializer=tf.keras.initializers.RandomUniform(-limit, limit))(x)

                bn_layer = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True,
                                                      gamma_initializer=tf.keras.initializers.RandomUniform(
                                                          -1.0 / math.sqrt(4 * self.growth_rate),
                                                          1.0 / math.sqrt(4 * self.growth_rate)))
                x = bn_layer(x, training=self.training)
                #x = tf.keras.layers.BatchNormalization(training=self.training, momentum=0.9, scale=True,
                #                                       gamma_initializer=tf.keras.initializers.RandomUniform(
                #                                           -1.0 / math.sqrt(4 * self.growth_rate),
                #                                           1.0 / math.sqrt(4 * self.growth_rate)))(x)
                x = tf.nn.relu(x)
                x = tf.keras.layers.Dropout(rate=self.dropout_rate)(x)

                #### [3, 3] convolution part for regular convolve operation
                limit = self.bound(4 * self.growth_rate, self.growth_rate, [3, 3])
                x = tf.keras.layers.Conv2D(filters=self.growth_rate, kernel_size=[3, 3], strides=1, padding='SAME',
                                           use_bias=False, kernel_initializer=tf.keras.initializers.RandomUniform(-limit, limit))(x)

                bn_layer = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True,
                                                      gamma_initializer=tf.keras.initializers.RandomUniform(
                                                          -1.0 / math.sqrt(self.growth_rate),
                                                          1.0 / math.sqrt(self.growth_rate)))
                x = bn_layer(x, training=self.training)
                #x = tf.keras.layers.BatchNormalization(training=self.training, momentum=0.9, scale=True,
                #                                       gamma_initializer=tf.keras.initializers.RandomUniform(
                #                                           -1.0 / math.sqrt(self.growth_rate),
                #                                           1.0 / math.sqrt(self.growth_rate)))(x)
                x = tf.nn.relu(x)

                x = tf.keras.layers.Dropout(rate=self.dropout_rate)(x)
                
                dense_out = tf.concat([dense_out, x], axis=3)
                x = dense_out
                
                #### calculate the filter number of dense block's output ####
                self.dense_channels += self.growth_rate

            if i < self.blocks - 1:
                compressed_channels = int(self.dense_channels * self.transition)

                #### new dense channels for new dense block ####
                self.dense_channels = compressed_channels
                limit = self.bound(self.dense_channels, compressed_channels, [1, 1])
                x = tf.keras.layers.Conv2D(filters=compressed_channels, kernel_size=[1, 1], strides=1, padding='VALID',
                                           use_bias=False, kernel_initializer=tf.keras.initializers.RandomUniform(-limit, limit))(x)
                
                bn_layer = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True,
                                                      gamma_initializer=tf.keras.initializers.RandomUniform(
                                                          -1.0 / math.sqrt(self.dense_channels),
                                                          1.0 / math.sqrt(self.dense_channels)))
                x = bn_layer(x, training=self.training)
                #x = tf.keras.layers.BatchNormalization(training=self.training, momentum=0.9, scale=True,
                #                                       gamma_initializer=tf.keras.initializers.RandomUniform(
                #                                           -1.0 / math.sqrt(self.dense_channels),
                #                                           1.0 / math.sqrt(self.dense_channels)))(x)
                x = tf.nn.relu(x)
                x = tf.keras.layers.Dropout(rate=self.dropout_rate)(x)
    
                x = tf.keras.layers.AveragePooling2D(pool_size=[2, 2], strides=2, padding='SAME')(x)

                dense_out = x
                mask_x = mask_x[:, 0::2, 0::2]
        
        self.output_dim = dense_out.shape[-1]

        return dense_out, mask_x
    '''


'''
ContextualAttention class implements contextual attention mechanism. 
'''
class ContextualAttention():
    def __init__(self, channels,                          # output of DenseEncoder | [batch, h, w, channels]
                 dim_decoder, dim_attend):                       # decoder hidden state: $h_{t-1}$ | [batch, dec_dim]

        self.channels = channels

        self.coverage_kernel = [11, 11]                      # kernel size of $Q$
        self.coverage_filters = dim_attend                  # filter numbers of $Q$ | 512

        self.dim_decoder = dim_decoder                      # 256
        self.dim_attend = dim_attend                        # unified dim of three parts calculating $e_ti$ i.e.
                                                            # $Q*beta_t$, $U_a * a_i$, $W_a x h_{t-1}$ | 512
        self.U_f = tf.Variable(utility.norm_weight(self.coverage_filters, self.dim_attend), name='U_f')  # $U_f x f_i$ | [cov_filters, dim_attend]
        self.U_f_b = tf.Variable(np.zeros((self.dim_attend,)).astype('float32'), name='U_f_b')  # $U_f x f_i + U_f_b$ | [dim_attend, ]

        self.U_a = tf.Variable(utility.norm_weight(self.channels, self.dim_attend), name='U_a')  # $U_a x a_i$ | [annotation_channels, dim_attend]
        self.U_a_b = tf.Variable(np.zeros((self.dim_attend,)).astype('float32'), name='U_a_b')  # $U_a x a_i + U_a_b$ | [dim_attend, ]

        self.W_a = tf.Variable(utility.norm_weight(self.dim_decoder, self.dim_attend), name='W_a')  # $W_a x h_{t-1}$ | [dec_dim, dim_attend]
        self.W_a_b = tf.Variable(np.zeros((self.dim_attend,)).astype('float32'), name='W_a_b')  # $W_a x h_{t-1} + W_a_b$ | [dim_attend, ]

        self.V_a = tf.Variable(utility.norm_weight(self.dim_attend, 1), name='V_a')  # $V_a x tanh(A + B + C)$ | [dim_attend, 1]
        self.V_a_b = tf.Variable(np.zeros((1,)).astype('float32'), name='V_a_b')  # $V_a x tanh(A + B + C) + V_a_b$ | [1, ]

        self.alpha_past_filter = tf.Variable(utility.conv_norm_weight(1, self.dim_attend, self.coverage_kernel), name='alpha_past_filter')


    def get_context(self, annotation4ctx, h_t_1, alpha_past4ctx, a_mask):
        """
        Method to compute the context vector, which involves applying the contextual attention mechanism.
        Args:
            annotation4ctx: Annotation features (e.g., from encoder).
            h_t_1: Decoder's previous hidden state.
            alpha_past4ctx: Accumulated alpha values from previous steps.
            a_mask: Attention mask (optional).

        Returns:
            context: The context vector computed from the attention mechanism.
            alpha: The attention weights.
            alpha_past4ctx: The updated accumulated alpha values.
        """
        
        #### calculate $U_f x f_i$ ####
        alpha_past_4d = alpha_past4ctx[:, :, :, None]

        Ft = tf.nn.conv2d(alpha_past_4d, filters=self.alpha_past_filter, strides=[1, 1, 1, 1], padding='SAME')
        coverage_vector = tf.tensordot(Ft, self.U_f, axes=1)  + self.U_f_b    # [batch, h, w, dim_attend]

        #### calculate $U_a x a_i$ ####
        dense_encoder_vector = tf.tensordot(annotation4ctx, self.U_a, axes=1) + self.U_a_b   # [batch, h, w, dim_attend]

        #### calculate $W_a x h_{t - 1}$ ####
        speller_vector = tf.tensordot(h_t_1, self.W_a, axes=1) + self.W_a_b   # [batch, dim_attend]
        speller_vector = speller_vector[:, None, None, :]    # [batch, None, None, dim_attend]

        tanh_vector = tf.tanh(coverage_vector + dense_encoder_vector + speller_vector)    # [batch, h, w, dim_attend]
        e_ti = tf.tensordot(tanh_vector, self.V_a, axes=1) + self.V_a_b  # [batch, h, w, 1]
        alpha = tf.exp(e_ti)
        alpha = tf.squeeze(alpha, axis=3)

        if a_mask is not None:
            alpha = alpha * a_mask

        alpha = alpha / tf.reduce_sum(alpha, axis=[1, 2], keepdims=True)    # normalized weights | [batch, h, w]
        alpha_past4ctx += alpha    # accumulated weights matrix | [batch, h, w]
        context = tf.reduce_sum(annotation4ctx * alpha[:, :, :, None], axis=[1, 2])   # context vector | [batch, feature_channels]
        
        return context, alpha, alpha_past4ctx


'''
Decoder class implements a 2 layered Decoder (GRU) which decodes an input image 
and outputs a sequence of characters using contextual attention mechanism. 
'''
class Decoder():
    def __init__(self, hidden_dim, word_dim, contextual_attention, context_dim, eol_index):

        self.contextual_attention = contextual_attention  # inner-instance of contextual_attention to provide context
        self.context_dim = context_dim                    # context dimension 684
        self.hidden_dim = hidden_dim                      # dim of hidden state 256
        self.word_dim = word_dim                          # dim of word embedding 256
        self.eol_index = eol_index

        # GRU 1 weights initialization starts here
        self.W_yz_yr = tf.Variable(np.concatenate(
            [utility.norm_weight(self.word_dim, self.hidden_dim), utility.norm_weight(self.word_dim, self.hidden_dim)], axis=1), name='W_yz_yr')  # [dim_word, 2 * dim_decoder]
        self.b_yz_yr = tf.Variable(np.zeros((2 * self.hidden_dim,)).astype('float32'), name='b_yz_yr')

        self.U_hz_hr = tf.Variable(np.concatenate(
            [utility.ortho_weight(self.hidden_dim), utility.ortho_weight(self.hidden_dim)], axis=1), name='U_hz_hr')  # [dim_hidden, 2 * dim_hidden]

        self.W_yh = tf.Variable(utility.norm_weight(self.word_dim, self.hidden_dim), name='W_yh')
        self.b_yh = tf.Variable(np.zeros((self.hidden_dim,)).astype('float32'), name='b_yh')  # [dim_decoder, ]

        self.U_rh = tf.Variable(utility.ortho_weight(self.hidden_dim), name='U_rh')  # [dim_hidden, dim_hidden]

        # GRU 2 weights initialization starts here
        self.U_hz_hr_nl = tf.Variable(np.concatenate(
            [utility.ortho_weight(self.hidden_dim), utility.ortho_weight(self.hidden_dim)], axis=1), name='U_hz_hr_nl')  # [dim_hidden, 2 * dim_hidden] non_linear

        self.b_hz_hr_nl = tf.Variable(np.zeros((2 * self.hidden_dim,)).astype('float32'), name='b_hz_hr_nl')  # [2 * dim_hidden, ]

        self.W_c_z_r = tf.Variable(utility.norm_weight(self.context_dim, 2 * self.hidden_dim), name='W_c_z_r')

        self.U_rh_nl = tf.Variable(utility.ortho_weight(self.hidden_dim), name='U_rh_nl')
        self.b_rh_nl = tf.Variable(np.zeros((self.hidden_dim,)).astype('float32'), name='b_rh_nl')

        self.W_c_h_nl = tf.Variable(utility.norm_weight(self.context_dim, self.hidden_dim), name='W_c_h_nl')

    def get_ht_ctx(self, emb_y, target_hidden_state_0, annotations, a_m, y_m):
        """
        This function will compute the hidden state and context using a recurrent approach.
        Args:
            emb_y: Embedding of the input at the current time step.
            target_hidden_state_0: Initial hidden state of the decoder.
            annotations: Attention weights for context.
            a_m: Attention mask.
            y_m: Previous output mask.
        Returns:
            The updated hidden state (h), context, and attention-related variables.
        """
        #print("emb_y.shape: ", emb_y.shape)
        #print("target_hidden_state_0.shape: ", target_hidden_state_0.shape)
        #print("annotations.shape: ", annotations.shape)
        #print("a_m.shape: ", a_m.shape)
        #print("y_m.shape: ", y_m.shape)
        initializer=(target_hidden_state_0,
                                  tf.zeros([tf.shape(annotations)[0], self.context_dim]),
                                  tf.zeros([tf.shape(annotations)[0], tf.shape(annotations)[1], tf.shape(annotations)[2]]),
                                  tf.zeros([tf.shape(annotations)[0], tf.shape(annotations)[1], tf.shape(annotations)[2]]),
                                  annotations, a_m)
        #print("initializer shapes:", [t.shape for t in initializer])
        res = tf.scan(self.one_time_step, elems=(emb_y, y_m),
                      initializer=initializer)

        return res

    def one_time_step(self, tuple_h0_ctx_alpha_alpha_past_annotation, tuple_emb_mask):
        """
        One-time step for GRU decoding with attention.
        Args:
            tuple_h0_ctx_alpha_alpha_past_annotation: Current hidden state, alpha values, and annotations.
            tuple_emb_mask: Current embedding and mask.
        Returns:
            Updated hidden state, context, and attention values.
        """
        target_hidden_state_0 = tuple_h0_ctx_alpha_alpha_past_annotation[0]
        alpha_past_one = tuple_h0_ctx_alpha_alpha_past_annotation[3]
        annotation_one = tuple_h0_ctx_alpha_alpha_past_annotation[4]
        a_mask = tuple_h0_ctx_alpha_alpha_past_annotation[5]

        emb_y, y_mask = tuple_emb_mask
        
        #print("------")
        #[print(i.shape) for i in tuple_h0_ctx_alpha_alpha_past_annotation]
        #print("------")
        #[print(i.shape) for i in tuple_emb_mask]
        #print("------")

        # GRU 1 starts here
        #print("Shapes of emb_y, self.W_yz_yr are ", emb_y.shape, self.W_yz_yr.shape)
        emb_y_z_r_vector = tf.tensordot(emb_y, self.W_yz_yr, axes=1) + self.b_yz_yr  # [batch, 2 * dim_decoder]
        #print("Shapes of target_hidden_state_0, self.U_hz_hr are ", target_hidden_state_0.shape, self.U_hz_hr.shape)
        hidden_z_r_vector = tf.tensordot(target_hidden_state_0, self.U_hz_hr, axes=1)  # [batch, 2 * dim_decoder]
        #print("hidden_z_r_vector shape: ", hidden_z_r_vector.shape)
        pre_z_r_vector = tf.sigmoid(emb_y_z_r_vector + hidden_z_r_vector)  # [batch, 2 * dim_decoder]
        #if len(pre_z_r_vector.shape) == 2:
        #    pre_z_r_vector = tf.expand_dims(pre_z_r_vector, axis=1)  # Convert to 3D if needed

        #print("pre_z_r_vector shape: ", pre_z_r_vector.shape)
        #print("hidden_dim: ", self.hidden_dim)
        r1 = pre_z_r_vector[:, :self.hidden_dim]  # [batch, dim_decoder]
        #print("r1 shape:", r1.shape)
        z1 = pre_z_r_vector[:, self.hidden_dim:]  # [batch, dim_decoder]
        #print("z1 shape:", z1.shape)
        #r1 = tf.slice(pre_z_r_vector, [0, 0, 0], [-1, -1, self.hidden_dim])
        #z1 = tf.slice(pre_z_r_vector, [0, 0, self.hidden_dim], [-1, -1, -1])
        #print("r1 shape after tf.slice:", r1.shape)


        emb_y_h_vector = tf.tensordot(emb_y, self.W_yh, axes=1) + self.b_yh  # [batch, dim_decoder]
        hidden_r_h_vector = tf.tensordot(target_hidden_state_0, self.U_rh, axes=1)  # [batch, dim_decoder]
        hidden_r_h_vector *= r1
        pre_h_proposal = tf.tanh(hidden_r_h_vector + emb_y_h_vector)

        pre_h = z1 * target_hidden_state_0 + (1. - z1) * pre_h_proposal

        if y_mask is not None:
            pre_h = y_mask[:, None] * pre_h + (1. - y_mask)[:, None] * target_hidden_state_0

        context, alpha, alpha_past_one = self.contextual_attention.get_context(annotation_one, pre_h, alpha_past_one, a_mask)  # [batch, dim_ctx]

        # GRU 2 starts here
        emb_y_z_r_nl_vector = tf.tensordot(pre_h, self.U_hz_hr_nl, axes=1) + self.b_hz_hr_nl
        context_z_r_vector = tf.tensordot(context, self.W_c_z_r, axes=1)
        z_r_vector = tf.sigmoid(emb_y_z_r_nl_vector + context_z_r_vector)

        r2 = z_r_vector[:, :self.hidden_dim]  # [batch, dim_decoder]
        z2 = z_r_vector[:, self.hidden_dim:]  # [batch, dim_decoder]
        #r2 = tf.slice(z_r_vector, [0, 0, 0], [-1, -1, self.hidden_dim])
        #z2 = tf.slice(z_r_vector, [0, 0, self.hidden_dim], [-1, -1, -1])
        #print("\nr2 shape: ", r2.shape)

        emb_y_h_nl_vector = tf.tensordot(pre_h, self.U_rh_nl, axes=1)
        emb_y_h_nl_vector *= r2
        emb_y_h_nl_vector = emb_y_h_nl_vector + self.b_rh_nl  # bias added after pointwise multiplication with r2
        context_h_vector = tf.tensordot(context, self.W_c_h_nl, axes=1)
        h_proposal = tf.tanh(emb_y_h_nl_vector + context_h_vector)
        h = z2 * pre_h + (1. - z2) * h_proposal

        if y_mask is not None:
            h = y_mask[:, None] * h + (1. - y_mask)[:, None] * pre_h
        
        #print("\n---------\nShapes returned by one_time_step")
        #print(h.shape, context.shape, alpha.shape, alpha_past_one.shape, annotation_one.shape, a_mask.shape)
        #print("---------\n")
        return h, context, alpha, alpha_past_one, annotation_one, a_mask


'''
CALText class is the main class. This class uses below three classes:
1) DenseEncoder (Encoder)
2) ContextualAttention (Contextual attention mechnism)
3) Decoder (2 layerd GRU Decoder)
CALText class implements two functions get_cost and get_sample, which are actually used for cost calculation and decoding.
'''
class CALText(tf.keras.Model):
    def __init__(self, dense_encoder, contextual_attention, decoder, hidden_dim, word_dim, context_dim, target_dim, training):
        super(CALText, self).__init__()

        self.hidden_dim = hidden_dim
        self.word_dim = word_dim
        self.context_dim = context_dim
        self.target_dim = target_dim

        # Embedding matrix
        self.embed_matrix = tf.Variable(utility.norm_weight(self.target_dim, self.word_dim), name='embed')

        # Encoder, attention, and decoder components
        self.dense_encoder = dense_encoder
        self.contextual_attention = contextual_attention
        self.decoder = decoder

        # Weight variables
        self.Wa2h = tf.Variable(utility.norm_weight(self.context_dim, self.hidden_dim), name='Wa2h')
        self.ba2h = tf.Variable(np.zeros((self.hidden_dim,)).astype('float32'), name='ba2h')
        self.Wc = tf.Variable(utility.norm_weight(self.context_dim, self.word_dim), name='Wc')
        self.bc = tf.Variable(np.zeros((self.word_dim,)).astype('float32'), name='bc')
        self.Wh = tf.Variable(utility.norm_weight(self.hidden_dim, self.word_dim), name='Wh')
        self.bh = tf.Variable(np.zeros((self.word_dim,)).astype('float32'), name='bh')
        self.Wy = tf.Variable(utility.norm_weight(self.word_dim, self.word_dim), name='Wy')
        self.by = tf.Variable(np.zeros((self.word_dim,)).astype('float32'), name='by')
        self.Wo = tf.Variable(utility.norm_weight(self.word_dim // 2, self.target_dim), name='Wo')
        self.bo = tf.Variable(np.zeros((self.target_dim,)).astype('float32'), name='bo')

        self.training = training

    @tf.function
    def get_cost(self, cost_annotation, cost_y, a_m, y_m, alpha_reg):
        """
        Compute the cost function for the training loop.
        """
        # Step 1: Preparation of embedding of labels sequences
        timesteps = tf.shape(cost_y)[0]
        batch_size = tf.shape(cost_y)[1]
        emb_y = tf.nn.embedding_lookup(self.embed_matrix, tf.reshape(cost_y, [-1]))
        emb_y = tf.reshape(emb_y, [timesteps, batch_size, self.word_dim])
        emb_pad = tf.fill((1, batch_size, self.word_dim), 0.0)
        emb_shift = tf.concat([emb_pad, tf.strided_slice(emb_y, [0, 0, 0], [-1, batch_size, self.word_dim], [1, 1, 1])], axis=0)
        new_emb_y = emb_shift

        # Step 2: Calculation of h_0
        #print("\nIn get_cost(): cost_annotation.shape is ", cost_annotation.shape)
        #print("\nIn get_cost(): a_m.shape is ", a_m.shape)
        anno_mean = tf.reduce_sum(cost_annotation * a_m[:, :, :, None], axis=[1, 2]) / tf.reduce_sum(a_m, axis=[1, 2])[:, None]
        #anno_mean = tf.reduce_sum(cost_annotation * a_m, axis=[1, 2]) / (tf.reduce_sum(a_m, axis=[1, 2]) + 1e-8)  # Added small value to prevent division by zero
        #print("\nIn get_cost(): anno_mean.shape is ", anno_mean.shape)
        #print("\nIn get_cost(): self.Wa2h.shape is ", self.Wa2h.shape)
        h_0 = tf.tensordot(anno_mean, self.Wa2h, axes=1) + self.ba2h  # [batch, hidden_dim]
        #print("\nIn get_cost(): h_0.shape is ", h_0.shape)
        h_0 = tf.tanh(h_0)

        # Step 3: Calculation of h_t and c_t at all time steps
        ret = self.decoder.get_ht_ctx(new_emb_y, h_0, cost_annotation, a_m, y_m)
        h_t = ret[0]  # h_t of all timesteps [timesteps, batch, hidden_dim]
        c_t = ret[1]  # c_t of all timesteps [timesteps, batch, context_dim]
        alpha = ret[2]  # alpha of all timesteps [timesteps, batch, h, w]

        # Step 4: Calculation of cost using h_t, c_t, and y_t_1
        y_t_1 = new_emb_y  # shifted y | [1:] = [:-1]
        logit_gru = tf.tensordot(h_t, self.Wh, axes=1) + self.bh
        logit_ctx = tf.tensordot(c_t, self.Wc, axes=1) + self.bc
        logit_pre = tf.tensordot(y_t_1, self.Wy, axes=1) + self.by
        logit = logit_pre + logit_ctx + logit_gru
        shape = tf.shape(logit)
        logit = tf.reshape(logit, [shape[0], -1, shape[2] // 2, 2])
        logit = tf.reduce_max(logit, axis=3)
        logit = tf.keras.layers.Dropout(0.2)(logit, training=self.training)
        logit = tf.tensordot(logit, self.Wo, axes=1) + self.bo
        logit_shape = tf.shape(logit)
        logit = tf.reshape(logit, [-1, logit_shape[2]])

        cost = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=tf.one_hot(tf.reshape(cost_y, [-1]), depth=self.target_dim))

        # Max pooling on vector with size equal to word_dim
        cost = tf.multiply(cost, tf.reshape(y_m, [-1]))
        cost = tf.reshape(cost, [shape[0], shape[1]])
        cost = tf.reduce_sum(cost, axis=0)
        cost = tf.reduce_mean(cost)

        # Alpha L1 regularization
        alpha_sum = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.abs(alpha), axis=[2, 3]), axis=0))
        cost = tf.cond(alpha_reg > 0, lambda: cost + (alpha_reg * alpha_sum), lambda: cost)

        return cost

    @tf.function
    def get_word(self, sample_y, sample_h_pre, alpha_past_pre, sample_annotation, training=False, stochastic=True):
        """
        Computes the next word probabilities and selects the next word using a GRU-based decoder.
        
        Args:
            sample_y (Tensor): Previous word index.
            sample_h_pre (Tensor): Previous hidden state.
            alpha_past_pre (Tensor): Previous attention weights.
            sample_annotation (Tensor): Context annotations.
            training (bool): If True, enables dropout for training.
            stochastic (bool): If True, samples the next word; otherwise, uses greedy decoding.
        
        Returns:
            next_probs (Tensor): Probabilities of the next word.
            next_word (Tensor): Sampled next word index (stochastic) or argmax (greedy).
            h_t (Tensor): Updated hidden state.
            alpha_past_t (Tensor): Updated attention weights.
            contextV (Tensor): Attention context vector.
        """
        
        # Embedding lookup with conditional zeroing for start token (-1)
        emb = tf.where(
            sample_y[0] < 0,
            tf.zeros((1, self.word_dim), dtype=tf.float32),
            tf.nn.embedding_lookup(self.embed_matrix, sample_y)
        )
        
        # Compute GRU gating values
        emb_y_z_r_vector = tf.tensordot(emb, self.decoder.W_yz_yr, axes=1) + self.decoder.b_yz_yr
        hidden_z_r_vector = tf.tensordot(sample_h_pre, self.decoder.U_hz_hr, axes=1)
        pre_z_r_vector = tf.sigmoid(emb_y_z_r_vector + hidden_z_r_vector)
        
        r1 = pre_z_r_vector[:, :self.decoder.hidden_dim]
        z1 = pre_z_r_vector[:, self.decoder.hidden_dim:]
        
        # Compute GRU candidate activation
        emb_y_h_vector = tf.tensordot(emb, self.decoder.W_yh, axes=1) + self.decoder.b_yh
        hidden_r_h_vector = tf.tensordot(sample_h_pre, self.decoder.U_rh, axes=1)
        hidden_r_h_vector *= r1
        pre_h_proposal = tf.tanh(hidden_r_h_vector + emb_y_h_vector)
        pre_h = z1 * sample_h_pre + (1.0 - z1) * pre_h_proposal
        
        # Compute attention context
        context, contextV, alpha_past = self.decoder.contextual_attention.get_context(
            sample_annotation, pre_h, alpha_past_pre, None
        )
        
        # Second GRU update with context
        emb_y_z_r_nl_vector = tf.tensordot(pre_h, self.decoder.U_hz_hr_nl, axes=1) + self.decoder.b_hz_hr_nl
        context_z_r_vector = tf.tensordot(context, self.decoder.W_c_z_r, axes=1)
        z_r_vector = tf.sigmoid(emb_y_z_r_nl_vector + context_z_r_vector)
        
        r2 = z_r_vector[:, :self.decoder.hidden_dim]
        z2 = z_r_vector[:, self.decoder.hidden_dim:]
        
        emb_y_h_nl_vector = tf.tensordot(pre_h, self.decoder.U_rh_nl, axes=1) + self.decoder.b_rh_nl
        emb_y_h_nl_vector *= r2
        context_h_vector = tf.tensordot(context, self.decoder.W_c_h_nl, axes=1)
        h_proposal = tf.tanh(emb_y_h_nl_vector + context_h_vector)
        
        # Final hidden state update
        h_t = z2 * pre_h + (1.0 - z2) * h_proposal
        alpha_past_t = alpha_past
        
        # Compute logits for next word prediction
        logit_gru = tf.tensordot(h_t, self.Wh, axes=1) + self.bh
        logit_ctx = tf.tensordot(context, self.Wc, axes=1) + self.bc
        logit_pre = tf.tensordot(emb, self.Wy, axes=1) + self.by
        logit = logit_pre + logit_ctx + logit_gru
        
        # Max pooling along word dimension
        shape = tf.shape(logit)
        logit = tf.reshape(logit, [-1, shape[1] // 2, 2])
        logit = tf.reduce_max(logit, axis=2)
        
        # Apply dropout (conditional on `training` flag)
        logit = tf.keras.layers.Dropout(0.2)(logit, training=training)
        
        # Final transformation for word selection
        logit = tf.tensordot(logit, self.Wo, axes=1) + self.bo
        
        # Compute probabilities
        next_probs = tf.nn.softmax(logit)
        
        # Sample next word (stochastic or greedy)
        if stochastic:
            next_word = tf.random.categorical(tf.math.log(next_probs), num_samples=1)
        else:
            next_word = tf.argmax(next_probs, axis=-1, output_type=tf.int32)
        
        if next_word.shape.rank == 2:  # Check if it has an extra dimension
            next_word = tf.squeeze(next_word, axis=-1)  # Remove last dimension safely
        
        return next_probs, next_word, h_t, alpha_past_t, contextV

    
    #@tf.function
    def get_sample(self, ctx0, h_0, k, maxlen, stochastic, training):
        """
        Computes a sequence of labels/characters from annotations using beam search if stochastic is set to False.
        """
        sample = []
        sample_score = []
        sample_att = []

        live_k = 1
        dead_k = 0

        hyp_samples = [[]] * 1
        hyp_scores = tf.zeros(live_k, dtype=tf.float32)
        hyp_states = []

        #next_alpha_past = tf.zeros_like(ctx0, dtype=tf.float32)
        next_alpha_past = tf.zeros((ctx0.shape[0], ctx0.shape[1], ctx0.shape[2]), dtype=tf.float32)
        next_w = tf.constant([-1], dtype=tf.int64)

        next_state = h_0

        for ii in tf.range(maxlen):
            ctx = tf.repeat(ctx0, live_k, axis=0)
            
            next_p, next_w, next_state, next_alpha_past, contextVec = self.get_word(next_w, next_state, next_alpha_past, ctx, training, stochastic)
            sample_att.append(contextVec[0, :, :])

            if stochastic:
                #nw = next_w[0].numpy() #Greedy. Pick most probable word
                sampled_w = tf.random.categorical(tf.math.log(next_p), num_samples=1)
                nw = sampled_w[0, 0].numpy()  # Extract sampled word
                sample.append(nw)
                sample_score.append(next_p[0, nw].numpy())

                if nw==0: #0 indicates end-of-line
                    break
            else:
                cand_scores = hyp_scores[:, None] - tf.math.log(next_p)
                cand_flat = tf.reshape(cand_scores, [-1])
                ranks_flat = tf.argsort(cand_flat)[: (k - dead_k)]
                voc_size = tf.shape(next_p)[1]
                #print("cand_scores: ", cand_scores)
                #print("cand_flat: ", cand_flat)
                #print("ranks_flat: ", ranks_flat)
                #print("hyp_scores length: ", len(hyp_scores))
                #print("hyp_scores: ", hyp_scores)
                #print("next_p.shape: ", next_p.shape)
                #print("voc_size: ", voc_size)

                trans_indices = ranks_flat // voc_size  # Beam indices
                word_indices = ranks_flat % voc_size  # Word indices
                costs = tf.gather(cand_flat, ranks_flat)

                new_hyp_samples = []
                new_hyp_scores = []
                new_hyp_states = []
                new_hyp_alpha_past = []

                #print("ranks_flat shape: ", tf.shape(ranks_flat))
                #print("hyp_samples length: ", len(hyp_samples))
                #print("trans_indices: ", trans_indices)
                #print("word_indices: ", word_indices)
                for idx in tf.range(tf.shape(ranks_flat)[0]):
                    ti = trans_indices[idx].numpy()
                    wi = word_indices[idx].numpy()

                    new_hyp_samples.append(hyp_samples[ti] + [wi])
                    new_hyp_scores.append(costs[idx].numpy())
                    new_hyp_states.append(tf.identity(next_state[ti]))
                    new_hyp_alpha_past.append(tf.identity(next_alpha_past[ti]))

                new_live_k = 0
                hyp_samples = []
                hyp_scores = []
                hyp_states = []
                hyp_alpha_past = []

                for idx in range(len(new_hyp_samples)):
                    if new_hyp_samples[idx][-1] == self.decoder.eol_index: #0:  # 0 indicates <eol>
                        sample.append(new_hyp_samples[idx])
                        sample_score.append(new_hyp_scores[idx])
                        dead_k += 1
                    else:
                        new_live_k += 1
                        hyp_samples.append(new_hyp_samples[idx])
                        hyp_scores.append(new_hyp_scores[idx])
                        hyp_states.append(new_hyp_states[idx])
                        hyp_alpha_past.append(new_hyp_alpha_past[idx])

                hyp_scores = tf.convert_to_tensor(hyp_scores, dtype=tf.float32)
                live_k = new_live_k

                if live_k < 1:
                    break
                if dead_k >= k:
                    break

                next_w = tf.convert_to_tensor([w1[-1] for w1 in hyp_samples], dtype=tf.int64)
                next_state = tf.stack(hyp_states)
                next_alpha_past = tf.stack(hyp_alpha_past)

        if not stochastic:
            # Store remaining hypotheses
            if live_k > 0:
                for idx in range(live_k):
                    sample.append(hyp_samples[idx])
                    sample_score.append(hyp_scores[idx].numpy())

        return sample, sample_score, sample_att

    
    '''
    @tf.function
    def get_word(self, sample_y, sample_h_pre, alpha_past_pre, sample_annotation):
        """
        Predict the next word based on previous word and state.
        """
        emb = tf.cond(
            sample_y[0] < 0,
            lambda: tf.fill((1, self.word_dim), 0.0),
            lambda: tf.nn.embedding_lookup(self.embed_matrix, sample_y)
        )

        # GRU Step (similar to the previous code, but updated for tf2.x)
        emb_y_z_r_vector = tf.tensordot(emb, self.decoder.W_yz_yr, axes=1) + self.decoder.b_yz_yr
        hidden_z_r_vector = tf.tensordot(sample_h_pre, self.decoder.U_hz_hr, axes=1)
        pre_z_r_vector = tf.sigmoid(emb_y_z_r_vector + hidden_z_r_vector)

        r1 = pre_z_r_vector[:, :self.decoder.hidden_dim]
        z1 = pre_z_r_vector[:, self.decoder.hidden_dim:]

        emb_y_h_vector = tf.tensordot(emb, self.decoder.W_yh, axes=1) + self.decoder.b_yh
        hidden_r_h_vector = tf.tensordot(sample_h_pre, self.decoder.U_rh, axes=1)
        hidden_r_h_vector *= r1
        pre_h_proposal = tf.tanh(hidden_r_h_vector + emb_y_h_vector)

        pre_h = z1 * sample_h_pre + (1. - z1) * pre_h_proposal

        context, contextV, alpha_past = self.decoder.contextual_attention.get_context(
            sample_annotation, pre_h, alpha_past_pre, None
        )

        emb_y_z_r_nl_vector = tf.tensordot(pre_h, self.decoder.U_hz_hr_nl, axes=1) + self.decoder.b_hz_hr_nl
        context_z_r_vector = tf.tensordot(context, self.decoder.W_c_z_r, axes=1)
        z_r_vector = tf.sigmoid(emb_y_z_r_nl_vector + context_z_r_vector)

        r2 = z_r_vector[:, :self.decoder.hidden_dim]
        z2 = z_r_vector[:, self.decoder.hidden_dim:]

        emb_y_h_nl_vector = tf.tensordot(pre_h, self.decoder.U_rh_nl, axes=1) + self.decoder.b_rh_nl
        emb_y_h_nl_vector *= r2
        context_h_vector = tf.tensordot(context, self.decoder.W_c_h_nl, axes=1)
        h_proposal = tf.tanh(emb_y_h_nl_vector + context_h_vector)
        h = z2 * pre_h + (1. - z2) * h_proposal

        h_t = h
        c_t = context
        alpha_past_t = alpha_past
        y_t_1 = emb
        logit_gru = tf.tensordot(h_t, self.Wh, axes=1) + self.bh
        logit_ctx = tf.tensordot(c_t, self.Wc, axes=1) + self.bc
        logit_pre = tf.tensordot(y_t_1, self.Wy, axes=1) + self.by
        logit = logit_pre + logit_ctx + logit_gru

        shape = tf.shape(logit)
        logit = tf.reshape(logit, [-1, shape[1] // 2, 2])
        logit = tf.reduce_max(logit, axis=2)

        logit = tf.keras.layers.Dropout(0.2)(logit, training=self.training)

        logit = tf.tensordot(logit, self.Wo, axes=1) + self.bo

        next_probs = tf.nn.softmax(logits=logit)
        next_word = tf.reduce_max(tf.random.categorical(next_probs, num_samples=1), axis=1)
        return next_probs, next_word, h_t, alpha_past_t, contextV

    @tf.function
    def get_sample(self, p, w, h, alpha, ctv, ctx0, h_0, k, maxlen, stochastic, session, training):
        """
        Perform sequence generation using beam search.
        """
        sample = []
        sample_score = []
        sample_att = []

        live_k = 1
        dead_k = 0

        hyp_samples = [[]] * 1
        hyp_scores = np.zeros(live_k).astype('float32')
        hyp_states = []

        next_alpha_past = np.zeros((ctx0.shape[0], ctx0.shape[1], ctx0.shape[2])).astype('float32')
        emb_0 = np.zeros((ctx0.shape[0], 256))

        next_w = -1 * np.ones((1,)).astype('int64')

        next_state = h_0
        for ii in range(maxlen):
            ctx = np.tile(ctx0, [live_k, 1, 1, 1])

            input_dict = {
                anno: ctx,
                infer_y: next_w,
                alpha_past: next_alpha_past,
                h_pre: next_state,
                if_trainning: training
            }

            next_p, next_w, next_state, next_alpha_past, contexVec = session.run(
                [p, w, h, alpha, ctv], feed_dict=input_dict
            )
            sample_att.append(contexVec[0, :, :])

            if stochastic:
                nw = next_w[0]
                sample.append(nw)
                sample_score += next_p[0, nw]
                if nw == 0:
                    break
            else:
                cand_scores = hyp_scores[:, None] - np.log(next_p)
                cand_flat = cand_scores.flatten()
                ranks_flat = cand_flat.argsort()[:beam_width]
                live_k = len(ranks_flat)
                dead_k = beam_width - live_k
                sample.append(hyp_samples)
                sample_score.append(np.array(sample_scores))
    '''

class Model(tf.keras.Model):
    def __init__(self, classes, eol_index):
        super(Model, self).__init__()

        #### Encoder setup parameters ####
        dense_blocks = 3
        levels_count = 16
        growth = 24

        #### Decoder setup parameters ####
        hidden_dim = 256
        word_dim = 256
        dim_attend = 512

        self.num_classes = classes

        # Learning rate and regularization parameters
        self.lr = 0.001 #tf.Variable(0.001, trainable=False, dtype=tf.float32)
        self.alpha_reg = tf.Variable(0.0, trainable=False, dtype=tf.float32)

        # Define Dense Encoder
        self.dense_encoder = DenseEncoder(
            blocks=dense_blocks,
            level=levels_count,
            growth_rate=growth,
            training=True
        )

        # Define Contextual Attention and Decoder
        self.contextual_attention = ContextualAttention(self.dense_encoder.output_dim, hidden_dim, dim_attend)
        self.decoder = Decoder(hidden_dim, word_dim, self.contextual_attention, self.dense_encoder.output_dim, eol_index)

        # Define CALText model
        self.caltext = CALText(
            self.dense_encoder, self.contextual_attention, self.decoder,
            hidden_dim, word_dim, self.dense_encoder.output_dim, self.num_classes, training=True
        )

        # Adam optimizer (alternative to Adadelta)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def call(self, inputs, training=False):
        x, y, x_mask, y_mask = inputs

        # Encode input features
        annotation, anno_mask = self.dense_encoder(x, x_mask, training=training) 

        # Compute loss
        loss = self.caltext.get_cost(annotation, y, anno_mask, y_mask, self.alpha_reg)

        return loss

    #@tf.function
    def train_step(self, batch_x, batch_y, batch_x_m, batch_y_m):
        """
        Custom training step that computes loss and applies gradients.
        """
        with tf.GradientTape() as tape:
            loss = self.call((batch_x, batch_y, batch_x_m, batch_y_m), training=True)

        # Compute gradients and update weights
        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        grads, _ = tf.clip_by_global_norm(grads, 100)  # Gradient clipping
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        return loss
    
    def test_step(self, batch_x, batch_y, batch_x_m, batch_y_m):
        """
        Computes loss on a test/validation batch.
        """
        loss = self.call((batch_x, batch_y, batch_x_m, batch_y_m), training=False)
        return loss
    
    def predict_step(self, batch_x, batch_x_m, maxlen=100, k=5, stochastic=False):
        """
        Predicts the sequence of labels/characters for a batch of input images.
    
        Args:
            batch_x (tf.Tensor): Input image tensor of shape (batch_size, height, width, channels).
            batch_x_m (tf.Tensor): Input image mask tensor of shape (batch_size, height, width)
            maxlen (int): Maximum sequence length to predict.
            k (int): Beam width for beam search decoding.
            stochastic (bool): If True, uses stochastic sampling instead of beam search.
    
        Returns:
            tuple: (predicted sequences, prediction scores, attention maps)
        """
    
        # Pass the image through the Dense Encoder
        annotation, anno_mask = self.dense_encoder(batch_x, batch_x_m, training=False)
    
        # Initialize decoder hidden state
        batch_size = tf.shape(batch_x)[0]
        #h_0 = tf.zeros((batch_size, self.decoder.hidden_dim))
        h_0 = tf.zeros(self.decoder.hidden_dim)
        
        samples=[]
        sample_scores=[]
        sample_atts=[]
        for idxx, ann in enumerate(annotation):
            ann = tf.expand_dims(ann,axis=0)
            # Use CALText's get_sample() function to generate predictions
            sample, sample_score, sample_att = self.caltext.get_sample(ctx0=ann, h_0=h_0, k=k, maxlen=maxlen, stochastic=stochastic, training=False)
            samples.append(sample)
            sample_scores.append(sample_score)
            sample_atts.append(sample_att)
        '''
        sample, sample_score, sample_att = self.caltext.get_sample(
            p=self.decoder.p,
            w=self.decoder.w,
            h=self.decoder.h,
            alpha=self.decoder.alpha,
            ctv=self.decoder.ctv,
            ctx0=annotation,
            h_0=h_0,
            k=k,
            maxlen=maxlen,
            stochastic=stochastic,
            training=False
        )
        '''
        
        return samples, sample_scores, sample_atts


'''
class GetWordLayer(tf.keras.layers.Layer):
    def __init__(self, caltext, **kwargs):
        super(GetWordLayer, self).__init__(**kwargs)
        self.caltext = caltext  # Store the CALText instance
    
    def call(self, inputs):
        infer_y, h_pre, alpha_past, anno = inputs
        return self.caltext.get_word(infer_y, h_pre, alpha_past, anno)

class GetCostLayer(tf.keras.layers.Layer):
    def __init__(self, caltext, alpha_reg, **kwargs):
        super(GetCostLayer, self).__init__(**kwargs)
        self.caltext = caltext
        self.alpha_reg = alpha_reg

    def call(self, inputs):
        annotations, y, anno_mask, y_mask = inputs
        print("In call(): ", annotations.shape, y.shape, anno_mask.shape, y_mask.shape)
        return self.caltext.get_cost(annotations, y, anno_mask, y_mask, self.alpha_reg)
               

class Model(tf.keras.Model):
    """
    This class defines a deep learning model using TensorFlow 2.
    
    Attributes:
        x (tf.keras.Input): Input tensor for image data.
        y (tf.keras.Input): Output tensor for sequence data.
        x_mask (tf.keras.Input): Mask tensor for input images.
        y_mask (tf.keras.Input): Mask tensor for output sequences.
        lr (tf.Variable): Learning rate variable.
        alpha_reg (tf.Variable): Regularization factor.
        trainer (tf.Operation): Training operation.
    
    Methods:
        build_model(classes): Constructs the model architecture.
        create_trainer(): Sets up the optimizer and training operation.
    """
    
    def build_model(self, classes):
        """
        Builds the model architecture by defining the encoder, decoder, and attention mechanisms.
        
        Args:
            classes (int): Number of output classes for the model.
        
        This method initializes input tensors, sets up the encoder using a DenseNet architecture,
        configures the decoder with contextual attention, and prepares the training cost function
        with L2 regularization. It also defines the optimizer and gradient clipping strategy.
        """
        #### encoder setup parameters ####
        dense_blocks = 3
        levels_count = 16
        growth = 24

        #### decoder setup parameters ####
        hidden_dim = 256
        word_dim = 256
        dim_attend = 512

        self.x = tf.keras.Input(shape=(None, None), dtype=tf.float32, name="image_input")  # Input tensor for images
        self.y = tf.keras.Input(shape=(None,), dtype=tf.int32, name="sequence_output")  # Output tensor for sequences
        self.x_mask = tf.keras.Input(shape=(None, None), dtype=tf.float32, name="image_mask")
        self.y_mask = tf.keras.Input(shape=(None,), dtype=tf.float32, name="sequence_mask")

        #global anno, infer_y, h_pre, alpha_past, if_training, num_classes

        self.num_classes = classes
        self.lr = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.if_training = False #tf.Variable(False, dtype=tf.bool, trainable=False)
        self.alpha_reg = tf.Variable(0.0, dtype=tf.float32, trainable=False)

        dense_encoder = DenseEncoder(blocks=dense_blocks, level=levels_count, growth_rate=growth, training=self.if_training)
        self.annotation, self.anno_mask = dense_encoder.dense_net(self.x, self.x_mask)

        # For initializing validation
        self.anno = tf.keras.Input(shape=(self.annotation.shape[1], self.annotation.shape[2], self.annotation.shape[3]), dtype=tf.float32)
        self.infer_y = tf.keras.Input(shape=(), dtype=tf.int64)
        self.h_pre = tf.keras.Input(shape=(hidden_dim), dtype=tf.float32)
        self.alpha_past = tf.keras.Input(shape=(self.annotation.shape[1], self.annotation.shape[2]), dtype=tf.float32)

        contextual_attention = ContextualAttention(self.annotation.shape[3], hidden_dim, dim_attend)
        decoder = Decoder(hidden_dim, word_dim, contextual_attention, self.annotation.shape[3])
        
        self.caltext = CALText(dense_encoder, contextual_attention, decoder, hidden_dim, word_dim, self.annotation.shape[3], self.num_classes, self.if_training)
        
        self.hidden_state_0 = tf.tanh(tf.linalg.matmul(tf.reduce_mean(self.anno, axis=[1, 2]), self.caltext.Wa2h) + self.caltext.ba2h)
        
        self.get_cost_layer = GetCostLayer(self.caltext, self.alpha_reg)
        print("\nIn build_model(): self.annotation.shape is ", self.annotation.shape)
        print("\nIn build_model(): self.y.shape is ", self.y.shape)
        self.cost = self.compute_cost(self.annotation, self.y, self.anno_mask, self.y_mask, self.alpha_reg)
        #self.cost = self.caltext.get_cost(self.annotation, self.y, self.anno_mask, self.y_mask, self.alpha_reg)

        # Regularization
        alpha_c = 0.5
        for vv in self.trainable_variables:
            if not vv.name.startswith('conv2d'):
                self.cost += 1e-4 * tf.reduce_sum(tf.pow(vv, 2))
        
        self.get_word_layer = GetWordLayer(self.caltext)
        self.p, self.w, self.h, self.alpha, self.contexV = self.get_word_layer([self.infer_y, self.h_pre, self.alpha_past, self.anno])
        #self.p, self.w, self.h, self.alpha, self.contexV = self.caltext.get_word(infer_y, h_pre, alpha_past, anno)
        
        # Setup training operation
        self.create_trainer()
    
    def compute_cost(self, annotation, y, anno_mask, y_mask, alpha_reg):
        return self.get_cost_layer([annotation, y, anno_mask, y_mask])# + alpha_reg * tf.reduce_sum(tf.pow(self.trainable_variables, 2))
    
    #def create_trainer(self):
    #    """
    #    Sets up the optimizer and training operation with gradient clipping.
    #    """
    #    optimizer = Adadelta(learning_rate=self.lr)
    #    trainable_vars = tf.compat.v1.trainable_variables()
    #    grads = tf.gradients(self.cost, trainable_vars)
    #    clipped_grads, _ = tf.clip_by_global_norm(grads, 100)
    #    self.trainer = optimizer.apply_gradients(zip(clipped_grads, trainable_vars))
    
    def create_trainer(self):
        """
        Sets up the optimizer and training operation with gradient clipping using TF2.
        """
        optimizer = tf.keras.optimizers.Adadelta(learning_rate=self.lr)
        
        # Use a GradientTape for automatic differentiation
        @tf.function
        def train_step():
            with tf.GradientTape() as tape:
                # Step 1: Encode input features into annotation
                annotation, anno_mask = self.dense_encoder.dense_net(self.x, self.x_mask)  # [batch, height, width, channels], [batch, height, width]
                #loss = self.get_cost_layer([self.annotation, self.y, self.anno_mask, self.y_mask])
                loss = self.compute_cost(self.annotation, self.y, self.anno_mask, self.y_mask, self.alpha_reg)
            
            # Compute gradients
            trainable_vars = self.trainable_variables
            grads = tape.gradient(loss, trainable_vars)
            
            # Clip gradients
            clipped_grads, _ = tf.clip_by_global_norm(grads, 100)
            
            # Apply gradients
            optimizer.apply_gradients(zip(clipped_grads, trainable_vars))
        
        self.trainer = train_step  # Assign train function


def model_infer(model, xx_pad, max_len, beam_size):
    """
    Performs inference using the trained model.

    Args:
        model (Model): The trained model instance.
        xx_pad (np.array): Input data padded for inference.
        max_len (int): Maximum sequence length for generation.
        beam_size (int): The beam width for beam search.

    Returns:
        tuple: The predicted sequence and attention weights.
    """
    annot = model.annotation(xx_pad, training=False)
    h_state = model.hidden_state_0(annot)
    
    sample, score, hypalpha = model.caltext.get_sample(
        model.p, model.w, model.h, model.alpha, model.contexV, 
        annot, h_state, beam_size, max_len, False, training=False
    )
    
    score = score / np.array([len(s) for s in sample])
    ss = sample[np.argmin(score)]
    
    return ss, hypalpha

def model_getcost(model, batch_x, batch_y, batch_x_m, batch_y_m):
    """
    Computes the loss for a batch of input-output data.

    Args:
        model (Model): The trained model instance.
        batch_x (tf.Tensor): Input batch.
        batch_y (tf.Tensor): Ground truth output batch.
        batch_x_m (tf.Tensor): Mask for input.
        batch_y_m (tf.Tensor): Mask for output.

    Returns:
        float: Computed loss value.
    """
    pprobs = model.cost(
        batch_x, batch_y, batch_x_m, batch_y_m, training=False, alpha_reg=1
    )
    return pprobs.numpy()

def model_train(model, batch_x, batch_y, batch_x_m, batch_y_m, lrate, alpha_reg):
    """
    Trains the model on a batch of data.

    Args:
        model (Model): The model instance.
        batch_x (tf.Tensor): Input batch.
        batch_y (tf.Tensor): Ground truth batch.
        batch_x_m (tf.Tensor): Input mask.
        batch_y_m (tf.Tensor): Output mask.
        lrate (float): Learning rate.
        alpha_reg (float): Regularization factor.

    Returns:
        float: The loss value after training.
    """
    with tf.GradientTape() as tape:
        tf.print("\nIn model_train(): batch_x.shape is ", batch_x.shape)
        tf.print("\nIn model_train(): batch_x_m.shape is ", batch_x_m.shape)
        loss = model.compute_cost(batch_x, batch_y, batch_x_m, batch_y_m, alpha_reg)
        #loss = model.cost(
        #    batch_x, batch_y, batch_x_m, batch_y_m, training=True, alpha_reg=alpha_reg
        #)
    
    trainable_vars = model.trainable_variables
    grads = tape.gradient(loss, trainable_vars)
    model.trainer.apply_gradients(zip(grads, trainable_vars))
    
    return loss.numpy()
'''
