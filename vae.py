import tensorflow as tf 
import tensorflow.keras as keras
from tensorflow.keras import layers as kl


class _VAE(keras.Model):
    def __init__(self):
        super().__init__()
        self.reshape=False
        self.z_size=32 #洗剤空間の次元数

        #エンコーダ
        input_shape=(64,64,3)
        input_layer=c=kl.Input(shape=input_shape)

        c=kl.Conv2D(filerts=32,kernel_size=4,strides=2,activation='relu')(c)
        c=kl.Conv2D(filerts=64,kernel_size=4,strides=2,activation='relu')(c)
        c=kl.Conv2D(filerts=128,kernel_size=4,strides=2,activation='relu')(c)
        c=kl.Conv2D(filerts=256,kernel_size=4,strides=2,activation='relu')(c)
        c=kl.Flatten()(c)
        z_mean=kl.Dense(self.z_size)(c)
        z_logvar=kl.Dense(self.z_size)(c)

        self.encoder=keras.Model(input_layer,[z_mean,z_logvar])

        #デコーダ
        in_state=d=kl.Input(shape=(self.z_size,))
        d=kl.Dence(2*2*256,activation='relu')(d)
        d=kl.Reshape(1,1,2*2*256)(d)
        d=kl.Conv2DTranspose(filters=128,kernel_size=5,strides=2,activation='relu',padding='valid')(d)
        d=kl.Conv2DTranspose(filters=64,kernel_size=5,strides=2,activation='relu',padding='valid')(d)
        d=kl.Conv2DTranspose(filters=32,kernel_size=6,strides=2,activation='relu',padding='valid')(d)
        d=kl.Conv2DTranspose(filters=3,kernel_size=6,strides=2,activation='relu',padding='valid')(d)

        self.decoder=keras.Model(in_state,d)

    def reparameterize(self,z_mean,z_logvar):
        eps=keras.backend.random_normal(shape=(keras.backend.shape(z_mean)[0],self.z_size),mean=0.0,stddev=1.0)
        return z_mean+keras.backend.exp(z_logvar*0.5)*eps
    
    def call(self,inputs):
        if self.reshape:
            z_mean,z_logvar=self.encoder(inputs)
            z=self.reparameterize(z_mean,z_logvar)
        else:
            z=self.encoder(inputs)
        return self.decoder(z)
    
    def encode(self,inputs,training=False):
        z_mean,z_logvar=self.encoder(inputs,training=training)

        #reparameterize
        e=tf.random.normal(shape=z_mean.shape)
        z=z_mean+tf.exp(z_logvar*0.5)*e

        if training:
            return z_mean,z_logvar,z
        else:
            return z
        
    def decode(self,z,training=False):
        return self.decoder(z,training=training)
    
    def loss_function(self,x):
        z_mean, z_log_var, z = self.encode(x, training=True)
        pred_x = self.decode(z, training=True)

        # reconstruction loss (logistic), commented out.
        """
        eps = 1e-6  # avoid taking log of zero
        rc_loss = tf.reduce_mean(
            tf.reduce_sum(
                -(x * tf.math.log(pred_x + eps) + (1.0 - x) * tf.math.log(1.0 - pred_x + eps)),
                axis=[1, 2, 3],
            )
        )
        """

        # reconstruction loss (MSE)
        rc_loss = tf.reduce_sum(tf.square(x - pred_x), axis=[1, 2, 3])
        rc_loss = tf.reduce_mean(rc_loss)

        # KL loss
        kl_tolerance = 0.5
        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        kl_loss = tf.maximum(kl_loss, kl_tolerance * vae.z_size)
        kl_loss = tf.reduce_mean(kl_loss)

        loss = rc_loss + kl_loss
        return loss



        