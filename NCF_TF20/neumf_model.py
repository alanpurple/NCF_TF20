import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Input,Model,regularizers

def neumf_model(Model):
    def __init__(self,params):
        num_users = params["num_users"]
        num_items = params["num_items"]

        model_layers = params["model_layers"]
        if len(model_layers)!=3:
            raise NotImplementedError('3 layers are allowed only.')

        mf_regularization = params["mf_regularization"]
        mlp_reg_layers = params["mlp_reg_layers"]

        mf_dim = params["mf_dim"]

        # Initializer for embedding layers
        embedding_initializer = "glorot_uniform"

        self.user_embed=layers.Embedding(num_users, mf_dim + model_layers[0] // 2)
        self.item_embed=layers.Embedding(num_items,mf_dim + model_layers[0] // 2)

        self.mf_user_latent=layers.Lambda(lambda x: x[:, :mf_dim])
        self.mf_item_latent=layers.Lambda(lambda x: x[:, :mf_dim])

        self.mlp_user_latent=layers.Lambda(lambda x: x[:, mf_dim:])
        self.mlp_item_latent=layers.Lambda(lambda x: x[:, mf_dim:])

        num_layer=len(model_layers)

        self.model_layers=[]
        for layer in range(num_layer):
            model_layer=layer.Dense(model_layers[layer],'relu',
            kernel_regularizer=regularizers.l2(mlp_reg_layers[layer]))
            self.model_layers.append(model_layer)

        self.pred_layer=layers.Dense(1)

    def call(self,user_inputs,item_inputs):
        user_x=self.user_embed(user_inputs)
        item_x=self.item_embed(item_inputs)

        user_mf=self.mf_user_latent(user_x)
        item_mf=self.mf_item_latent(item_x)

        user_mlp=self.mlp_user_latent(user_x)
        item_mlp=self.mlp_item_latent(item_x)

        mf_vector=layers.multiply([user_mf,item_mf])
        mlp_vector=layers.concatenate([user_mlp,item_mlp])



        


def _strip_first_and_last_dimension(x, batch_size):
  return tf.reshape(x[0, :], (batch_size,))