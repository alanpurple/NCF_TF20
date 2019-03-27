import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Input,Model,regularizers,metrics
import constants as rconst

import movielens

class Neumf(Model):
    def __init__(self,params):
        super(Neumf,self).__init__(name='Neural MF Model')
        num_users = params["num_users"]
        num_items = params["num_items"]
        batch_size = params["batch_size"]

        model_layers = params["model_layers"]

        mf_regularization = params["mf_regularization"]
        mlp_reg_layers = params["mlp_reg_layers"]

        mf_dim = params["mf_dim"]

        self.mf_slice=layers.Lambda(lambda x: x[:, :mf_dim])
        self.mlp_slice=layers.Lambda(lambda x: x[:, mf_dim:])

        self.reshape=layers.Reshape((mf_dim + model_layers[0] // 2,))

        # Initializer for embedding layers
        embedding_initializer = "glorot_uniform"

        self.user_embed=layers.Embedding(num_users, mf_dim + model_layers[0] // 2,
                                         embeddings_initializer=embedding_initializer,
                                         embeddings_regularizer=tf.keras.regularizers.l2(mf_regularization),
                                         input_length=1, name="embedding_user")
        self.item_embed=layers.Embedding(num_items,mf_dim + model_layers[0] // 2,
                                         embeddings_initializer=embedding_initializer,
                                         embeddings_regularizer=tf.keras.regularizers.l2(mf_regularization),
                                         input_length=1, name="embedding_item")

        num_layer=len(model_layers)


        self.model_layers=[]
        for layer in range(1,num_layer):
            model_layer=layers.Dense(model_layers[layer],'relu',
            kernel_regularizer=regularizers.l2(mlp_reg_layers[layer]))
            self.model_layers.append(model_layer)

        self.pred_layer=layers.Dense(1,kernel_initializer='lecun_uniform',
                                     name=movielens.RATING_COLUMN)

    def call(self,inputs):

        user_x=self.user_embed(inputs[movielens.USER_COLUMN])
        item_x=self.item_embed(inputs[movielens.ITEM_COLUMN])

        user_x=self.reshape(user_x)
        item_x=self.reshape(item_x)

        user_mf=self.mf_slice(user_x)
        item_mf=self.mf_slice(item_x)

        user_mlp=self.mlp_slice(user_x)
        item_mlp=self.mlp_slice(item_x)

        mf_vector=layers.multiply([user_mf,item_mf])
        mlp_vector=layers.concatenate([user_mlp,item_mlp])

        for layer in self.model_layers:
            mlp_vector=layer(mlp_vector)
        pred_vector=layers.concatenate([mf_vector,mlp_vector])
        logits= self.pred_layer(pred_vector)
        zeros = layers.Lambda(lambda x: x * 0)(logits)
        softmax_logits=layers.concatenate([zeros,logits])

        return softmax_logits



        


def _strip_first_and_last_dimension(x, batch_size):
  return tf.reshape(x[0, :], (batch_size,))


def compute_eval_loss_and_metrics_helper(logits,              # type: tf.Tensor
                                         softmax_logits,      # type: tf.Tensor
                                         duplicate_mask,      # type: tf.Tensor
                                         num_training_neg,    # type: int
                                        ):
  """Model evaluation with HR and NDCG metrics.

  The evaluation protocol is to rank the test interacted item (truth items)
  among the randomly chosen 999 items that are not interacted by the user.
  The performance of the ranked list is judged by Hit Ratio (HR) and Normalized
  Discounted Cumulative Gain (NDCG).

  For evaluation, the ranked list is truncated at 10 for both metrics. As such,
  the HR intuitively measures whether the test item is present on the top-10
  list, and the NDCG accounts for the position of the hit by assigning higher
  scores to hits at top ranks. Both metrics are calculated for each test user,
  and the average scores are reported.

  Specifically, if the evaluation negatives contain duplicate items, it will be
  treated as if the item only appeared once. Effectively, for duplicate items in
  a row, the predicted score for all but one of the items will be set to
  -infinity

  For example, suppose we have that following inputs:
  logits_by_user:     [[ 2,  3,  3],
                       [ 5,  4,  4]]

  items_by_user:     [[10, 20, 20],
                      [30, 40, 40]]

  # Note: items_by_user is not explicitly present. Instead the relevant \
          information is contained within `duplicate_mask`

  top_k: 2

  Then with match_mlperf=True, the HR would be 2/2 = 1.0. With
  match_mlperf=False, the HR would be 1/2 = 0.5. This is because each user has
  predicted scores for only 2 unique items: 10 and 20 for the first user, and 30
  and 40 for the second. Therefore, with match_mlperf=True, it's guaranteed the
  first item's score is in the top 2. With match_mlperf=False, this function
  would compute the first user's first item is not in the top 2, because item 20
  has a higher score, and item 20 occurs twice.

  Args:
    logits: A tensor containing the predicted logits for each user. The shape
      of logits is (num_users_per_batch * (1 + NUM_EVAL_NEGATIVES),) Logits
      for a user are grouped, and the last element of the group is the true
      element.

    softmax_logits: The same tensor, but with zeros left-appended.

    duplicate_mask: A vector with the same shape as logits, with a value of 1
      if the item corresponding to the logit at that position has already
      appeared for that user.

    num_training_neg: The number of negatives per positive during training.

  Returns:
    in_top_k: hit rate metric
  """

  return compute_top_k_and_ndcg(logits, duplicate_mask)


def compute_top_k_and_ndcg(logits,              # type: tf.Tensor
                           duplicate_mask,      # type: tf.Tensor
                          ):
  """Compute inputs of metric calculation.

  Args:
    logits: A tensor containing the predicted logits for each user. The shape
      of logits is (num_users_per_batch * (1 + NUM_EVAL_NEGATIVES),) Logits
      for a user are grouped, and the first element of the group is the true
      element.
    duplicate_mask: A vector with the same shape as logits, with a value of 1
      if the item corresponding to the logit at that position has already
      appeared for that user.
    match_mlperf: Use the MLPerf reference convention for computing rank.

  Returns:
    is_top_k, ndcg and weights, all of which has size (num_users_in_batch,), and
    logits_by_user which has size
    (num_users_in_batch, (rconst.NUM_EVAL_NEGATIVES + 1)).
  """
  logits_by_user = tf.reshape(logits, (-1, rconst.NUM_EVAL_NEGATIVES + 1))


  # Determine the location of the first element in each row after the elements
  # are sorted.
  sort_indices = tf.argsort(
      logits_by_user, axis=1, direction="DESCENDING")

  # Use matrix multiplication to extract the position of the true item from the
  # tensor of sorted indices. This approach is chosen because both GPUs and TPUs
  # perform matrix multiplications very quickly. This is similar to np.argwhere.
  # However this is a special case because the target will only appear in
  # sort_indices once.
  one_hot_position = tf.cast(tf.equal(sort_indices, rconst.NUM_EVAL_NEGATIVES),
                             tf.int32)
  sparse_positions = tf.multiply(
      one_hot_position, tf.range(logits_by_user.shape[1])[tf.newaxis, :])
  position_vector = tf.reduce_sum(sparse_positions, axis=1)

  in_top_k = tf.cast(tf.less(position_vector, rconst.TOP_K), tf.float32)



  return in_top_k