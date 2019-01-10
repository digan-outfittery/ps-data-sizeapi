import logging

import tensorflow as tf

import edward

log = logging.getLogger(__name__)

def ed_gaussian(x, fmax=0.5, loc=0., scale=1., min_val=0.02):
    return fmax * tf.exp(-(tf.square(x - loc)/(2 * tf.square(scale)))) + min_val


def ed_sigmoid(x, right_shift=0.4, compr=2.5, min=0.02, max=0.98):
    ''' function for a sigmoid-like activation '''
    x_int = compr * (x - right_shift)
    f_x = x_int / (tf.sqrt(1 + tf.square(x_int)))
    # now transform into the area between min and max (normal range is (-1, 1) )
    return f_x * (0.5 * max - 0.5 * min) + 0.5 * (min + max)


def make_edward_model(mu_cust_priors,
                      mu_item_priors,
                      customers,
                      items,
                      num_custs,
                      num_items,
                      obs_kept,
                      obs_toosmall,
                      obs_toobig,
                      gauss_max=0.7,
                      sigmoid_max=0.8,
                      right_shift=0.4):
    #  data placeholders - here the index column is fed in
    cust_ph = tf.placeholder(tf.int32, [None])
    item_ph = tf.placeholder(tf.int32, [None])

    cust_sizes_std = 1.
    item_sizes_std = 0.8

    # these are the prior distributions of our algorithm
    cust_sizes = edward.models.StudentT(df=2., loc=tf.ones(num_custs) * mu_cust_priors, scale=tf.ones(num_custs))
    item_sizes = edward.models.Normal(loc=tf.ones(num_items) * mu_item_priors, scale=tf.ones(num_items) * item_sizes_std)

    # gather maps the index (in the placeholder tensor) to the variable of the right customer/item)
    size_diff = tf.gather(item_sizes, item_ph) - tf.gather(cust_sizes, cust_ph)

    matching = edward.models.Bernoulli(probs=ed_gaussian(size_diff, fmax=gauss_max, scale=0.4))
    too_big = edward.models.Bernoulli(probs=ed_sigmoid(size_diff, compr=2.5, max=sigmoid_max, right_shift=right_shift))
    too_small = edward.models.Bernoulli(probs=ed_sigmoid(-size_diff, max=sigmoid_max, right_shift=right_shift))

    data = {
        cust_ph: customers,
        item_ph: items,
        matching: obs_kept,
        too_big: obs_toobig,
        too_small: obs_toosmall
    }

    # initialize posterior and give it a good starting point
    # (the q_* variables are the distributions [to optimize while running] we look at in the end)

    # set startingpoint: "prior values" as mean and a softplus of a noisy 1 as std
    q_cust_sizes = edward.models.Normal(
    loc=tf.Variable(mu_cust_priors),
    scale=tf.nn.softplus(tf.Variable(tf.random_normal([num_custs], mean=1.))))
    q_item_sizes = edward.models.Normal(
    loc=tf.Variable(mu_item_priors),
    scale=tf.nn.softplus(tf.Variable(tf.random_normal([num_items], mean=1.))))
    latent_variables = {
        cust_sizes: q_cust_sizes,
        item_sizes: q_item_sizes
    }
    inference = edward.KLqp(latent_variables, data)
    return (inference, {'cust_sizes': q_cust_sizes, 'item_sizes': q_item_sizes})


def make_edward_model_trousers(
            mu_cust_trousers_length_priors,
            mu_cust_trousers_width_priors,
            mu_item_trousers_length_priors,
            mu_item_trousers_width_priors,
            customers,
            items,
            num_custs,
            num_items,
            obs_kept,
            obs_toosmall,
            obs_toobig,
            gauss_max=0.3,
            sigmoid_max=0.9,
            right_shift=0.5):
    #  data placeholders - here the index column is fed in
    cust_ph = tf.placeholder(tf.int32, [None])
    item_ph = tf.placeholder(tf.int32, [None])

    cust_sizes_std = 2.0
    item_sizes_std = 1.0

    # these are the prior distributions of our algorithm
    cust_trousers_width = edward.models.Normal(loc=tf.ones(num_custs) * mu_cust_trousers_width_priors, scale=tf.ones(num_custs) * item_sizes_std)
    cust_trousers_length = edward.models.Normal(loc=tf.ones(num_custs) * mu_cust_trousers_length_priors, scale=tf.ones(num_custs) * item_sizes_std)
    item_trousers_width = edward.models.Normal(loc=tf.ones(num_items) * mu_item_trousers_width_priors, scale=tf.ones(num_items) * item_sizes_std)
    item_trousers_length = edward.models.Normal(loc=tf.ones(num_items) * mu_item_trousers_length_priors, scale=tf.ones(num_items) * item_sizes_std)

    # gather maps the index (in the placeholder tensor) to the variable of the right customer/item)
    diff_trousers_width = tf.gather(item_trousers_width, item_ph) - tf.gather(cust_trousers_width, cust_ph)
    diff_trousers_length = tf.gather(item_trousers_length, item_ph) - tf.gather(cust_trousers_length, cust_ph)

    matching = edward.models.Bernoulli(probs=ed_gaussian(
                tf.maximum(diff_trousers_width, diff_trousers_length), fmax=gauss_max, scale=1.0))
    too_big = edward.models.Bernoulli(probs=ed_sigmoid(
                tf.maximum(diff_trousers_width, diff_trousers_length), max=sigmoid_max, right_shift=2*right_shift))
    too_small = edward.models.Bernoulli(probs=ed_sigmoid(
                -tf.minimum(diff_trousers_width, diff_trousers_length), max=sigmoid_max, right_shift=2*right_shift))

    data = {
        cust_ph: customers,
        item_ph: items,
        matching: obs_kept,
        too_big: obs_toobig,
        too_small: obs_toosmall
    }

    # initialize posterior and give it a good starting point
    # (the q_* variables are the distributions [to optimize while running] we look at in the end)

    # startingpoint is the "prior values" as mean and a softplus of a noisy 1 as std
    q_cust_trousers_width = edward.models.Normal(
        loc=tf.Variable(mu_cust_trousers_width_priors),
        scale=tf.nn.softplus(tf.Variable(tf.random_normal([num_custs], mean=1.))))
    q_cust_trousers_length = edward.models.Normal(
        loc=tf.Variable(mu_cust_trousers_length_priors),
        scale=tf.nn.softplus(tf.Variable(tf.random_normal([num_custs], mean=1.))))
    q_item_trousers_width = edward.models.Normal(
        loc=tf.Variable(mu_item_trousers_width_priors),
        scale=tf.nn.softplus(tf.Variable(tf.random_normal([num_items], mean=1.))))
    q_item_trousers_length = edward.models.Normal(
        loc=tf.Variable(mu_item_trousers_length_priors),
        scale=tf.nn.softplus(tf.Variable(tf.random_normal([num_items], mean=1.))))
    latent_variables = {
        cust_trousers_width: q_cust_trousers_width,
        cust_trousers_length: q_cust_trousers_length,
        item_trousers_width: q_item_trousers_width,
        item_trousers_length: q_item_trousers_length,
    }
    inference = edward.KLqp(latent_variables, data)
    return (inference, {'cust_trouserswidth': q_cust_trousers_width,
                        'cust_trouserslength': q_cust_trousers_length,
                        'item_trouserswidth': q_item_trousers_width,
                        'item_trouserslength': q_item_trousers_length
                        })
