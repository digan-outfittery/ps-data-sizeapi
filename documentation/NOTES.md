Sizemodel Development Notes (2017-2018)
===================================

The sizemodel is mostly developed within jupyter notebooks and tested with deja.
This file contains knowledge collected in this development.

The idea of the sizemodel is to combine the customer and item information and also get a measure of uncertainty. For this the bayesian probabilistic modelling approach seemed a good choice.


The ATL V1 model
-------------------------------
The original ATL size model used return information only to find out if an article was rather big or rather small in its specific size.
The model was trained on the labels "items_kept", "too_big" and "too_small". This means a beta distribution was used with parameters too_big vs the others and too_small vs the others to determine combinations of customer size and item size with a very high probability of returns due to size. The respective combination of customer size and item size was flagged red.
This approach worked well, but only for a very limited number of cases where the article labelling was the cause of the size problem and where we had a lot of observations. For some categories, for example shoes, there were never any red flags observed. Also for oversize items and for size combinations like S-XL there were never red flags.



The ATL V2 basic model
-----------------------------
The basic model is a statistical bayesian model stating that
  * customer_sizes are normally distributed
  * item sizes are normally distributed
  * -> starting values come from stated sizes in funnel/navision , std set between 0.5 and 1
  * probability of keeping articles is the value of a gaussian bell curve centered at 0, max height = 0.5 on the difference between customer size and item size. Here the observations of kepts/returns are used as input.

This is how this looks like in pymc3 code:

      cust_sizes = pm.Normal('cust_sizes', mu=mu_cust_priors, sd=0.6, shape=num_custs)
      item_sizes = pm.Normal('item_sizes', mu=mu_item_priors, sd=0.5, shape=num_articles)
      keep_proba = gaussian(item_sizes[items] - cust_sizes[customers], fmax=0.9, width=0.4)
      matching = pm.Bernoulli('matching', p=keep_proba, observed=obs)     

It turns out that this model has huge problems recognizing size problems because the return reason was not taken into account. We extend this model to learn also when articles are too big or too small by connecting these observations also to the difference of customer and item sizes, but with a sigmoid-shaped function:
$$
\overline{x} := c \cdot (x - rightshift)
$$
$$
sigmoid = \frac{\overline{x}}{\sqrt{1 + \overline{x}^2}}
$$

In pymc3 code this looks like:

    with pm.Model() as model:
        cust_sizes = pm.StudentT('cust_sizes', mu=mu_cust_priors, nu=2, lam=1, shape=num_custs)
        item_sizes = pm.Normal('item_sizes', mu=mu_item_priors, sd=1, shape=num_articles)
        diff = item_sizes[items] - cust_sizes[customers]
        keep_proba_gaussian = gaussian(diff, fmax=0.9, width=0.4)
        matching = pm.Bernoulli('matching', p=keep_proba_gaussian, observed=obs_kept)
        too_big = pm.Bernoulli('too_big', p=my_sigmoid(diff,compr=2.5), observed=obs_toobig)
        too_small = pm.Bernoulli('too_small', p=my_sigmoid(-diff), observed=obs_toosmall)

        return model

For this model the shirt sizes are translated into numeric values (0=XS to 8=5XL). For trousers width and length of the trousers are estimated in the same model. This is the model actually used in ATL.


Ideas that were tested, but abandoned
----------------------------------------

One idea was to add brand offsets. It was implemented as a variable changing the difference of customer and item sizes depending on the brand. The model could indeed learn size offsets for different brands, but the prediction quality did not improve substantially. This still could be an idea for better estimation on new articles.

Another idea was to add noise to the observations if an article was kept. The problem is that the distributions tend to shrink towards the discrete starting values of the stated sizes, just because this is where most observations are made. It turned out that this approach could not solve this problem though.

The attempt to predict shirt and trouser sizes in the same model failed mainly because it was much harder to train on this huge dataset and brought very little improvements.

We tried also to model the probability of a customer keeping an article individually and giving this as fmax parameter to the gaussian. This was intended to reduce the problem that bad customers got more often bad fit predictions because their size estimation was pushed away from the initial values if they did not keep anything, even if it was not due to size reasons.




Tested Frameworks:
----------------------------

  * pymc3
    * starting point, with standard sampling algorithms this is slow and consumes a lot of memory.
    * with ADVI (autodiff variational inference) we get decent performance.
    * unclear where the development is heading after theano (the tensor backend) is discontinued

  * Stan
    * (py)Stan needed a bit less to run, but the memory exploded in the end. Maybe this was due to pystan transferring data to python
    * since the performance was not much better this was abandoned.

  * Edwardlib
    * Runs with tensorflow as backend, is fast and seems the most memory efficient.
    * More possibilities to configure the model.
    * basically tested the klqp algorithm, sampling also available but slower.
    * do not need to explicitly sample to get the posterior distribution


Parameter Tuning / Optimization Notes
-------------------------------------------

The variable parameters that were not obvious by the modelling itself were tested using deja / quickreport. The following reports document the impact of these settings.

Training parameters 1:
https://ml-share-1.apps.outfittery.de/reports/2018-01-24-114458/
https://ml-share-1.apps.outfittery.de/reports/2018-01-24-142919/

Test the impact of preprocessing:
  * min_obs: is the minimum nr of observations for items to make it into the train set (if we don't set this we risk overfitting on items we sent out 2 times and both were returned...).
    * 3 or 5 seem to be reasonable choices, so take min_obs=3 or 4
  * size_feedback_offset: when estimating the starting point for the estimation we try to use the size-related feedback of the customers by counting there stated size +- size_feedback_offset like a kept.
    * it looks like slightly higher values than 1.0 are good choices (and this is the logical choice), so use 1.0 or 1.1


Prediction parameters 1:
https://ml-share-1.apps.outfittery.de/reports/2018-01-24-160359/

The prediction parameters show (as expected) that there is an negative correlation between number of flagged articles and effect size. Also First and Follow-on orders react different to the parameters: First Orders react strongly on min/max_mean_diff params, while Repeat/Club react strongly on the thresholds for the distribution width. Therefore we have to find good ways to determine the cutoff point.


Training parameters 2:
https://ml-share-1.apps.outfittery.de/reports/2018-01-26-180033/
https://ml-share-1.apps.outfittery.de/reports/2018-01-26-175720/
https://ml-share-1.apps.outfittery.de/reports/2018-01-26-182556/

The training parameters regarding the probabilistic model are the maximum height of the gaussian kept-probability (gauss_max) curve and of the sigmoid curve (sigmoid_max) and the offset of the strongest ascend of the sigmoid curve ("right_shift").

The rate of returns and of sizefeedback is not coherent in this case! Put more emphasis on sizefeedback to be on the safe side.
It seems that we really have the problem that customers with low kept rates in the past are "pushed away" from the item sizes the got sent, especially if the gauss_max is set to higher numbers than usual kept rates. I would therefore set this value to 0.3 to 0.4.
The max of the sigmoid functions can be set much higher without any problem, so 0.9 is a good choice. The "right_shift" parameter telling the algorithm at what point customers return the clothes as too small or too big has also the problem of shifting the full-return customers too much when set to values close to 1, therefore it is chosen to be 0.5.


Prediction Parameters 2:
https://ml-share-1.apps.outfittery.de/reports/2018-02-05-134526/
https://ml-share-1.apps.outfittery.de/reports/2018-02-05-134907/
https://ml-share-1.apps.outfittery.de/reports/2018-02-05-140428/

Suggestions:
min_mean_diff: 0.6
max_mean_diff: 2.0
customer_thresh: 2.0
item_thresh: 2.0 *(cust and item thresh is difficult: for repeat and club there seems to be only little increase in accuracy (esp for toobig) while we lose many orders by increasing the thresholds. If we want a lower discrimination rate we should increase both thresholds, preferably the item threshold a bit higher)*
trousers_multiple: 2.0 (seems to make no difference at all!)
