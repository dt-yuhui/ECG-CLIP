import numpy as np
import torch


def make_surv_array(t, f, breaks):
    """Transforms censored survival data into vector format that can be used in Keras.
      Arguments
          t: Array of failure/censoring times.
          f: Censoring indicator. 1 if failed, 0 if censored.
          breaks: Locations of breaks between time intervals for discrete-time survival model (always includes 0)
      Returns
          Two-dimensional array of survival data, dimensions are number of individuals X number of time intervals*2
    """
    n_samples = t.shape[0]
    n_intervals = len(breaks) - 1
    time_gap = breaks[1:] - breaks[:-1]
    breaks_midpoint = breaks[:-1] + 0.5 * time_gap
    y_train = np.zeros((n_samples, n_intervals * 2))
    for i in range(n_samples):
        if f[i]:
            y_train[i, 0:n_intervals] = 1.0 * (t[i] >= breaks[1:])
            if t[i] < breaks[-1]:
                y_train[i, n_intervals + np.where(t[i] < breaks[1:])[0][0]] = 1
        else:
            y_train[i, 0:n_intervals] = 1.0 * (t[i] >= breaks_midpoint)
    return y_train

arr = make_surv_array(np.array([800]), np.array([0]), np.arange(0., 365. * 5, 365))[0]
print(arr[:4],arr[4:])


def surv_likelihood(n_intervals):
    """Create custom Pytorch loss function for neural network survival model.
    Arguments
        n_intervals: the number of survival time intervals
    Returns
        Custom loss function
    """

    def loss(y_pred, y_true):
        """
        Arguments
            y_true: Tensor.
                First half of the values is 1 if individual survived that interval, 0 if not.
                Second half of the values is for individuals who failed, and is 1 for time interval during which failure occured, 0 for other intervals.
                See make_surv_array function.
            y_pred: Tensor, predicted survival probability (1-hazard probability) for each time interval.
        Returns
            Vector of losses for this minibatch.
        """
        cens_uncens = 1. + y_true[:, 0:n_intervals] * (y_pred - 1.)  # component for all individuals
        uncens = 1. - y_true[:, n_intervals:2 * n_intervals] * y_pred  # component for only uncensored individuals
        concatenated = torch.cat((cens_uncens, uncens), dim=-1)
        clipped = torch.clamp(concatenated, min=torch.finfo(concatenated.dtype).eps)
        neg_log = -torch.log(clipped)
        return torch.sum(neg_log)

    return loss


def pred_surv(y_pred, breaks, fu_time):
    """
    Predicted survival probability from survival model
    Inputs are Numpy arrays.
    y_pred: Rectangular array, each individual's conditional probability of surviving each time interval
    breaks: Break-points for time intervals used for survival model, starting with 0
    fu_time: Follow-up time point at which predictions are needed

    Returns: predicted survival probability for each individual at specified follow-up time
    """
    y_pred = np.cumprod(y_pred, axis=1)
    pred_surv = []
    for i in range(y_pred.shape[0]):
        pred_surv.append(np.interp(fu_time, breaks[1:], y_pred[i, :]))
    return np.array(pred_surv)
