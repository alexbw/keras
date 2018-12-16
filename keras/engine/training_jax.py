"""Part of the training engine related to plain array data (e.g. Numpy).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as onp
import jax.numpy as np
from jax import grad
from jax import lax
from scipy.sparse import issparse

from .training_utils import batch_shuffle
from .training_utils import make_batches
from .training_utils import check_num_samples
from .training_utils import weighted_masked_objective
from .. import backend as K
from .. import callbacks as cbks
from ..utils.generic_utils import Progbar
from ..utils.generic_utils import slice_arrays
from ..utils.generic_utils import to_list
from ..utils.generic_utils import unpack_singleton

def _metrics_fn(model, outputs, targets, sample_weights=None, masks=None):
    """Calculates the metrics for each output of the given model.

    Arguments:
            model: The model on which metrics are being calculated.
            outputs: The outputs of the given model.
            targets: The predictions or targets of the given model.
            sample_weights: Optional list of sample weights for each output.
            masks: Optional list of masks for each output.

    Returns:
            Returns the metric results for each output of the model.
    """
    outputs = to_list(outputs)
    targets = to_list(targets)
    metric_results = model.handle_metrics(outputs, weights=sample_weights)
    return [K.mean(t) for t in metric_results]


def _model_loss(model, inputs, targets, sample_weights=None, training=False):
    """Calculates the loss for a given model.

    Arguments:
            model: The model on which metrics are being calculated.
            inputs: Either a dictionary of inputs to the model or a list of input
                arrays.
            targets: List of target arrays.
            sample_weights: Optional list of sample weight arrays.
            training: Whether the model should be run in inference or training mode.

    Returns:
         Returns the model output, total loss, loss value calculated using the
         specified loss function and masks for each output. The total loss includes
         regularization losses and applies masking and sample weighting
         to the loss value.
    """
    total_loss = 0
    kwargs = {}
    if model._expects_training_arg:
        kwargs['training'] = training
    if len(inputs) == 1 and not isinstance(inputs, dict):
        inputs = inputs[0]

    outs = model.call(inputs, **kwargs)
    masks = None

    outs = to_list(outs)
    if masks is None:
        masks = [None for _ in outs]
    targets = to_list(targets)

    loss_metrics = []
    with K.name_scope('loss'):
        for i, loss_fn in enumerate(model.loss_functions):
            if sample_weights:
                weights = sample_weights[i]
            else:
                weights = None
            mask = masks[i]

            weighted_masked_fn = weighted_masked_objective(loss_fn)
            with K.name_scope(model.output_names[i] + '_loss'):
                output_loss = weighted_masked_fn(
                        targets[i], outs[i], weights, mask=mask)
            # If the number of outputs is 1 then we don't append the loss metric
            # associated with each model output. When there are multiple outputs
            # associated with a model, each output's loss is calculated and returned
            # as part of the loss_metrics.
            if len(model.outputs) > 1:
                loss_metrics.append(K.mean(output_loss))

            loss_weight = model.loss_weights_list[i]
            if total_loss is None:
                total_loss = loss_weight * output_loss
            else:
                total_loss += loss_weight * output_loss

        total_loss = K.mean(total_loss)
        # Add regularization losses
        custom_losses = []
        for layer in model.layers:
            if layer.losses:
                custom_losses += layer.losses

        if custom_losses:
            total_loss += sum(custom_losses)

    return outs, total_loss, loss_metrics, masks


def _process_single_batch(model,
                            inputs,
                            targets,
                            sample_weights=None,
                            training=False):
    """Calculate the loss and gradient for one input batch.

         The model weights are updated if training is set to True.

    Arguments:
            model: Model whose loss has to be calculated.
            inputs: List of input arrays.
            targets: List of target arrays.
            sample_weights: Optional list of sample weight arrays.
            training: The boolean represents if the weights of the model are updated.
                            'fit' methods will set this to True while 'evaluate' methods will
                            set this to False.

    Returns:
            output of the model, total loss, the loss and the mask
            associated with each output.

    Raises:
            ValueError: If the model has no loss to optimize.
    """
    outs, loss, loss_metrics, masks = _model_loss(
            model,
            inputs,
            targets,
            sample_weights=sample_weights,
            training=training)
    if loss is None:
        raise ValueError('The model cannot be run '
                                         'because it has no loss to optimize.')
    if training:
        if not model._collected_trainable_weights:
            logging.warning('The list of trainable weights is empty. Make sure that'
                                            ' you are not setting model.trainable to False before '
                                            'compiling the model.')
        else:
            def f(collected_trainable_weights,
                    model,
                    inputs,
                    targets,
                    sample_weights=sample_weights,
                    training=training):
                outs, loss, loss_metrics, masks = _model_loss(
                        model,
                        inputs,
                        targets,
                        sample_weights=sample_weights,
                        training=training)
                return loss
            grads = grad(f)(model._collected_trainable_weights,
                model,
                inputs,
                targets,
                sample_weights=sample_weights,
                training=training)

            # Hide the gradients in a lambda
            K.gradients = lambda loss, params : grads

            # Apply the optimizer update (which calls `K.gradients()`
            # to retrieve gradients via `Optimizer.get_gradients()`)

            # Updates are applied in-place.
            training_updates = model.optimizer.get_updates(
                    params=model._collected_trainable_weights,
                    loss=model.total_loss)
    return outs, loss, loss_metrics, masks


def train_on_batch(model, inputs, targets, sample_weights=None):
    """Calculates the loss and gradient updates for one input batch.

    Arguments:
            model: Model whose loss has to be calculated.
            inputs: Input batch data.
            targets: Target batch data.
            sample_weights: Sample weight batch data.

    Returns:
            total loss and the loss associated with each output.
    """
    inputs = [np.array(val, dtype=K.floatx()) for val in inputs]
    targets = [np.array(val, dtype=K.floatx()) for val in targets]
    if sample_weights:
        sample_weights = [np.array(val, dtype=K.floatx())
                if val is not None else None for val in sample_weights]

    outs, loss, loss_metrics, masks = _process_single_batch(
            model, inputs, targets, sample_weights=sample_weights, training=True)
    if not isinstance(outs, list):
        outs = [outs]
    # metrics_results = _metrics_fn(
    #         model, outs, targets, sample_weights=sample_weights, masks=masks)
    metrics_results = []
    loss = to_list(loss)

    return loss + loss_metrics + metrics_results



def test_on_batch(model, inputs, targets, sample_weights=None):
    """Calculates the loss for one input batch.

    Arguments:
            model: Model whose loss has to be calculated.
            inputs: Input batch data.
            targets: Target batch data.
            sample_weights: Sample weight batch data.

    Returns:
            total loss, loss and metrics associated with each output.
    """
    inputs = [np.array(val, dtype=K.floatx()) for val in inputs]
    targets = [np.array(val, dtype=K.floatx()) for val in targets]
    if sample_weights:
        sample_weights = [np.array(val, dtype=K.floatx())
                if val is not None else None for val in sample_weights]

    outs, loss, loss_metrics, masks = _model_loss(
            model, inputs, targets, sample_weights=sample_weights, training=False)

    if not isinstance(outs, list):
        outs = [outs]
    metrics_results = _metrics_fn(
            model, outs, targets, sample_weights=sample_weights, masks=masks)
    loss = to_list(loss)

    return loss + loss_metrics + metrics_results


def fit_loop(model,
             inputs,
             targets,
             sample_weights=None,
             class_weight=None,
             val_inputs=None,
             val_targets=None,
             val_sample_weights=None,
             batch_size=None,
             epochs=1,
             verbose=1,
             callbacks=None,
             shuffle=True,
             initial_epoch=0,
             steps_per_epoch=None,
             validation_steps=None):
    """Abstract fit function for `fit_function(fit_inputs)`.

    Assumes that fit_function returns a list, labeled by out_labels.

    # Arguments
        model: Keras model instance.
        fit_function: Keras function returning a list of tensors
        fit_inputs: List of tensors to be fed to `fit_function`
        out_labels: List of strings, display names of
            the outputs of `fit_function`
        batch_size: Integer batch size or None if unknown.
        epochs: Number of times to iterate over the data
        verbose: Verbosity mode, 0, 1 or 2
        callbacks: List of callbacks to be called during training
        val_function: Keras function to call for validation
        val_inputs: List of tensors to be fed to `val_function`
        shuffle: Whether to shuffle the data at the beginning of each epoch
        callback_metrics: List of strings, the display names of the metrics
            passed to the callbacks. They should be the
            concatenation of list the display names of the outputs of
             `fit_function` and the list of display names
             of the outputs of `fit_inputs`.
        initial_epoch: Epoch at which to start training
            (useful for resuming a previous training run)
        steps_per_epoch: Total number of steps (batches of samples)
            before declaring one epoch finished and starting the
            next epoch. Ignored with the default value of `None`.
        validation_steps: Number of steps to run validation for
            (only if doing validation from data tensors).
            Ignored with the default value of `None`.

    # Returns
        `History` object.
    """
    do_validation = False
    if val_function and val_inputs:
        do_validation = True
        if (verbose and fit_inputs and
           hasattr(fit_inputs[0], 'shape') and hasattr(val_inputs[0], 'shape')):
            print('Train on %d samples, validate on %d samples' %
                  (fit_inputs[0].shape[0], val_inputs[0].shape[0]))
    if validation_steps:
        do_validation = True
        if steps_per_epoch is None:
            raise ValueError('Can only use `validation_steps` '
                             'when doing step-wise '
                             'training, i.e. `steps_per_epoch` '
                             'must be set.')
    elif do_validation:
        if steps_per_epoch:
            raise ValueError('Must specify `validation_steps` '
                             'to perform validation '
                             'when doing step-wise training.')

    num_train_samples = check_num_samples(fit_inputs,
                                          batch_size=batch_size,
                                          steps=steps_per_epoch,
                                          steps_name='steps_per_epoch')
    if num_train_samples is not None:
        index_array = np.arange(num_train_samples)

    model.history = cbks.History()
    _callbacks = [cbks.BaseLogger(
        stateful_metrics=model.stateful_metric_names)]
    if verbose:
        if steps_per_epoch is not None:
            count_mode = 'steps'
        else:
            count_mode = 'samples'
        _callbacks.append(
            cbks.ProgbarLogger(
                count_mode,
                stateful_metrics=model.stateful_metric_names))
    _callbacks += (callbacks or []) + [model.history]
    callbacks = cbks.CallbackList(_callbacks)
    out_labels = out_labels or []

    # it's possible to callback a different model than itself
    # (used by Sequential models)
    if hasattr(model, 'callback_model') and model.callback_model:
        callback_model = model.callback_model
    else:
        callback_model = model

    callbacks.set_model(callback_model)
    callbacks.set_params({
        'batch_size': batch_size,
        'epochs': epochs,
        'steps': steps_per_epoch,
        'samples': num_train_samples,
        'verbose': verbose,
        'do_validation': do_validation,
        'metrics': callback_metrics or [],
    })
    callbacks.on_train_begin()
    callback_model.stop_training = False
    for cbk in callbacks:
        cbk.validation_data = val_inputs

    # To prevent a slowdown,
    # we find beforehand the arrays that need conversion.
    feed = (model._feed_inputs +
            model._feed_targets +
            model._feed_sample_weights)
    indices_for_conversion_to_dense = []
    for i in range(len(feed)):
        if issparse(fit_inputs[i]) and not K.is_sparse(feed[i]):
            indices_for_conversion_to_dense.append(i)

    for epoch in range(initial_epoch, epochs):
        # Reset stateful metrics
        for m in model.stateful_metric_functions:
            m.reset_states()
        callbacks.on_epoch_begin(epoch)
        epoch_logs = {}
        if steps_per_epoch is not None:
            for step_index in range(steps_per_epoch):
                batch_logs = {}
                batch_logs['batch'] = step_index
                batch_logs['size'] = 1
                callbacks.on_batch_begin(step_index, batch_logs)
                outs = fit_function(fit_inputs)

                outs = to_list(outs)
                for l, o in zip(out_labels, outs):
                    batch_logs[l] = o

                callbacks.on_batch_end(step_index, batch_logs)
                if callback_model.stop_training:
                    break

            if do_validation:
                val_outs = test_loop(model, val_function, val_inputs,
                                     steps=validation_steps,
                                     verbose=0)
                val_outs = to_list(val_outs)
                # Same labels assumed.
                for l, o in zip(out_labels, val_outs):
                    epoch_logs['val_' + l] = o
        else:
            if shuffle == 'batch':
                index_array = batch_shuffle(index_array, batch_size)
            elif shuffle:
                np.random.shuffle(index_array)

            batches = make_batches(num_train_samples, batch_size)
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                try:
                    if isinstance(fit_inputs[-1], float):
                        # Do not slice the training phase flag.
                        ins_batch = slice_arrays(
                            fit_inputs[:-1], batch_ids) + [fit_inputs[-1]]
                    else:
                        ins_batch = slice_arrays(fit_inputs, batch_ids)
                except TypeError:
                    raise TypeError('TypeError while preparing batch. '
                                    'If using HDF5 input data, '
                                    'pass shuffle="batch".')
                batch_logs = {}
                batch_logs['batch'] = batch_index
                batch_logs['size'] = len(batch_ids)
                callbacks.on_batch_begin(batch_index, batch_logs)
                for i in indices_for_conversion_to_dense:
                    ins_batch[i] = ins_batch[i].toarray()

                outs = fit_function(ins_batch)
                outs = to_list(outs)
                for l, o in zip(out_labels, outs):
                    batch_logs[l] = o

                callbacks.on_batch_end(batch_index, batch_logs)
                if callback_model.stop_training:
                    break

                if batch_index == len(batches) - 1:  # Last batch.
                    if do_validation:
                        val_outs = test_loop(model, val_function, val_inputs,
                                             batch_size=batch_size,
                                             verbose=0)
                        val_outs = to_list(val_outs)
                        # Same labels assumed.
                        for l, o in zip(out_labels, val_outs):
                            epoch_logs['val_' + l] = o
        callbacks.on_epoch_end(epoch, epoch_logs)
        if callback_model.stop_training:
            break
    callbacks.on_train_end()
    return model.history

def test_loop(model, f, ins, batch_size=None, verbose=0, steps=None):
    """Abstract method to loop over some data in batches.

    # Arguments
        model: Keras model instance.
        f: Keras function returning a list of tensors.
        ins: list of tensors to be fed to `f`.
        batch_size: integer batch size or `None`.
        verbose: verbosity mode.
        steps: Total number of steps (batches of samples)
            before declaring predictions finished.
            Ignored with the default value of `None`.

    # Returns
        Scalar loss (if the model has a single output and no metrics)
        or list of scalars (if the model has multiple outputs
        and/or metrics). The attribute `model.metrics_names` will give you
        the display labels for the scalar outputs.
    """

    if hasattr(model, 'metrics'):
        for m in model.stateful_metric_functions:
            m.reset_states()
        stateful_metric_indices = [
            i for i, name in enumerate(model.metrics_names)
            if str(name) in model.stateful_metric_names]
    else:
        stateful_metric_indices = []

    num_samples = check_num_samples(ins,
                                    batch_size=batch_size,
                                    steps=steps,
                                    steps_name='steps')
    outs = []
    if verbose == 1:
        if steps is not None:
            progbar = Progbar(target=steps)
        else:
            progbar = Progbar(target=num_samples)

    # To prevent a slowdown,
    # we find beforehand the arrays that need conversion.
    feed = (model._feed_inputs +
            model._feed_targets +
            model._feed_sample_weights)
    indices_for_conversion_to_dense = []
    for i in range(len(feed)):
        if issparse(ins[i]) and not K.is_sparse(feed[i]):
            indices_for_conversion_to_dense.append(i)

    if steps is not None:
        for step in range(steps):
            batch_outs = f(ins)
            if isinstance(batch_outs, list):
                if step == 0:
                    for _ in enumerate(batch_outs):
                        outs.append(0.)
                for i, batch_out in enumerate(batch_outs):
                    if i in stateful_metric_indices:
                        outs[i] = float(batch_out)
                    else:
                        outs[i] += batch_out
            else:
                if step == 0:
                    outs.append(0.)
                outs[0] += batch_outs
            if verbose == 1:
                progbar.update(step + 1)
        for i in range(len(outs)):
            if i not in stateful_metric_indices:
                outs[i] /= steps
    else:
        batches = make_batches(num_samples, batch_size)
        index_array = np.arange(num_samples)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            if isinstance(ins[-1], float):
                # Do not slice the training phase flag.
                ins_batch = slice_arrays(ins[:-1], batch_ids) + [ins[-1]]
            else:
                ins_batch = slice_arrays(ins, batch_ids)
            for i in indices_for_conversion_to_dense:
                ins_batch[i] = ins_batch[i].toarray()

            batch_outs = f(ins_batch)
            if isinstance(batch_outs, list):
                if batch_index == 0:
                    for batch_out in enumerate(batch_outs):
                        outs.append(0.)
                for i, batch_out in enumerate(batch_outs):
                    if i in stateful_metric_indices:
                        outs[i] = batch_out
                    else:
                        outs[i] += batch_out * len(batch_ids)
            else:
                if batch_index == 0:
                    outs.append(0.)
                outs[0] += batch_outs * len(batch_ids)

            if verbose == 1:
                progbar.update(batch_end)
        for i in range(len(outs)):
            if i not in stateful_metric_indices:
                outs[i] /= num_samples
    return unpack_singleton(outs)


def predict_loop(model, f, ins, batch_size=32, verbose=0, steps=None):
    """Abstract method to loop over some data in batches.

    # Arguments
        model: Keras model instance.
        f: Keras function returning a list of tensors.
        ins: list of tensors to be fed to `f`.
        batch_size: integer batch size.
        verbose: verbosity mode.
        steps: Total number of steps (batches of samples)
            before declaring `predict_loop` finished.
            Ignored with the default value of `None`.

    # Returns
        Array of predictions (if the model has a single output)
        or list of arrays of predictions
        (if the model has multiple outputs).
    """
    num_samples = check_num_samples(ins,
                                    batch_size=batch_size,
                                    steps=steps,
                                    steps_name='steps')
    if verbose == 1:
        if steps is not None:
            progbar = Progbar(target=steps)
        else:
            progbar = Progbar(target=num_samples)

    indices_for_conversion_to_dense = []
    for i in range(len(model._feed_inputs)):
        if issparse(ins[i]) and not K.is_sparse(model._feed_inputs[i]):
            indices_for_conversion_to_dense.append(i)

    if steps is not None:
        # Step-based predictions.
        # Since we do not know how many samples
        # we will see, we cannot pre-allocate
        # the returned Numpy arrays.
        # Instead, we store one array per batch seen
        # and concatenate them upon returning.
        unconcatenated_outs = []
        for step in range(steps):
            batch_outs = f(ins)
            batch_outs = to_list(batch_outs)
            if step == 0:
                for batch_out in batch_outs:
                    unconcatenated_outs.append([])
            for i, batch_out in enumerate(batch_outs):
                unconcatenated_outs[i].append(batch_out)
            if verbose == 1:
                progbar.update(step + 1)
        if len(unconcatenated_outs) == 1:
            return np.concatenate(unconcatenated_outs[0], axis=0)
        return [np.concatenate(unconcatenated_outs[i], axis=0)
                for i in range(len(unconcatenated_outs))]
    else:
        # Sample-based predictions.
        outs = []
        batches = make_batches(num_samples, batch_size)
        index_array = np.arange(num_samples)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            if ins and isinstance(ins[-1], float):
                # Do not slice the training phase flag.
                ins_batch = slice_arrays(ins[:-1], batch_ids) + [ins[-1]]
            else:
                ins_batch = slice_arrays(ins, batch_ids)
            for i in indices_for_conversion_to_dense:
                ins_batch[i] = ins_batch[i].toarray()

            batch_outs = f(ins_batch)
            batch_outs = to_list(batch_outs)
            if batch_index == 0:
                # Pre-allocate the results arrays.
                for batch_out in batch_outs:
                    shape = (num_samples,) + batch_out.shape[1:]
                    outs.append(np.zeros(shape, dtype=batch_out.dtype))
            for i, batch_out in enumerate(batch_outs):
                outs[i] = lax.dynamic_update_slice(outs[i], batch_out, onp.array(batch_start))
            if verbose == 1:
                progbar.update(batch_end)
        return unpack_singleton(outs)
