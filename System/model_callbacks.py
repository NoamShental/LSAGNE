from configuration import config
import keras


class ModelCallback(keras.callbacks.Callback):
    def __init__(self, data, model, tester, printer, loss_tracker):
        keras.callbacks.Callback.__init__(self)
        self.data = data
        self.model_handler = model
        self.tester = tester
        self.printer = printer
        self.loss_tracker = loss_tracker

    def on_epoch_end(self, epoch, logs={}):
        """
        Callback function, called on epoch end
        :param epoch: number of current epochs
        :param logs: dictionary with quantities relevant to current epochs
        """
        # Update reference points
        if epoch in config.config_map['reference_points_change']:
            self.data.update_reference_points_and_set_to_train_and_test()

        # log the losses per cloud
        self.loss_tracker.post_epoch_append_losses(epoch, logs)
