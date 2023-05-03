from configuration import config
from itertools import cycle
import pandas as pd
import numpy as np
import os


class LossTracker:
    """
    This class can keep track of losses of the system
    """

    def __init__(self, data, tester, printer):
        """
        Initialize, set all parameters
        :param data: Data handler.
        :param tester: tester object.
        :param printer: Printer class
        """
        self.data = data
        self.tester = tester
        self.printer = printer

        self.losses_df = pd.DataFrame(
            columns=["loss", "Decoder_loss", "Classifier_loss", "Pert_and_time_coliniarity_loss",
                     "Pert_only_coliniarity_loss", "Parallel_loss", "Distance_between_vectors_loss",
                     "Distance_from_ref_point_loss"])

        # Build a DataFrame contains all the treated clouds names(=perturbation, tumor).
        self.unique_clouds_df = self.data.info_df.drop_duplicates(['perturbation', 'tumor'])[['perturbation', 'tumor']]

        # Extract the clouds participated in the current learning session,
        # and build cloud-->color dictionary.
        self.clouds_in_learning_session = []
        self.clouds_colors_dict = {}
        cycol = cycle('rgbcmyp')
        for cloud_id in range(self.unique_clouds_df.shape[0]):
            # Extract the current cloud.
            current_cloud = self.unique_clouds_df.iloc[cloud_id]
            current_cloud_name = current_cloud.tumor + '_' + current_cloud.perturbation
            self.clouds_in_learning_session.append(current_cloud_name)
            self.clouds_colors_dict[current_cloud_name] = next(cycol)

        self.clouds_colors_dict["All"] = "black"

        # Build loss_type-->color dictionary.
        self.training_losses = ["loss", "Decoder_loss", "Classifier_loss", "Pert_and_time_coliniarity_loss",
                                "Pert_only_coliniarity_loss", "Parallel_loss", "Distance_between_vectors_loss",
                                "Distance_from_ref_point_loss"]
        self.training_losses_colors_dict = {}
        cycol = cycle('krgbcmy')
        for loss_type in self.training_losses:
            self.training_losses_colors_dict[loss_type] = next(cycol)

        self.training_losses_colors_dict["distance_from_reference"] = "teal"

    def clean(self):
        """
        Clean all the losses history
        """
        self.losses_df.clear()

    def post_epoch_append_losses(self, epoch_num, losses):
        loss = pd.Series(losses)
        loss.name = epoch_num
        self.losses_df = self.losses_df.append(loss)

    def print_losses_history(self):
        """
        Create several figures, one for each loss, that shows the loss during epochs
        """
        epsilon = 1e-9  # for prevention log(0), which raise divide by zero warning.
        figure_name = config.config_map['test_number'] + '_losses.png'
        self.printer.losses_plot(np.log2(self.losses_df + epsilon),
                                    data_columns=self.training_losses,
                                    colors_dict=self.training_losses_colors_dict,
                                    figure_name=figure_name + '.png',
                                    output_directory=os.path.join(config.config_map['pictures_folder'], 'Losses'))
