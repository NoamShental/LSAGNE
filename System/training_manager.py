from configuration import config
import logging
import numpy as np
import pandas as pd
import os
from tessellation_tester import TessellationTester
from keras.callbacks import CSVLogger, EarlyStopping
from sklearn.utils import class_weight


class TrainingManager:
    """
    This class is responsible to train the model
    """
    def __init__(self, data, model, model_callbacks):
        self.data = data
        self.model = model
        self.input_list = []
        self.output_list = []
        csv_logger = CSVLogger(os.path.join(model.models_folder, 'fit_log.csv'), append=True, separator=',')
        early_stopping = EarlyStopping(monitor='Pert_only_coliniarity_loss',
                                       patience=config.config_map['early_stopping_patience'], restore_best_weights=True,
                                       verbose=1)
        self.warm_up_callbacks = [csv_logger, model_callbacks]
        self.final_callbacks = [csv_logger, model_callbacks, early_stopping]
        self.create_model_io()
        self._create_treated_to_control_pert_time_cloud_dict()
        self.checkpoints = list()
        self.checkpoints.extend(config.config_map['epochs_of_filtering'])
        self.checkpoints.append(config.config_map['warmup_epochs'])
        if config.config_map['should_update_factors']:
            self.checkpoints.append(config.config_map['update_factors_delay'])
        self.checkpoints.sort()

        # Save start count of samples for each cloud
        self.dropped_samples_df = self.data.train_info_df[~self.data.train_info_df.perturbation.isin(config.config_map['untreated_labels'])].groupby(
            ['tumor', 'perturbation']).count()[['classifier_labels']]

        self.dropped_samples_df['start'] = self.dropped_samples_df['classifier_labels']
        self.dropped_samples_df.drop('classifier_labels', axis=1, inplace=True)
        self.dropped_samples_df['dropped'] = 0

    def _create_treated_to_control_pert_time_cloud_dict(self):
        """
        Create dictionary from treated clouds number to their perturbation time control cloud number
        """
        self.treated_to_control_pert_time_cloud_dict = {}
        info_df = self.data.info_df
        treated_clouds = info_df[
            ~info_df.perturbation.isin(config.config_map['untreated_labels'])].numeric_labels.unique()
        for cloud_number in treated_clouds:
            tumor = info_df[info_df.numeric_labels == cloud_number].iloc[0].tumor
            control_pert_time_number = info_df[
                (info_df.tumor == tumor) & (info_df.perturbation.isin(config.config_map['untreated_labels'])) & (
                    info_df.pert_time.isin(config.config_map['perturbation_times']))].iloc[0].numeric_labels
            self.treated_to_control_pert_time_cloud_dict[cloud_number] = control_pert_time_number

    def create_model_io(self):
        """
        Set input and output lists for model training
        """
        self.input_list = [self.data.train_data_df.values]
        self.input_list.extend(self.data.train_reference_points)
        selectors = self.data.train_selectors

        # If should use class weights - read the classified data and weight it
        if config.config_map['should_use_class_weights']:
            # Set weights for classifier
            classifier_selectors = selectors[1]
            data_that_classified = self.data.train_info_df.loc[classifier_selectors['calculate'] != 0]
            classified_classes = np.unique(data_that_classified.numeric_labels)
            all_classes = np.unique(self.data.train_info_df.numeric_labels)
            class_weights = class_weight.compute_class_weight('balanced',
                                                              classified_classes,
                                                              data_that_classified.numeric_labels)
            class_weights = class_weights * (self.data.train_info_df.shape[0] / data_that_classified.shape[0])
            for c in all_classes.tolist():
                if c in classified_classes:
                    index = np.where(classified_classes == c)
                    classifier_selectors[self.data.train_info_df.numeric_labels == c] = class_weights[index][0]
                else:
                    # In that case classifier_selectors are already 0 for those classes, don't do anything
                    pass

            # Set weights for collinearity
            collinearity_selectors = selectors[2]
            data_that_collinearity = self.data.train_info_df.loc[collinearity_selectors['calculate'] != 0]
            collinearity_classes = np.unique(data_that_collinearity.numeric_labels)
            class_weights = class_weight.compute_class_weight('balanced',
                                                              collinearity_classes,
                                                              data_that_collinearity.numeric_labels)
            class_weights = class_weights * (self.data.train_info_df.shape[0] / data_that_collinearity.shape[0])
            for c in all_classes.tolist():
                if c in collinearity_classes:
                    index = np.where(collinearity_classes == c)
                    collinearity_selectors[self.data.train_info_df.numeric_labels == c] = class_weights[index][0]
                else:
                    # In that case classifier_selectors are already 0 for those classes, don't do anything
                    pass

            # Set weights for distance
            distance_selectors = selectors[3]
            data_with_distance = self.data.train_info_df.loc[distance_selectors['calculate'] != 0]
            distance_classes = np.unique(data_with_distance.numeric_labels)
            class_weights = class_weight.compute_class_weight('balanced',
                                                              distance_classes,
                                                              data_with_distance.numeric_labels)
            class_weights = class_weights * (self.data.train_info_df.shape[0] / data_with_distance.shape[0])
            for c in all_classes.tolist():
                if c in distance_classes:
                    index = np.where(distance_classes == c)
                    distance_selectors[self.data.train_info_df.numeric_labels == c] = class_weights[index][0]
                else:
                    # In that case classifier_selectors are already 0 for those classes, don't do anything
                    pass

        self.input_list.extend(selectors)
        classifier_labels_np = np.array(self.data.train_info_df.one_hot_labels.values.tolist())
        mock_labels_np = np.zeros([classifier_labels_np.shape[0]], dtype=np.float64)
        self.output_list = [self.data.train_data_df.values, classifier_labels_np, mock_labels_np, mock_labels_np,
                            mock_labels_np, mock_labels_np, mock_labels_np]
        if self.model.other_pert_loss_layer is not None:
            self.output_list.append(mock_labels_np)

    def filter_out_points(self):
        """
        filter out points
        """
        info_df = self.data.train_info_df.copy()
        treated_info_df = info_df[~info_df.perturbation.isin(config.config_map['untreated_labels'])]
        selector_df = self.data.train_selectors[1].loc[treated_info_df.index]
        selector_df = selector_df[selector_df.calculate != 0]
        treated_info_df = treated_info_df.loc[selector_df.index]

        # Move all train data (include control and treated) to latent space, and fit the svm on that
        data_to_check_df = self.data.train_data_df
        selector_df = self.data.train_selectors[1].loc[data_to_check_df.index]
        selector_df = selector_df[selector_df.calculate != 0]
        data_to_check_df = data_to_check_df.loc[selector_df.index]
        latent_space_df = self.model.predict_latent_space(data_to_check_df)
        tessellation_tester = TessellationTester()
        tessellation_tester.fit(latent_space_df, info_df.loc[selector_df.index])
        treated_data_df = latent_space_df.loc[treated_info_df.index]

        # Get predictions of treated data, and choose who to drop according to prediction
        _, predictions_df = tessellation_tester.get_accuracy(treated_data_df, treated_info_df)
        samples_to_drop = predictions_df[predictions_df['real'] != predictions_df['predicted']]

        # If we don't have any samples to drop - return now
        if samples_to_drop.shape[0] == 0:
            logging.info('All treated samples predicted successfully, continue without dropping')
            return

        # Filter out samples that predicted wrong, but not as the treated's control pert time cloud
        # Disable SettingWithCopyWarning, because we want to make that copy (the samples_to_drop df)
        prev_warning_mode = pd.options.mode.chained_assignment
        pd.options.mode.chained_assignment = None
        samples_to_drop['control_cloud'] = samples_to_drop.apply(
             lambda row: self.treated_to_control_pert_time_cloud_dict[row['real']], axis=1)
        pd.options.mode.chained_assignment = prev_warning_mode
        samples_to_drop = samples_to_drop[samples_to_drop.predicted == samples_to_drop.control_cloud]

        logging.info('Dropping %d points because the svm was wrong about them',
                     samples_to_drop.shape[0])

        # Actually drop the samples
        if samples_to_drop.shape[0] > 0:
            # Save dropped samples
            dropped_samples = self.data.train_info_df.loc[samples_to_drop.index]
            current_dropped_samples_count = dropped_samples.groupby(
                ['tumor', 'perturbation']).count()['classifier_labels']
            self.dropped_samples_df.loc[
                current_dropped_samples_count.index, 'dropped'] =\
                self.dropped_samples_df.loc[current_dropped_samples_count.index].dropped + current_dropped_samples_count

            # Actually drop them
            self.data.drop_from_train_set(samples_to_drop.index)

            # Update model IO lists
            self.create_model_io()

    def fit(self):
        """
        Fit the model on the data
        """
        current_epoch = 0
        for checkpoint in self.checkpoints:
            self.model.full_model.fit(self.input_list, self.output_list, batch_size=config.config_map['batch_size'],
                                      verbose=1, epochs=checkpoint, initial_epoch=current_epoch,
                                      callbacks=self.warm_up_callbacks)
            if checkpoint in config.config_map['epochs_of_filtering']:
                logging.info('Stop fitting to filter samples')
                self.filter_out_points()
            if config.config_map['should_update_factors'] and checkpoint == config.config_map['update_factors_delay']:
                config.config_map['vae_loss_factor'] = config.config_map['updated_vae_loss_factor']
                config.config_map['log_xy_loss_factor'] = config.config_map['updated_log_xy_loss_factor']
                config.config_map['KL_loss_factor'] = config.config_map['updated_KL_loss_factor']
                config.config_map['classifier_loss_factor'] = config.config_map['updated_classifier_loss_factor']
                config.config_map['coliniarity_pert_and_time_loss_factor'] = config.config_map['updated_coliniarity_pert_and_time_loss_factor']
                config.config_map['coliniarity_pert_loss_factor'] = config.config_map['updated_coliniarity_pert_loss_factor']
                config.config_map['parallel_vectors_loss_factor'] = config.config_map['updated_parallel_vectors_loss_factor']
                config.config_map['distance_between_vectors_loss_factor'] = config.config_map['updated_distance_between_vectors_loss_factor']
                config.config_map['distance_from_reference_loss_factor'] = config.config_map['updated_distance_from_reference_loss_factor']
                config.config_map['collinearity_other_perts_loss_factor'] = config.config_map['updated_collinearity_other_perts_loss_factor']
                self.model.recompile_model()
            current_epoch = checkpoint
        logging.info('Finish warm up session')
        self.model.full_model.fit(self.input_list, self.output_list, batch_size=config.config_map['batch_size'],
                                  verbose=1, epochs=config.config_map['epochs'], initial_epoch=current_epoch,
                                  callbacks=self.final_callbacks)
        logging.info('Fitting finish')
        self.dropped_samples_df.to_csv(os.path.join(config.config_map['models_folder'], 'dropped_points.csv'))
