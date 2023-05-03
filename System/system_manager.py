from data_handler import DataHandler
from model_handler import ModelHandler
from tester import Tester
from printer import PrintingHandler
from calculated_cloud_tester import CalculatedCloudTester
from model_callbacks import ModelCallback
from loss_tracker import LossTracker
from training_manager import TrainingManager
from configuration import config
from helper_functions import set_logger, set_output_folders
import keras.backend as K
import os
import logging
import json


class SystemManager:
    """
    This class will manage the whole system, and will be in charge of calling each component
    """
    def __init__(self, test_name, weights_path=None, reference_points_path=None):
        """
        Initialize - set all parameters
        :param test_name: name of current test
        :param weights_path: path with the model's weights to initialize the model from
        :param reference_points_path: path to saved reference points
        """

        self.test_name = test_name
        K.clear_session()
        set_output_folders(test_name, delete_old_folder=True)
        set_logger()

        self.data = DataHandler()
        self.model = ModelHandler(self.data.number_of_drugs, weights_path)
        self.printer = PrintingHandler(self.model, self.data)
        self.tester = Tester(self.model, self.data, self.printer)
        self.loss_tracker = LossTracker(self.data, self.tester, self.printer)
        self.model_callbacks = ModelCallback(self.data, self.model, self.tester, self.printer, self.loss_tracker)
        self.training_manager = TrainingManager(self.data, self.model, self.model_callbacks)
        self.cloud_tester = CalculatedCloudTester(self.data, self.model, self.tester, self.printer,
                                                  self.training_manager)

        # Change picking reference point function to be by center of cloud
        self.data.pick_reference_point_function = lambda samples_df:\
            DataHandler.find_reference_point_by_center_of_cloud(self.model.predict_latent_space, samples_df)
        if reference_points_path is None:
            self.data.update_reference_points_and_set_to_train_and_test()
        else:
            self.data.load_reference_points(reference_points_path)

        # Save the configuration to file
        configuration_file_path = os.path.join(config.config_map['output_folder'], 'configuration.txt')
        with open(configuration_file_path, 'w') as fp:
            json.dump(config.config_map, fp, indent=4, default=(lambda o: 'Unserializable'))

    def run_tests(self, run_sanity_tests, run_arithmetic_tests, label):
        """
        Run tests
        :param run_sanity_tests: boolean, if True than run sanity tests
        :param run_arithmetic_tests: boolean, if True than run arithmetic tests
        :param label: label of tests
        :return results dictionary
        """
        results = {}

        # Show all losses
        if run_sanity_tests:
            results['Losses'] = self.tester.do_sanity_tests(label)

        # Arithmetic tests, if needed
        if run_arithmetic_tests:
            results['Arithmetic'] = self.cloud_tester.test_latent_space_arithmetic(label)
        return results

    def run(self):
        """
        Run the system
        """

        # Log start of running
        logging.info("Starting: %s", str(self.test_name))

        # Start tests
        self.run_tests(False, False, "Start")

        # Fit the network
        logging.info('Start fitting')
        self.training_manager.fit()

        # Save model weights, before the tests
        self.model.save_network_weights()

        # Save final reference points
        self.data.save_reference_points()

        # End tests
        self.loss_tracker.print_losses_history()
        results = self.run_tests(True, True, "End")

        # Save the results to file
        results_file_path = os.path.join(config.config_map['output_folder'], 'results.json')
        with open(results_file_path, 'w') as fp:
            json.dump(results, fp, indent=4, default=(lambda o: 'Unserializable'))

        # Log end of test
        logging.info("Ending: %s", str(self.test_name))
