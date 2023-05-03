from configuration import config
import logging
import os
import pandas as pd


class EncodeDecode:

    """
    Create set of encoded-decoded
    """
    def __init__(self, do_decode):
        """
        Initialize the class
        :param do_decode: if set, create decoded data and not encoded data
        """
        self.do_decode = do_decode
        samples_path = config.config_map['encode_decode_samples_path']
        if samples_path:
            with open(samples_path) as f:
                self.samples = f.readlines()
                self.samples = [line.rstrip('\n') for line in self.samples]
        else:
            self.samples = None

    def run(self, test_name, data, model):
        """
        Save encode and decode for a given list of samples
        :param test_name: name of running test
        :param data: data handler
        :param model: model handler to use
        """
        logging.info("Starting drug combinations tests: %s", str(test_name))
        output_folder = os.path.join(config.config_map['output_folder'], 'EncodeDecode')

        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        data_df, info_df, reference_points = data.get_all_data_and_info()
        if self.samples is not None:
            data_df = data_df.loc[self.samples]
        encoded_df = model.predict_latent_space(data_df)
        if self.do_decode:
            decoded_df = model.predict_decoder(encoded_df)
            decoded_df.columns = data_df.columns
            all_genes_decoded_df = data.get_12k_data(decoded_df)
            all_genes_decoded_df.to_hdf(os.path.join(output_folder, "decoded_12K.h5"), "df")
            decoded_df = data.get_unscaled_data(decoded_df)
            decoded_df.to_hdf(os.path.join(output_folder, "decoded_977.h5"), "df")
        else:
            encoded_df.to_hdf(os.path.join(output_folder, "encoded.h5"), "df")

