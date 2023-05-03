from system_manager import SystemManager
from post_tests import PostTests
from configuration import config, mock
import argparse
import string


def parse_arguments():
    """
    Parse command line arguments
    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Gene expression prediction and testing system",
                                     usage="%(prog)s [-h] --[run|post|find-kill] [more options]")

    # Set run mode mutually exclusive group
    run_modes_group = parser.add_mutually_exclusive_group(required=True)
    run_modes_group.add_argument('--run', help='if set, run normal test', action='store_true')
    run_modes_group.add_argument('--post', help='if set, run post processing tests', action='store_true')
    run_mode, remain_args = parser.parse_known_args()

    # Add option to each mode
    if run_mode.run:
        parser.usage = "%(prog)s [-h] --run -tumor test_tumor -pert test_pert -output test_output " + \
                       "[-organized-folder organized_foldere] [-start start_num -end end_num] " + \
                       "[-test-num test_number] [-whitelist-pert perturbations_whitelist] "
        parser.add_argument('-tumor', help='Tumor to leave out', dest='tumor', required=True)
        parser.add_argument('-pert', help='Perturbation to leave out', dest='pert', required=True)
        parser.add_argument('-output', help='Output folder for test', dest='output', required=True)
        parser.add_argument('-organized-folder', help='Organized data folder, as input folder', dest='organized')
        parser.add_argument('-start', help='start copy of test', dest='start', type=int)
        parser.add_argument('-end', help='End copy of test', dest='end', type=int)
        parser.add_argument('-test-num', help='Number of test', dest='test_num', type=int)
        parser.add_argument('-whitelist-pert', help='whitelist of perturbation, separated by a " "',
                            dest='whitelist_pert')

    # Post processing mode: run on tests and add more data
    elif run_mode.post:
        parser.usage = "%(prog)s [-h] --post -results results_path [-start-repeat int] [-end-repeat int] " + \
                       "[-organized-folder organized_folder]_[--one-pert] [--blackbox-test] [--perturbations-test]"
        parser.add_argument('-tumor', help='Tumor to leave out', dest='tumor', required=True)
        parser.add_argument('-pert', help='Perturbation to leave out', dest='pert', required=True)
        parser.add_argument('-output', help='Output folder for post process', dest='output', required=True)
        parser.add_argument('-start', help='Start repeat to do post tests', type=int)
        parser.add_argument('-end', help='End repeat to do post tests', type=int)
        parser.add_argument('-test-num', help='Number of test', dest='test_num', type=int)
        parser.add_argument('-organized-folder', help='Organized data folder, as input folder', dest='organized')
        parser.add_argument('-tcga-folder', help='Organized tcga folder, as input folder', dest='tcga_folder')
        parser.add_argument('-whitelist-pert', help='whitelist of perturbation, separated by a " "',
                            dest='whitelist_pert')

        parser.add_argument('--perturbations-test', help='If set, do perturbations post test',
                            dest='perturbations_test', action='store_true')
        parser.add_argument('--tcga', help='If set, run tcga tests', action='store_true')
        parser.add_argument('--extended-times', help='If set, run distances tests', action='store_true',
                            dest='extended_times')
        parser.add_argument('--semi-supervised', help='If set, run semi supervised tests', action='store_true',
                            dest='semi_supervised')
        parser.add_argument('--confusion-table', help='If set, create confusion tables between treated', action='store_true',
                            dest='confusion_table')
        parser.add_argument('--statistics-test', help="Run statistics calculator, 0 - don't run, 1: run regular, 2: run top 100", type=int,
                            default=0)
        parser.add_argument('--traj-test', help="If set, run trajectories tests", action='store_true', dest='traj_test')
        parser.add_argument('--random-traj-test', help="If set, run trajectories tests", action='store_true',
                            dest='random_traj_test')
        parser.add_argument('--drug-combinations', help="If set, find perts combination", action='store_true',
                            dest='drug_combination')
        parser.add_argument('--encode-samples', help="If set, encode set of samples", action='store_true',
                            dest='encode_samples')
        parser.add_argument('--decode-samples', help="If set, decode set of samples", action='store_true',
                            dest='decode_samples')

    args = parser.parse_args()
    return args


def _file_name_escaping(filename):
    """
    Make escaping for file names (i.e: omit '|' or '\'.
    :param filename: filename to escape
    :return: escaped filename
    """
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    return ''.join(c for c in filename if c in valid_chars)


def run(args):
    """
    Run the system in normal mode
    :param args: argparse object that contain the run configuration
    """
    config.config_map['leave_out_tissue_name'] = args.tumor
    config.config_map['leave_out_perturbation_name'] = args.pert
    config.config_map['root_output_folder'] = args.output
    if args.organized is not None:
        config.config_map['organized_data_folder'] = args.organized
    if args.whitelist_pert:
        whitelist = args.whitelist_pert.split(' ')
        config.config_map['run_perturbations_whitelist'] = whitelist

    start = 0
    end = 1
    test_num = 0
    if args.start is not None and args.end is not None:
        start = args.start
        end = args.end
    if args.test_num is not None:
        test_num = args.test_num
    for repeat in range(start, end):
        test_full_name = _file_name_escaping(str(repeat) + '_' + str(test_num) + '_' + args.tumor + '_' + args.pert)
        config.config_map['test_number'] = str(repeat) + '_' + str(test_num)

        manager = SystemManager(test_full_name)
        manager.run()


def post(args):
    """
    Apply post tests to already tested results
    :param args: argparse object that contain the run configuration
    """
    config.config_map['leave_out_tissue_name'] = args.tumor
    config.config_map['leave_out_perturbation_name'] = args.pert
    config.config_map['root_output_folder'] = args.output
    if args.organized is not None:
        config.config_map['organized_data_folder'] = args.organized
    if args.tcga_folder is not None:
        config.config_map['organized_tcga_folder'] = args.tcga_folder
    if args.whitelist_pert:
        whitelist = args.whitelist_pert.split(' ')
        config.config_map['run_perturbations_whitelist'] = whitelist
    start = 0
    end = 1
    test_num = 0
    if args.start is not None and args.end is not None:
        start = args.start
        end = args.end
    if args.test_num is not None:
        test_num = args.test_num
    for repeat in range(start, end):
        test_full_name = _file_name_escaping(str(repeat) + '_' + str(test_num) + '_' + args.tumor + '_' + args.pert)
        config.config_map['test_number'] = str(repeat) + '_' + str(test_num)
        post_tests = PostTests(test_full_name,
                               do_perturbations_tests=args.perturbations_test, do_tcga_tests=args.tcga,
                               do_extended_times_tests=args.extended_times,
                               do_semi_supervised_tests=args.semi_supervised, do_confusion_table=args.confusion_table,
                               do_statistics_tests=args.statistics_test, do_traj_test=args.traj_test,
                               do_random_traj_test=args.random_traj_test, do_drug_combination=args.drug_combination,
                               encode_samples=args.encode_samples, decode_samples=args.decode_samples)
        post_tests.run()


def main():
    args = parse_arguments()
    if args.run:
        run(args)
    elif args.post:
        post(args)
    else:
        print("ERROR, have to set at least 1 run configuration")


if __name__ == '__main__':
    main()
    #name = _file_name_escaping(config.config_map['leave_out_tissue_name'] + '_' + config.config_map['leave_out_perturbation_name'])
    #config.config_map['test_number'] = ''
    #manager = SystemManager(name, weights_path='d:\\model.h5', reference_points_path='d:\\reference_points.p')
    #manager = SystemManager(name)
    #manager.run()
