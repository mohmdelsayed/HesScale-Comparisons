"""
Define grid search over curvature and damping hyperparameters.
Generate scripts with jobs that can be run on a compute cluster.

Notes:
------
- Run scripts will be generated in ../grid_search_command_scripts per default
"""

import os
import pprint
from math import inf as INFINITY
from itertools import product

from deepobs.tuner import GridSearch
from deepobs import config as dobs_config

from .runners import BPOptimRunner


class GridSearchLocal(GridSearch):
    """
    A basic Grid Search tuner.
    """
    def __init__(self, optimizer_class, hyperparam_names, grid, ressources, runner):
        """
        Args:
            grid (dict): Holds the discrete values for each hyperparameter as lists.
        """
        super(GridSearch, self).__init__(optimizer_class, hyperparam_names, ressources, runner)
        self.__check_if_grid_is_valid(grid, ressources)
        self._grid = grid
        self._search_name = 'grid_search'

    @staticmethod
    def __check_if_grid_is_valid(grid, ressources):
        grid_size = len(list(product(*[values for values in grid.values()])))
        if grid_size > ressources:
            raise RuntimeError('Grid is too large for the available number of iterations.')

    def _sample(self):
        all_values = []
        all_keys = []
        for key, values in self._grid.items():
            all_values.append(values)
            all_keys.append(key)

        samples = []
        for sample in product(*all_values):
            sample_dict = {}
            for index, value in enumerate(sample):
                sample_dict[all_keys[index]] = value
            samples.append(sample_dict)

        return samples

    def generate_commands_script(self, testproblem, run_script, output_dir='./results', random_seed=42,
                                 generation_dir = './command_scripts', **kwargs):
        """
        Args:
            testproblem (str): Testproblem for which to generate commands.
            run_script (str): Name the run script that is used from the command line.
            output_dir (str): The output path where the execution results are written to.
            random_seed (int): The random seed for the tuning.
            generation_dir (str): The path to the directory where the generated scripts are written to.

        Returns:
            str: The relative file path to the generated commands script.

        """

        os.makedirs(generation_dir, exist_ok=True)
        file_path = os.path.join(generation_dir, 'jobs_' + self._optimizer_name + '_' + self._search_name + '_' + testproblem + '.txt')
        file = open(file_path, 'w')
        kwargs_string = self._generate_kwargs_format_for_command_line(**kwargs)
        self._set_seed(random_seed)
        params = self._sample()
        for sample in params:
            sample_string = self._generate_hyperparams_format_for_command_line(sample)
            file.write('python3 ' + run_script + ' ' + testproblem + ' ' + sample_string + ' --output_dir ' + output_dir + ' ' + kwargs_string + '\n')
        file.close()
        return file_path
    
class BPGridSearchBase():
    """Basic functionality to create runscripts for grid searches.

    Notes:
    ------
    - Need to add automatic creation of the python script that is executed.
    - Need to add creation of scripts for multiple batch sizes.
    """
    def __init__(self,
                 deepobs_problem,
                 optim_cls,
                 curv_hyperparams,
                 damping_hyperparams,
                 runner_cls,
                 output_dir="../grid_search",
                 generation_dir="../grid_search_command_scripts",
                 import_optim_from="torch.optim",
                 import_runner_from="deepobs.pytorch.runner"):
        # grid search jobs
        self.deepobs_problem = deepobs_problem
        self.optim_cls = optim_cls
        self.curv_hyperparams = curv_hyperparams
        self.damping_hyperparams = damping_hyperparams
        # script and results directories
        self.output_dir = output_dir
        self.generation_dir = generation_dir
        # for Python runscript
        self.runner_cls = runner_cls
        self.import_optim_from = import_optim_from
        self.import_runner_from = import_runner_from

    def run_sequentially(self, grid, rerun_best_setting=False, **kwargs):
        """Run the grid search sequentially without creating runscripts."""
        tuner = self.create_deepobs_tuner(grid)
        tuner.tune(self.deepobs_problem,
                   output_dir=self.output_dir,
                   rerun_best_setting=rerun_best_setting,
                   **kwargs)

    def get_runner_cls(self):
        return self.runner_cls

    def get_optim_cls(self):
        return self.optim_cls

    def get_hyperparams(self):
        """Names and types of all tunable hyperparameters."""
        return self._check_unique_and_combine(self.curv_hyperparams,
                                              self.damping_hyperparams)

    def get_output_dir(self):
        return self.output_dir

    def get_generation_dir(self):
        return self.generation_dir

    def get_deepobs_problem(self):
        return self.deepobs_problem

    def get_path(self):
        """Use DeepOBS convention."""
        return os.path.join(
            self.output_dir,
            self.get_path_appended_by_deepobs(),
        )

    def get_path_appended_by_deepobs(self):
        """Subdirectories appended to `self.output_dir` by DeepOBS."""
        return os.path.join(self._get_problem_path_appended_by_deepobs(),
                            self._get_optim_path_appended_by_deepobs())

    def _get_problem_path_appended_by_deepobs(self):
        return self.deepobs_problem

    def _get_optim_path_appended_by_deepobs(self):
        return self._get_optim_name()

    def create_deepobs_tuner(self, grid):
        return GridSearchLocal(self.optim_cls,
                          self.get_hyperparams(),
                          grid,
                          ressources=INFINITY,
                          runner=self.runner_cls)

    def create_runscript_multi_batch(self, DEFAULT_TEST_PROBLEMS_SETTINGS, grid, **kwargs):
        """Grid search over batch size on top."""
        script_str = ""
        
        # accumulate jobs for different batch sizes
        script_file = self.create_runscript(grid,
                                            batch_size=DEFAULT_TEST_PROBLEMS_SETTINGS[self.deepobs_problem]["batch_size"],
                                            num_epochs=DEFAULT_TEST_PROBLEMS_SETTINGS[self.deepobs_problem]["num_epochs"],
                                            **kwargs)

        with open(script_file, "r") as file:
            script_str += file.read()

        with open(script_file, "w") as file:
            file.write(script_str)

    def create_runscript(self, grid, **kwargs):
        """Write jobs with hyperaparams tuned by command line to file.

        Generate appropriate .py run file on the fly.
        """
        py_script = self._generate_python_script()
        tuner = self.create_deepobs_tuner(grid)
        script_path = tuner.generate_commands_script(
            self.deepobs_problem,
            py_script,
            output_dir=self.output_dir,
            generation_dir=self.generation_dir,
            **kwargs)
        print("[GRID-SEARCH] Run scripts for {} created in {}".format(
            self.output_dir, self.generation_dir))
        return script_path

    def _generate_python_script(self):
        self._try_create_generation_dir()
        py_name, py_path = self._get_python_script_name_and_path()
        with open(py_path, "w") as f:
            f.write(self._get_python_script_str())
        return py_name

    def _get_python_script_str(self):
        result = "# imports provided by import_statements\n"
        for line in self._get_import_statement():
            result += "{}\n".format(line)

        result += "\n# imports above have to import runner and optimizer\n"
        result += "runner_cls = {}\n".format(self._get_runner_name())
        result += "optim_cls = {}\n".format(self._get_optim_name())
        result += "runner_hyperparams = {}\n".format(
            self._hyperparams_as_str())

        result += "\n# build runner\n"
        result += "runner = runner_cls(optim_cls, runner_hyperparams)\n"

        # possibility to skip running the training
        result += "\n# arguments from command line\n"
        result += self._python_script_command_before_training()
        result += "command = 'python3 {}\'.format(' '.join(sys.argv))\n"
        result += "signal.signal(signal.SIGUSR1, terminate_signal)\n\n"
        result += "try:"
        result += "\n\tstart_time = time.time()"
        result += "\n\trunner.run()"
        # possibility to log successful run somewhere
        result += self._python_script_command_after_training_finished('\t', f"finished_{self._get_optim_name()}")
        result += "except Exception as e:\n"
        result += "\tlogger.error(e, exc_info=True)"
        result += self._python_script_command_after_training_failed('\t', f"failed_{self._get_optim_name()}")
        
        return result

    def _python_script_command_before_training(self):
        """Command that is run before training loop is started.

        The user might want to check whether the run was already
        executed and cancel before re-running.
        """
        return ""

    def _python_script_command_after_training_finished(self, c, filename):
        """Command that is run after training loop.

        Can be used to log somewhere that the run was successful.
        """
        return ""

    def _python_script_command_after_training_failed(self, c, filename):
        """Command that is run after training loop.

        Can be used to log somewhere that the run was successful.
        """
        return ""

    def _try_create_generation_dir(self):
        if not os.path.isdir(self.generation_dir):
            os.makedirs(self.generation_dir, exist_ok=True)

    def _get_python_script_name_and_path(self):
        filename = self._get_python_script_name()
        filepath = os.path.join(self.generation_dir, filename)
        return filename, filepath

    def _get_python_script_name(self):
        return "{}.py".format(self._get_optim_name())

    def _get_import_statement(self):
        import_statements = []

        import_statements.append("from {} import {}".format(
            self.import_optim_from, self._get_optim_name()))
        import_statements.append("from {} import {}".format(
            self.import_runner_from, self._get_runner_name()))
        import_statements.append("import signal")
        import_statements.append("import time")
        import_statements.append("import logging\nlogger = logging.Logger('catch_all')")
        import_statements.append("\ndef terminate_signal(signum, frame):\n\tprint('Exit signal: ', signum)\n\twith open('timeout.txt', 'a') as f:\n\t\tf.write(command+'\\n')\n\texit(0)")
        return import_statements

    def _get_optim_name(self):
        return self.optim_cls.__name__

    def _get_runner_name(self):
        return self.runner_cls.__name__

    def _check_unique_and_combine(self, dict1, dict2):
        self._validate_keys(dict1, dict2)
        return {**dict1, **dict2}

    def _validate_keys(self, dict1, dict2):
        if not self._have_unique_keys(dict1, dict2):
            keys1, keys2 = set(dict1.keys()), set(dict2.keys())
            overlap = keys1.intersection(keys2)
            raise ValueError(
                "Keys {} and {} must be unique. Non-unique: {}.".format(
                    keys1, keys2, overlap))

    @staticmethod
    def _have_unique_keys(dict1, dict2):
        key_list = list(dict1.keys()) + list(dict2.keys())
        key_set = set(key_list)
        return len(key_list) == len(key_set)

    def _hyperparams_as_str(self):
        hyper_params = self.get_hyperparams()
        del hyper_params["random_seed"]
        hyperparams_str = pprint.pformat(hyper_params)
        hyperparams_str = self._fix_data_type_formatting(hyperparams_str)
        return hyperparams_str

    @staticmethod
    def _fix_data_type_formatting(hyperparams_str):
        blank = ""
        remove = [r"<class '", r"'>"]
        for r in remove:
            hyperparams_str = hyperparams_str.replace(r, blank)
        return hyperparams_str


class BPGridSearch(BPGridSearchBase):
    """Grid search for experiments with BackPACK-Optim."""
    def __init__(self,
                 deepobs_problem,
                 optim_cls,
                 tune_curv,
                 tune_damping,
                 output_dir="../grid_search",
                 generation_dir="../grid_search_command_scripts"):
        """
        Parameters:
        -----------
        tune_curv: Tuning
           Tuning instance for the curvature scheme (see tuning.py)
        tune_damping: Tuning
           Tuning instance for the damping scheme (see tuning.py)
        """
        super().__init__(deepobs_problem,
                         optim_cls,
                         curv_hyperparams=tune_curv.get_hyperparams(),
                         damping_hyperparams=tune_damping.get_hyperparams(),
                         runner_cls=BPOptimRunner,
                         output_dir=output_dir,
                         generation_dir=generation_dir,
                         import_optim_from="bpoptim",
                         import_runner_from="bp_dops_integration.runners")
        self.tune_curv = tune_curv
        self.tune_damping = tune_damping

    def create_runscript_multi_batch(self, DEFAULT_TEST_PROBLEMS_SETTINGS, grid=None, **kwargs):
        """Generate script with runs over `grid` for multiple batch sizes.

        Use grid defined by `self.tune_curv` and `self.tune_damping` as
        default. Training hyperparameters (number of epochs, logging frequency,
        ...) can be handed over via `kwargs`.
        """
        if grid is None:
            grid = self._get_grid()
        return super().create_runscript_multi_batch(DEFAULT_TEST_PROBLEMS_SETTINGS, grid,
                                                    **kwargs)

    def create_runscript(self, grid=None, **kwargs):
        if grid is None:
            grid = self._get_grid()
        return super().create_runscript(grid, **kwargs)

    def run_sequentially(self, grid=None, rerun_best_setting=False, **kwargs):
        if grid is None:
            grid = self._get_grid()
        super().run_sequentially(grid,
                                 rerun_best_setting=rerun_best_setting,
                                 **kwargs)

    def _get_grid(self):
        curv_grid = self.tune_curv.get_grid()
        damping_grid = self.tune_damping.get_grid()
        return self._check_unique_and_combine(curv_grid, damping_grid)

    def get_grid_dim(self):
        grid = self._get_grid()
        dim = 1
        del grid["random_seed"]
        for _, values in grid.items():
            dim *= len(values)
        return dim

    def _python_script_command_after_training_finished(self, c, filename):
        result = "\n" + c + "# Write command to 'filename.txt'\n"
        result += c +"with open(" + "'" + filename + ".txt', 'a') as f:\n"
        result += c+"\tf.write(command+' '+str(time.time() - start_time)+'\\n')\n"
        return result

    def _python_script_command_after_training_failed(self, c, filename):
        result = "\n" + c + "# Write command to 'filename.txt'\n"
        result += c +"with open(" + "'" + filename + ".txt', 'a') as f:\n"
        result += c+"\tf.write(command+'\\n')\n"
        return result

    def _get_import_statement(self):
        return ["import sys"] + super()._get_import_statement()
