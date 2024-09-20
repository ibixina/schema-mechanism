import argparse
import itertools
import logging
import logging.config
import os
import subprocess
from pathlib import Path
from time import time
from typing import Iterable
from typing import Iterator
from typing import Protocol

logger = logging.getLogger('scripts')

PYTHON_EXEC = '/home/archimidiz/PycharmProjects/schema39/.venv/bin/python'  # check for python
SCRIPT = 'src/examples/multi_arm_bandits.py' # location of the program

SOURCE_DIRS = [
    'src',
]

EXPERIMENT_UID = str(int(time()))
OUTPUT_DIR = Path(os.path.join('local/experiments', EXPERIMENT_UID)) # location for the output


class TrialConfigurator(Protocol, Iterator[list[str]]):
    name: str


class SameStatsVariableMachinesTrialConfigurator(TrialConfigurator):
    """
        variable parameters: n_machines = {2, 4, ..., 2 ** max_exponent}
        fixed parameters: max(p_win) = 1.0, min(p_win) = 0.0, and expected_value(p_win) = 0.5
    """

    def __init__(self, base_run_args: list[str], min_exponent: int = 2, max_exponent: int = 5):
        self.base_run_args = base_run_args
        self.min_exponent = min_exponent
        self.max_exponent = max_exponent

        self.name = 'SameStatsVariableMachines'

        self._n_machines = None
        self._max_machines = 2 ** self.max_exponent

    def __iter__(self) -> Iterator[list[str]]:
        self._n_machines = 2 ** self.min_exponent
        return self

    def __next__(self) -> list[str]:
        if self._n_machines > self._max_machines:
            raise StopIteration

        # always have at least these 2 machines
        p_wins = [0.0, 1.0]

        # fill remaining machines, keeping max(p_win) = 1.0, min(p_win) = 0.0, and expected_value(p_win) = 0.5
        if self._n_machines > 2:
            p_win_increment = 1.0 / (self._n_machines - 2)
            p_wins.extend([p_win_increment / 2.0 + m * p_win_increment for m in range(self._n_machines - 2)])
            p_wins = sorted(p_wins)

        self._n_machines *= 2

        return self._generate_args(p_wins)

    def _generate_args(self, p_wins: Iterable[float]) -> list[str]:
        machine_args = itertools.chain.from_iterable([['-M', str(round(p_win, 4))] for p_win in p_wins])
        return [*self.base_run_args, *machine_args]


class NeedleInAHaystackTrialConfigurator(TrialConfigurator):
    """ Single "good" machine. All others have a very small chance to win.
        variable parameters: n_machines = {2, 4, ..., 2 ** max_exponent}
    """

    def __init__(self, base_run_args: list[str], min_exponent: int = 2, max_exponent: int = 5):
        self.base_run_args = base_run_args
        self.min_exponent = min_exponent
        self.max_exponent = max_exponent

        self.name = 'NeedleInAHaystack'

        self._n_machines = None
        self._max_machines = 2 ** self.max_exponent

    def __iter__(self) -> Iterator[list[str]]:
        self._n_machines = 2 ** self.min_exponent
        return self

    def __next__(self) -> list[str]:
        if self._n_machines > self._max_machines:
            raise StopIteration

        # fill remaining machines, keeping max(p_win) = 1.0, min(p_win) = 0.0, and expected_value(p_win) = 0.5
        p_wins = [0.1] * (self._n_machines - 1)
        p_wins.append(0.9)

        self._n_machines *= 2

        return self._generate_args(p_wins)

    def _generate_args(self, p_wins: Iterable[float]) -> list[str]:
        machine_args = itertools.chain.from_iterable([['-M', str(p_win)] for p_win in p_wins])
        return [*self.base_run_args, *machine_args]


DEFAULT_STEPS = 100
DEFAULT_RUNS_PER_TRIAL = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Multi-Armed Bandit Experiment Runner (Schema Mechanism)')
    parser.add_argument(
        '--steps_per_run', type=int, required=False, default=DEFAULT_STEPS,
        help=f'the number of time steps before terminating experiment (default: {DEFAULT_STEPS})'
    )
    parser.add_argument(
        '--runs_per_trial',
        metavar='N',
        type=int,
        required=False,
        default=DEFAULT_RUNS_PER_TRIAL,
        help=f'the number of runs executed per experimental trial (default: {DEFAULT_RUNS_PER_TRIAL})'
    )

    return parser.parse_args()


def config_env() -> None:
    # setup environment variables
    os.environ['PYTHONPATH'] = os.pathsep.join([os.environ['PYTHONPATH'], *SOURCE_DIRS])

    # create output subdirectory
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir()


def execute_run(run_id: int, run_args: list[str], output_path: Path) -> None:
    logger.info(f'beginning run {run_id}: args:{run_args}')

    # 'python src/examples/multi_arm_bandits.py -M 1.0 -M 0.75 -M 0.25 -M 0.0 --steps 25000'
    with output_path.open('a') as file:
        file.write(f'\n***** start of run {run_id} *****\n')
        file.flush()
        _ = subprocess.run([PYTHON_EXEC, SCRIPT, *run_args], text=True, check=True, stdout=file)
        file.write(f'\n***** end of run {run_id} *****\n')


def execute_trial(trial_configurator: TrialConfigurator, runs_per_trial: int) -> None:
    logger.info(f'beginning trial: {trial_configurator.name}')

    trial_output_path = OUTPUT_DIR / trial_configurator.name
    trial_output_path.mkdir(parents=True)

    for config_id, run_args in enumerate(trial_configurator):
        run_output_path = trial_output_path / f'config{config_id}_results.txt'

        with run_output_path.open('w') as file:
            file.write(f'***** trial configuration {config_id}: {run_args}')

        for run_id in range(runs_per_trial):
            execute_run(run_id, run_args, run_output_path)


def execute_experiment() -> None:
    # configure logger
    logging.config.fileConfig('config/logging.conf')

    # parse command line arguments
    args = parse_args()

    # configure environment variables
    config_env()

    trials = [
        SameStatsVariableMachinesTrialConfigurator(['--steps', str(args.steps_per_run)], min_exponent=2, max_exponent=3),
        NeedleInAHaystackTrialConfigurator(['--steps', str(args.steps_per_run)], min_exponent=3, max_exponent=4)
    ]

    for trial in trials:
        execute_trial(trial, args.runs_per_trial)


if __name__ == '__main__':
    execute_experiment()
