import argparse
import logging.config
import typing
from collections import Counter
from collections.abc import Collection
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from functools import partial
from pathlib import Path
from statistics import mean
from typing import Iterable
from typing import Optional

from examples import EpisodeSummary
from examples import RANDOM_SEED
from examples import Runner
from examples import SaveCallback
from examples import display_item_values
from examples import display_known_schemas
from examples import display_schema_memory
from examples import parse_optimizer_args
from examples import parse_run_args
from examples import parser_serialization_args
from examples import run_episode
from examples import save_every_n_steps
from examples.environments import Environment
from examples.optimizers import get_optimizer_report_filename
from examples.optimizers import optimize
from schema_mechanism.core import Action
from schema_mechanism.core import Item
from schema_mechanism.core import State
from schema_mechanism.core import get_global_params
from schema_mechanism.func_api import sym_item
from schema_mechanism.func_api import sym_state
from schema_mechanism.modules import SchemaMechanism
from schema_mechanism.modules import SelectionDetails
from schema_mechanism.modules import init
from schema_mechanism.modules import load
from schema_mechanism.modules import save
from schema_mechanism.parameters import GlobalParams
from schema_mechanism.serialization import DEFAULT_ENCODING
from schema_mechanism.strategies.evaluation import DefaultEvaluationStrategy
from schema_mechanism.strategies.evaluation import DefaultExploratoryEvaluationStrategy
from schema_mechanism.strategies.evaluation import DefaultGoalPursuitEvaluationStrategy
from schema_mechanism.strategies.evaluation import DefaultGreedyEvaluationStrategy
from schema_mechanism.util import get_random_seed
from schema_mechanism.util import rng
from schema_mechanism.util import set_random_seed

logger = logging.getLogger('examples.environments.multi_arm_bandits')

# For reproducibility, we we also need to set PYTHONHASHSEED=RANDOM_SEED in the environment
set_random_seed(RANDOM_SEED)

# global constants
N_MACHINES = 16
N_STEPS = 8000


class Machine:
    def __init__(self, id_: str, p_win: float) -> None:
        self.id = id_
        self.p_win = p_win

        self._outcomes = ['L', 'W']
        self._weights = [1.0 - p_win, p_win]
        self._rng = rng()

    def __str__(self) -> str:
        return f'M{self.id}'

    def __repr__(self) -> str:
        return f'{self}[p_win={self.p_win:.2}]'

    def play(self, count: int = 1) -> Sequence[str]:
        return self._rng.choice(self._outcomes, size=count, p=self._weights)


@dataclass
class BanditEnvironmentEpisodeSummary(EpisodeSummary):
    winnings: int = 0
    times_won: int = 0
    times_played: int = 0
    states_visited: Optional[Counter] = field(default_factory=lambda: Counter())
    machines_played: Optional[Counter] = field(default_factory=lambda: Counter())

    @property
    def win_percentage(self) -> float:
        return self.times_won / self.times_played


class BanditEnvironment(Environment):
    DEFAULT_INIT_STATE = sym_state('S')  # standing

    def __init__(self,
                 machines: Sequence[Machine],
                 currency_to_play: int = 1,
                 currency_on_win: int = 2,
                 init_state: Optional[State] = None) -> None:
        super().__init__()

        if not machines:
            raise ValueError('Must supply at least one machine')

        self._id = 'multi-arm-bandits'
        self.machines: Sequence[Machine] = list(machines)
        self.currency_to_play = currency_to_play
        self.currency_on_win = currency_on_win

        self._actions = [Action(a_str) for a_str in ['deposit', 'stand', 'play']]

        self._sit_actions = [Action(f'sit({machine})') for machine in self.machines]
        self._actions.extend(self._sit_actions)

        self.states = [
            sym_state('W'),  # agent's last play won
            sym_state('L'),  # agent's last play lost
            sym_state('S'),  # agent is standing
            sym_state('P'),  # money is deposited in current machine
        ]

        # add machine related states
        self._machine_base_states = [sym_state(str(machine)) for machine in self.machines]
        self._machine_play_states = [sym_state(f'{machine},P') for machine in self.machines]
        self._machine_win_states = [sym_state(f'{machine},W') for machine in self.machines]
        self._machine_lose_states = [sym_state(f'{machine},L') for machine in self.machines]

        self.states.extend([
            *self._machine_base_states,
            *self._machine_play_states,
            *self._machine_win_states,
            *self._machine_lose_states
        ])

        self._init_state = init_state or sym_state('S')
        self._current_state = self._init_state

        if self._init_state not in self.states:
            raise ValueError(f'initial state is an invalid state: {self._init_state}')

        self._episode_summary = BanditEnvironmentEpisodeSummary()

    @property
    def id(self) -> str:
        return self._id

    @property
    def actions(self) -> Iterable[Action]:
        return self._actions

    @property
    def episode_summary(self) -> BanditEnvironmentEpisodeSummary:
        return self._episode_summary

    @property
    def is_terminal(self) -> bool:
        # currently, this environment is not episodic, so is_terminal always returns false. A future iteration
        # of the environment could make it episodic; for example, if the agent has a starting budget, that, if
        # exhausted, would end the episode.
        return False

    def step(self, action: Action) -> tuple[State, bool]:
        if action not in self._actions:
            raise ValueError(f'Invalid action: {action}')

        # standing
        if self._current_state == sym_state('S'):
            # sit at machine
            if action in self._sit_actions:
                m_ndx = self._sit_actions.index(action)
                self._current_state = self._machine_base_states[m_ndx]

        # at machine
        elif self._current_state in self._machine_base_states:
            if action == Action('deposit'):
                self._episode_summary.winnings -= self.currency_to_play

                m_ndx = self._machine_base_states.index(self._current_state)
                self._current_state = self._machine_play_states[m_ndx]
            elif action == Action('stand'):
                self._current_state = sym_state('S')

        # at machine, coin deposited
        elif self._current_state in self._machine_play_states:
            if action == Action('play'):
                m_ndx = self._machine_play_states.index(self._current_state)

                if self.machines[m_ndx].play()[0] == 'W':
                    self._episode_summary.times_won += 1
                    self._episode_summary.winnings += self.currency_on_win

                    self._current_state = self._machine_win_states[m_ndx]
                else:
                    self._current_state = self._machine_lose_states[m_ndx]

                self._episode_summary.times_played += 1
                self._episode_summary.machines_played[m_ndx] += 1

            elif action == Action('stand'):
                self._current_state = sym_state('S')

        # at machine, just won
        elif self._current_state in self._machine_win_states:
            m_ndx = self._machine_win_states.index(self._current_state)
            if action == Action('deposit'):
                self._episode_summary.winnings -= self.currency_to_play
                self._current_state = self._machine_play_states[m_ndx]
            elif action == Action('stand'):
                self._current_state = sym_state('S')
            else:
                self._current_state = self._machine_base_states[m_ndx]

        # at machine, just lost
        elif self._current_state in self._machine_lose_states:
            m_ndx = self._machine_lose_states.index(self._current_state)
            if action == Action('deposit'):
                self._episode_summary.winnings -= self.currency_to_play
                self._current_state = self._machine_play_states[m_ndx]
            elif action == Action('stand'):
                self._current_state = sym_state('S')
            else:
                self._current_state = self._machine_base_states[m_ndx]

        self._episode_summary.steps += 1
        self._episode_summary.states_visited[self._current_state] += 1

        return self._current_state, self.is_terminal

    def reset(self) -> tuple[State, bool]:
        self._episode_summary = BanditEnvironmentEpisodeSummary()
        self._current_state = self._init_state

        return self._current_state, self.is_terminal

    def render(self) -> str:
        return f'{self._current_state} [{self._episode_summary.winnings}]'


def parse_env_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--n_machines', type=int, metavar='N', required=False, default=N_MACHINES,
                        help=f'the id of the agent to which this action will be sent (default: {N_MACHINES})')
    parser.add_argument('-M', '--machine', metavar='MACHINE', dest='machines',
                        type=float, action='append', required=False, default=None,
                        help=f'specifies a new machine with a specific win probability (e.g., \'-M 0.75\')')

    return parser


def parse_args():
    """ Parses command line arguments.
    :return: argparse parser with parsed command line args
    """
    parser = argparse.ArgumentParser(description='Multi-Armed Bandit Example Environment (Schema Mechanism)')

    parser = parse_run_args(parser)
    parser = parse_env_args(parser)
    parser = parse_optimizer_args(parser)
    parser = parser_serialization_args(parser)

    return parser.parse_args()


def init_params() -> GlobalParams:
    params = get_global_params()

    params.set('learning_rate', 0.001)

    params.set('composite_actions.update_frequency', 0.01)
    params.set('composite_actions.backward_chains.max_length', 3)
    params.set('composite_actions.min_baseline_advantage', 0.4)

    params.set('schema.reliability_threshold', 0.8)

    # item correlation test used for determining relevance of extended context items
    params.set('ext_context.correlation_test', 'DrescherCorrelationTest')

    # thresholds for determining the relevance of extended result items
    #     from 0.0 [weakest correlation] to 1.0 [strongest correlation]
    params.set('ext_context.positive_correlation_threshold', 0.9)

    # item correlation test used for determining relevance of extended result items
    params.set('ext_result.correlation_test', 'CorrelationOnEncounter')

    # thresholds for determining the relevance of extended result items
    #     from 0.0 [weakest correlation] to 1.0 [strongest correlation]
    params.set('ext_result.positive_correlation_threshold', 1.0)

    return params


def load_schema_mechanism(model_filepath: Path) -> SchemaMechanism:
    schema_mechanism: Optional[SchemaMechanism] = None
    try:
        schema_mechanism = load(model_filepath)
    except FileNotFoundError as e:
        logger.error(e)

    return schema_mechanism


def create_schema_mechanism(env: Environment, primitive_items: Iterable[Item]) -> SchemaMechanism:
    schema_mechanism = init(
        items=primitive_items,
        actions=env.actions,
        global_params=init_params()
    )

    schema_mechanism.schema_selection.evaluation_strategy = DefaultEvaluationStrategy(
        goal_pursuit_strategy=DefaultGoalPursuitEvaluationStrategy(
            reliability_max_penalty=0.6,
            pending_focus_max_value=0.9,
            pending_focus_decay_rate=0.4
        ),
        exploratory_strategy=DefaultExploratoryEvaluationStrategy(
            epsilon=0.9999,
            epsilon_min=0.025,
            epsilon_decay_rate=0.9999,
        ),
    )

    return schema_mechanism


def calculate_score(episode_summary: BanditEnvironmentEpisodeSummary) -> float:
    return episode_summary.winnings


def display_n_schemas(schema_mechanism: SchemaMechanism, step: int, **kwargs) -> None:
    logger.info(f'known schemas [step: {step}]: {len(schema_mechanism.schema_memory)}')


def display_score(env: Environment, step: int, **kwargs) -> None:
    logger.info(f'score [step: {step}]: {calculate_score(env.episode_summary)}')


def display_weights(schema_mechanism: SchemaMechanism, step: int, **kwargs) -> None:
    weights: Optional[Sequence[float]] = None
    try:
        weights = schema_mechanism.schema_selection.evaluation_strategy.weights
    except AttributeError as e:
        pass

    logger.info(f'weights [step: {step}]: {weights} ')


def display_state(state: State, **kwargs):
    logger.info(f'state: {state}')


def display_action(selection_details: Optional[SelectionDetails], **kwargs):
    logger.info(f'action: {selection_details.selected.action}')


def display_combined_step(step: int, env: Environment, schema_mechanism: SchemaMechanism, **kwargs):
    n_known_schemas = len(schema_mechanism.schema_memory)
    score = calculate_score(env.episode_summary)
    weights = schema_mechanism.schema_selection.evaluation_strategy.weights

    logger.info(f'{step},{score},{n_known_schemas},{weights[0]}')


def display_environment_summary(env: BanditEnvironment):
    logger.info(f'Environment Summary ({env.id}):')

    logger.info(f'\tRandom Seed: {get_random_seed()}')
    logger.info(f'\tMachines:')
    for machine in env.machines:
        logger.info(f'\t\t{repr(machine)}')

    logger.info(f'\tMean Probability of Win: {mean(machine.p_win for machine in env.machines)}')


def display_performance_summary(episode_summaries: Collection[BanditEnvironmentEpisodeSummary]) -> None:
    logger.info(f'Performance Summary:')

    steps_in_episode = [episode_summary.steps for episode_summary in episode_summaries]
    winnings_in_episode = [episode_summary.winnings for episode_summary in episode_summaries]
    times_won_in_episode = [episode_summary.times_won for episode_summary in episode_summaries]
    times_played_in_episode = [episode_summary.times_played for episode_summary in episode_summaries]
    machines_played_in_episode = [episode_summary.machines_played for episode_summary in episode_summaries]
    states_visited_in_episode = [episode_summary.states_visited for episode_summary in episode_summaries]

    cumulative_steps = sum(steps_in_episode)

    total_times_played = sum(times_played_in_episode)
    total_times_won = sum(times_won_in_episode)

    win_percentage = 0.0 if total_times_played == 0 else (total_times_won / total_times_played) * 100.0

    avg_winnings_per_episode = mean(winnings_in_episode)
    min_winnings_per_episode = min(winnings_in_episode)
    max_winnings_per_episode = max(winnings_in_episode)

    # aggregate episode specific counters
    machines_played_all_episodes = Counter()
    for counter in machines_played_in_episode:
        machines_played_all_episodes.update(counter)

    states_visited_all_episodes = Counter()
    for counter in states_visited_in_episode:
        states_visited_all_episodes.update(counter)

    logger.info(f'\t\tEpisodes: {len(episode_summaries)}')
    logger.info(f'\t\tSteps (over all episodes): {cumulative_steps}')
    logger.info(f'\t\tAverage winnings: {avg_winnings_per_episode}')
    logger.info(f'\t\tMinimum winnings: {min_winnings_per_episode}')
    logger.info(f'\t\tMaximum winnings: {max_winnings_per_episode}')
    logger.info(f'\t\tWin percentage (over all episodes): {win_percentage:.2f}%')
    logger.info(f'\t\tMachines played (over all episodes): {machines_played_all_episodes}')
    logger.info(f'\t\tStates visited (over all episodes): {states_visited_all_episodes}')


def create_machines(n: int, win_rates: list[float] = None) -> list[Machine]:
    if win_rates is None:
        return [Machine(str(i), p_win=rng().uniform(0, 1)) for i in range(n)]

    machines = []
    for i, p_win in enumerate(win_rates):
        if p_win < 0.0 or p_win > 1.0:
            raise ValueError('Machine win probabilities must be between 0.0 and 1.0 (inclusive)')

        machines.append(Machine(str(i), p_win))

    return machines


def main():
    # configure logger
    logging.config.fileConfig('config/logging.conf')

    args = parse_args()

    load_path: Path = args.load_path
    save_path: Path = args.save_path

    machines = create_machines(args.n_machines, args.machines)
    env = BanditEnvironment(machines)

    primitive_items = [
        sym_item('W', primitive_value=1.0),
        sym_item('L', primitive_value=-1.0),
        sym_item('P', primitive_value=-0.5),
    ]

    if args.optimize:
        # this typing cast is a workaround to a current limitation in type inference for partials that affects PyCharm
        # and likely other IDEs/static code analysis tools.
        run: Runner = typing.cast(Runner, partial(run_episode, on_step=None))

        data_frame = optimize(
            env=env,
            run_episode=run,
            primitive_items=primitive_items,
            calculate_score=calculate_score,
            sampler=args.optimizer_sampler,
            pruner=args.optimizer_pruner,
            n_trials=args.optimizer_trials,
            n_runs_per_trial=args.optimizer_runs_per_trial,
            n_episodes_per_run=args.episodes,
            n_steps_per_episode=args.steps,
            study_name=args.optimizer_study_name,
            use_database=args.optimizer_use_db,
        )

        report_filename = get_optimizer_report_filename(
            study_name=args.optimizer_study_name,
            sampler=args.optimizer_sampler,
            pruner=args.optimizer_pruner,
        )

        report_path = f'local/optimize/{env.id}/{report_filename}'
        logger.info("Writing report to {}".format(report_path))

        data_frame.to_csv(report_path)
    else:
        on_step_callbacks = [
            display_action,
            display_state,
            display_combined_step
            # default_display_on_step,
            # display_n_schemas,
            # display_weights,
            # display_score,
        ]
        if save_path:
            on_step_callbacks.append(
                SaveCallback(
                    path=save_path,
                    conditions=[save_every_n_steps(args.save_frequency)])
            )

        # this typing cast is a workaround to a current limitation in type inference for partials that affects PyCharm
        # and likely other IDEs/static code analysis tools.
        run: Runner = typing.cast(Runner, partial(run_episode, on_step=on_step_callbacks))

        schema_mechanism = load_schema_mechanism(load_path) if load_path else None
        if not schema_mechanism:
            schema_mechanism = create_schema_mechanism(
                env=env,
                primitive_items=primitive_items,
            )

        # command-line argument overrides schema mechanism's strategy parameters
        if args.greedy:
            schema_mechanism.schema_selection.evaluation_strategy = DefaultGreedyEvaluationStrategy(
                reliability_threshold=get_global_params().get('schema.reliability_threshold')
            )

        episode_summary = run(
            env=env,
            schema_mechanism=schema_mechanism,
            max_steps=args.steps,
            render_env=False
        )

        display_environment_summary(env)
        # print("Performance Summary")
        display_performance_summary([episode_summary])

        display_item_values()
        # print("Schema Known")
        display_known_schemas(schema_mechanism)
        print("Schema Memory")
        display_schema_memory(schema_mechanism)

        # this save is in addition to any runner callbacks to make sure final model is persisted
        if save_path:
            save(
                modules=[schema_mechanism],
                path=save_path,
                encoding=DEFAULT_ENCODING,
            )


if __name__ == '__main__':
    main()
