import typing as t
from enum import Enum

from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_only
from test_tube import Experiment


class Mode(Enum):
    Training = "training"
    Validation = "validatoin"


class TensorBoardLogger(LightningLoggerBase):
    NAME_CSV_TAGS = "meta_tags.csv"

    def __init__(
        self,
        save_dir,
        version=None,
        train_key="train",
        valid_key="valid",
        log_train_every_n_step=100,
        create_git_tag=False,
        description=None,
        debug=False,
        **kwargs,
    ):
        super().__init__()
        self.save_dir = save_dir
        self.train_key = train_key
        self.valid_key = valid_key
        self.log_every_n_step = log_train_every_n_step
        self.create_git_tag = create_git_tag
        self.description = description
        self.debug = debug

        self._version = version

        self._experiments: t.Dict[Mode, t.Optional[Experiment]] = {
            Mode.Training: None,
            Mode.Validation: None,
        }
        self._mode = Mode.Training

        self.tags = {}
        self.kwargs = kwargs

    def train(self):
        self._mode = Mode.Training
        return self

    def valid(self):
        self._mode = Mode.Validation
        return self

    @property
    def mode(self):
        return self._mode

    def init_experiments(self):
        if (
            self._experiments[Mode.Training] is not None
            and self._experiments[Mode.Validation] is not None
        ):
            return

        self._experiments[Mode.Training] = Experiment(
            save_dir=self.save_dir,
            name="train",
            debug=self.debug,
            version=self.version,
            description=self.description,
            create_git_tag=self.create_git_tag,
            rank=self.rank,
        )
        self._experiments[Mode.Validation] = Experiment(
            save_dir=self.save_dir,
            name="valid",
            debug=self.debug,
            version=self.version,
            description=self.description,
            create_git_tag=self.create_git_tag,
            rank=self.rank,
        )

    @property
    def experiment(self):
        if self._experiments[self.mode] is not None:
            return self._experiments[self.mode]

        self.init_experiments()

        return self._experiments[self.mode]

    @rank_zero_only
    def log_hyperparams(self, params):
        self.experiment.debug = self.debug
        self.experiment.argparse(params)

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        self.experiment.debug = self.debug
        if step is None:
            self.experiment.log(metrics, global_step=step)
        else:
            if (
                self.mode == Mode.Validation
                or step % self.log_every_n_step == 0
            ):
                self.experiment.log(metrics, global_step=step)

    def log_histogram(self, name, data, step=None):
        self.experiment.debug = self.debug
        if step is None:
            self.experiment.add_histogram(name, data, None)
        else:
            if (
                self.mode == Mode.Validation
                or step % self.log_every_n_step == 0
            ):
                self.experiment.add_histogram(name, data, global_step=step)

    @rank_zero_only
    def save(self):
        # TODO: HACK figure out where this is being set to true
        self._experiments[Mode.Training].debug = self.debug
        self._experiments[Mode.Training].save()

        self._experiments[Mode.Validation].debug = self.debug
        self._experiments[Mode.Validation].save()

    @rank_zero_only
    def finalize(self, status):
        self._experiments[Mode.Training].debug = self.debug
        self._experiments[Mode.Validation].debug = self.debug
        self.save()
        self.close()

    @rank_zero_only
    def close(self):
        self._experiments[Mode.Training].debug = self.debug
        self._experiments[Mode.Validation].debug = self.debug
        if not self.debug:
            self._experiments[Mode.Training].close()
            self._experiments[Mode.Validation].close()

    @property
    def rank(self):
        return self._rank

    @rank.setter
    def rank(self, value):
        self._rank = value
        if self._experiments[Mode.Training] is not None:
            self._experiments[Mode.Training].rank = value

        if self._experiments[Mode.Validation] is not None:
            self._experiments[Mode.Validation].rank = value

    @property
    def name(self):
        if self._experiments[self.mode] is None:
            return "train" if self.mode == Mode.Training else "valid"
        else:
            return self.experiment.name

    @property
    def version(self):
        if self._experiments[self.mode] is None:
            return self._version
        else:
            return self.experiment.version

    # Test tube experiments are not pickleable, so we need to override a few
    # methods to get DDP working. See
    # https://docs.python.org/3/library/pickle.html#handling-stateful-objects
    # for more info.
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_experiment"] = self.experiment.get_meta_copy()
        return state

    def __setstate__(self, state):
        self._experiments[self.mode] = state["_experiment"].get_non_ddp_exp()
        del state["_experiment"]
        self.__dict__.update(state)
