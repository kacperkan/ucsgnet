import logging as log
import os
import json

import pytorch_lightning as pl


class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def format_checkpoint_name(self, epoch, metrics, ver=None):
        """Save model

        Custom callback since file naming in the original implementation was
        too complex and unnecessary.
        """
        if epoch is not None:
            comps = self.filename.split(".")
            return os.path.join(self.dirpath, f"{comps[0]}_{epoch}.{comps[1]}")
        return os.path.join(self.dirpath, self.filename)

    def _dump_metrics(self, epoch, metrics):
        if metrics is None:
            return
        if epoch is not None:
            with open(
                os.path.join(self.dirpath, f"metrics_{epoch}.json"), "w"
            ) as f:
                json.dump(metrics, f)
        else:
            with open(os.path.join(self.dirpath, f"metrics.json"), "w") as f:
                json.dump(metrics, f)

    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        self.epochs_since_last_check += 1
        filepath = self.format_checkpoint_name(epoch, metrics)
        if self.epochs_since_last_check >= self.period:
            self.epochs_since_last_check = 0
            filepath = self.format_checkpoint_name(epoch, metrics)
            self._dump_metrics(epoch, metrics)
            self._save_model(filepath)
        if self.verbose > 0:
            log.info(f"\nEpoch {epoch:05d}: saving model to {filepath}")
        filepath = self.format_checkpoint_name(None, metrics)
        self._dump_metrics(None, metrics)
        self._save_model(filepath)
