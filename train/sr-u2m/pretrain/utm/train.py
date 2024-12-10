
import logging
import pathlib as pl
import sys

import numpy as np
import torch
import torchaudio
import tqdm
from hyperpyyaml import load_hyperpyyaml
from torch.nn.parallel import DistributedDataParallel

import speechbrain as sb


logger = logging.getLogger(__name__)


class S2UT(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Computes the forward pass.

        Arguments
        ---------
        batch : torch.Tensor or tensors
            An element from the dataloader, including inputs for processing.
        stage : Stage
            The stage of the experiment: Stage.TRAIN, Stage.VALID, Stage.TEST

        Returns
        -------
        (torch.Tensor or torch.Tensors, list of float or None, list of str or None)
            The outputs after all processing is complete.
        """
        batch = batch.to(self.device)
        tgt_mel_fore, _ = batch.tgt_mel_fore

        code, code_lens = batch.code

        # Use default padding value for wav2vec2
        tgt_mel_fore[tgt_mel_fore == self.hparams.pad_index] = 0.0
        enc_out = self.modules.embedding_model(tgt_mel_fore)
        # calculate mellogits
        
        fft_out = self.modules.fft(
                        code, 
                        pad_idx=self.hparams.pad_index
                        )
        pred_mel = self.modules.mel_adapter(enc_out, fft_out, target_length = code_lens)
        return pred_mel


    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs.
        Arguments
        ---------
        predictions : torch.Tensor
            The model generated spectrograms and other metrics from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """
        pred_mel = predictions
        tgt_mel_back, tgt_mel_back_lens = batch.tgt_mel_back
        tgt_mel_back[tgt_mel_back == self.hparams.pad_index] = 0
        ids = batch.id
        # speech translation loss
        loss = self.hparams.mel_cost(pred_mel, tgt_mel_back, tgt_mel_back_lens)

        return loss

    def freeze_optimizers(self, optimizers):
        """Freezes the wav2vec2 optimizer according to the warmup steps"""
        valid_optimizers = {}
        valid_optimizers["model_optimizer"] = optimizers["model_optimizer"]
        return valid_optimizers

    def init_optimizers(self):
        """Called during ``on_fit_start()``, initialize optimizers
        after parameters are fully configured (e.g. DDP, jit).
        """
        self.optimizers_dict = {}

        # Initializes the wav2vec2 optimizer if the model is not wav2vec2_frozen
        self.model_optimizer = self.hparams.opt_class(
            self.hparams.model.parameters()
        )
        self.optimizers_dict["model_optimizer"] = self.model_optimizer

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "model_optimizer", self.model_optimizer
            )

    def on_fit_batch_start(self, batch, should_step):
        """Called at the beginning of ``fit_batch()``.

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.
        should_step : boolean
            Whether optimizer.step() was called or not.
        """
        pass

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """Called after ``fit_batch()``, meant for calculating and logging metrics.

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.
        outputs : list or dictionary of torch.Tensors
            Returned value of compute_forward().
        loss : torch.Tensor
            Returned value of compute_objectives().
        should_step : boolean
            Whether optimizer.step() was called or not.
        """
        if should_step:
            # anneal model lr every update
            self.hparams.noam_annealing(self.model_optimizer)

    def on_stage_start(self, stage, epoch):
        """Gets called when a stage starts.

        Arguments
        ---------
        stage : Stage
            The stage of the experiment: Stage.TRAIN, Stage.VALID, Stage.TEST
        epoch : int
            The current epoch count.

        Returns
        -------
        None
        """
        if stage != sb.Stage.TRAIN:
            if (
                stage == sb.Stage.VALID
                and epoch % self.hparams.evaluation_interval != 0
            ):
                return

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_loss

        # At the end of validation, we can write
        elif (
            stage == sb.Stage.VALID
            and epoch % self.hparams.evaluation_interval == 0
        ):
            # delete vocoder and asr to free memory for next training epoch

            stage_stats = {"loss": stage_loss}
            

            current_epoch = self.hparams.epoch_counter.current
            lr_model = self.hparams.noam_annealing.current_lr

               
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": current_epoch,
                    "lr_model": lr_model,
                },
                train_stats={"loss": self.train_stats},
                valid_stats=stage_stats,
            )
            

            # Save the current checkpoint and delete previous checkpoints.
            self.checkpointer.save_and_keep_only(
                meta={
                    "loss": stage_stats["loss"],
                    "epoch": epoch,
                },
                min_keys=["loss"],
                num_to_keep=10,
            )

        elif stage == sb.Stage.TEST:
            stage_stats = {"loss": stage_loss}

            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

    def _save_progress_sample(self, epoch):
        """Save samples and BLEU score from last batch for current epoch.

        Arguments
        ---------
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.

        Returns
        -------
        None
        """
        if self.last_batch is None:
            return

        (
            ids,
            (wavs, transcripts),
            (tgt_transcripts, tgt_wavs),
            (mel_transcripts, mel_wavs),
        ) = self.last_batch

        save_folder = pl.Path(self.hparams.progress_sample_path) / f"{epoch}"
        save_folder.mkdir(parents=True, exist_ok=True)

        sample_size = self.hparams.progress_batch_sample_size
        if len(ids) < sample_size:
            sample_size = len(ids)
        for i in tqdm.tqdm(range(sample_size)):
            utt_id = ids[i]
            wav = wavs[i]
            transcript = transcripts[i]
            tgt_transcript = tgt_transcripts[i]
            tgt_wav = tgt_wavs[i]
            mel_transcript = mel_transcripts[i]
            mel_wav = mel_wavs[i]

            sample_path = save_folder / f"{utt_id}_pred.wav"
            sb.dataio.dataio.write_audio(
                sample_path, wav, self.hparams.sample_rate
            )

            sample_path = save_folder / f"{utt_id}_ref.wav"
            sb.dataio.dataio.write_audio(
                sample_path, tgt_wav, self.hparams.sample_rate
            )
            
            sample_path = save_folder / f"{utt_id}_mel.wav"
            sb.dataio.dataio.write_audio(
                sample_path, mel_wav, self.hparams.sample_rate
            )

            sample_path = save_folder / f"{utt_id}.txt"
            with open(sample_path, "w") as file:
                file.write(f"pred: {transcript}\n")
                file.write(f"ref: {tgt_transcript}\n")
                file.write(f"mel: {mel_transcript}\n")

        self.bleu_metric.append(
            ids[:sample_size],
            transcripts[:sample_size],
            [tgt_transcripts[:sample_size]],
        )
        
        self.mel_bleu_metric.append(
            ids[:sample_size],
            mel_transcripts[:sample_size],
            [tgt_transcripts[:sample_size]],
        )

        bleu_path = save_folder / "bleu.txt"
        with open(bleu_path, "w") as file:
            file.write(
                f"BLEU score: {round(self.bleu_metric.summarize('BLEU'), 2)}\n"
            )
            file.write(
                f"MEL_BLEU score: {round(self.mel_bleu_metric.summarize('BLEU'), 2)}\n"
            )


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """
    codes_folder = pl.Path(hparams["codes_folder"])
    # Define audio pipeline. In this case, we simply read the audio contained
    # in the variable src_audio with the custom reader.
    @sb.utils.data_pipeline.takes("tgt_audio")
    @sb.utils.data_pipeline.provides("tgt_mel_fore","tgt_mel_back")
    def src_audio_pipeline(wav):
        """Load the source language audio signal.
        This is done on the CPU in the `collate_fn`
        """
        info = torchaudio.info(wav)
        sig = sb.dataio.dataio.read_audio(wav)
        sig = torchaudio.transforms.Resample(
            info.sample_rate, hparams["sample_rate"]
        )(sig)
        audio_length = len(sig)
        half_length = audio_length // 2
        audio_fore = sig[:half_length]
        audio_back = sig[half_length:]
        tgt_mel_fore = hparams["mel_spectogram"](audio=audio_fore).transpose(0,1)
        tgt_mel_back = hparams["mel_spectogram"](audio=audio_back).transpose(0,1)
        
        return tgt_mel_fore, tgt_mel_back
    
    @sb.utils.data_pipeline.takes("id")
    @sb.utils.data_pipeline.provides("code")
    def unit_pipeline(utt_id):
        """Load target codes"""
        code = np.load(codes_folder / f"{utt_id}_tgt.npy")
        code = torch.LongTensor(code)
        yield code

    datasets = {}
    for split in hparams["splits"]:
        datasets[split] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{split}_json"],
            dynamic_items=[
                src_audio_pipeline,
                unit_pipeline
            ],
            output_keys=[
                "id",
                "tgt_mel_fore",
                "tgt_mel_back",
                "code",
            ],
        )

    # Sorting training data with ascending order makes the code  much
    # faster  because we minimize zero-padding. In most of the cases, this
    # does not harm the performance.
    if hparams["sorting"] == "ascending":
        datasets["train"] = datasets["train"].filtered_sorted(
            sort_key="duration"
        )
        datasets["valid"] = datasets["valid"].filtered_sorted(
            sort_key="duration"
        )

        hparams["train_dataloader_opts"]["shuffle"] = False
        hparams["valid_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        datasets["train"] = datasets["train"].filtered_sorted(
            sort_key="duration", reverse=True
        )
        datasets["valid"] = datasets["valid"].filtered_sorted(
            sort_key="duration", reverse=True
        )

        hparams["train_dataloader_opts"]["shuffle"] = False
        hparams["valid_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        hparams["train_dataloader_opts"]["shuffle"] = True
        hparams["valid_dataloader_opts"]["shuffle"] = False

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    # Dynamic Batching is used, we instantiate the needed samplers.
    train_batch_sampler = None
    if hparams["dynamic_batching"]:
        from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa

        dynamic_hparams = hparams["dynamic_batch_sampler"]
        num_buckets = dynamic_hparams["num_buckets"]

        train_batch_sampler = DynamicBatchSampler(
            datasets["train"],
            dynamic_hparams["max_batch_len"],
            num_buckets=num_buckets,
            length_func=lambda x: x["duration"],
            shuffle=dynamic_hparams["shuffle_ex"],
            batch_ordering=dynamic_hparams["batch_ordering"],
            max_batch_ex=dynamic_hparams["max_batch_ex"],
        )

    return datasets, train_batch_sampler


if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )


    datasets, train_bsampler = dataio_prepare(hparams)

    s2ut_brain = S2UT(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]

    if train_bsampler is not None:
        train_dataloader_opts = {
            "batch_sampler": train_bsampler,
            "num_workers": hparams["num_workers"],
            "collate_fn": hparams["train_dataloader_opts"]["collate_fn"],
        }

    s2ut_brain.fit(
        s2ut_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid_small"],
        train_loader_kwargs=train_dataloader_opts,
        valid_loader_kwargs=valid_dataloader_opts,
    )

    test_dataloader_opts = {
        "batch_size": 1,
    }

    for dataset in ["valid", "test"]:
        s2ut_brain.evaluate(
            datasets[dataset],
            test_loader_kwargs=test_dataloader_opts,
        )
