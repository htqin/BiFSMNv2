from .config import *
from .data_processor import CluenerProcessor
import urllib.request
import zipfile
import os
import logging
import torch


def download_ner():
    TASK2PATH = 'https://storage.googleapis.com/cluebenchmark/tasks/cluener_public.zip'
    logging.info("Downloading and extracting cluener...")
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    data_file = os.path.join(data_dir, "cluener_public.zip")
    urllib.request.urlretrieve(TASK2PATH, data_file)
    with zipfile.ZipFile(data_file) as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(data_file)
    logging.info(
        "Completed! Downloaded cluener data to directory {}".format(data_dir))

    processor = CluenerProcessor(data_dir=data_dir)
    processor.get_vocab()

    train_dataset = data_dir / 'cached_crf-train_bilstm_crf_ner'
    eval_dataset = data_dir / 'cached_crf-dev_bilstm_crf_ner'
    if not train_dataset.exists():
        train_examples = processor.get_train_examples()
        torch.save(train_examples, str(train_dataset))
    if not eval_dataset.exists():
        eval_examples = processor.get_dev_examples()
        torch.save(eval_examples, str(eval_dataset))

    return processor
