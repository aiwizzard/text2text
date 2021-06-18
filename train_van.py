import json
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import optimizer
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils import clip_grad_norm_

from tqdm import tqdm

import config as config

from model.dataset_van import ChatDataSet, SampledDataLoader
from model.model import ChatModel
from optim.optimizer import ScheduledOptimizer
from utils import create_masks, create_train_data
from tokenizer import Tokenizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)



def train(epoch: int, config, model: ChatModel, data_loader, criterion, optimizer):
    model.train()
    with tqdm(total=len(data_loader), desc=f"Epoch {epoch+1}") as pbar:
        for i, (x, y) in enumerate(data_loader):
            x = x.to(config.device)
            y = y.to(config.device)
            
            # prepare target data
            target = y[:, :-1]
            target_y = y[:, 1:]

            # create mask and add dimension

            source_mask, target_mask = create_masks(x, target)

            out = model(x, source_mask, target, target_mask)

            optimizer.zero_grad()

            loss = criterion(out.transpose(1, 2), target_y).mean()
            loss.backward()
            optimizer.step()
            clip_grad_norm_(model.parameters(), config.max_grad_norm)

            pbar.update(1)
            pbar.set_postfix_str(f"loss: {loss.item():.5f}")
    # Always overwrite
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        f"{config.data_dir}/{config.fn}.pth",
    )
    # not overwrite
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        },
        f'{config.data_dir}/{config.fn}_{epoch}.pth'
    )
    logger.info("Model Saved")


def main(config):
    tokenizer = Tokenizer.from_pretrained(config.bert_model_name)
    train_data = create_train_data(
        config.pickle_path, tokenizer, use_pickle=config.use_pickle
    )
    # dataset
    dataset = ChatDataSet(train_data, Tokenizer.pad_token_id)
    # data loader
    data_loader = SampledDataLoader(
        dataset, batch_size=config.batch_size, padding=tokenizer.pad_token_id
    )
    model = ChatModel(config)
    model = model.to(config.device)

    criterion = nn.CrossEntropyLoss(reduction="none")

    adam_optimizer = optim.AdamW(
        model.parameters(), lr=config.lr, betas=config.betas, eps=1e-9
    )
    optimizer = ScheduledOptimizer(
        adam_optimizer, factor=2, model_dim=config.model_dim, warmup=config.warmup
    )

    for epoch in range(0, config.epochs):
        train(epoch, config, model, data_loader, criterion, adam_optimizer)

    logger.info("finished Training")

if __name__ == "__main__":
    main(config)