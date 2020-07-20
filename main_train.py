import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from isonet import ISONet
from data_set import ISONetData
from utils import time2str

if __name__ == '__main__':

    # set paramaters
    cuda = True
    resume = False
    batch_size = 128
    start_epoch = 1
    resume_checkpoint = "net_28_2020-07-20_12:17:10.cpth"
    milestones = [10, 30]
    lr = 1e-3
    data_path = "data_64_32"

    print("load data....")
    dataset = ISONetData(data_path=data_path)
    dataset_test = ISONetData(data_path=data_path, train=False)

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)
    print("load data success...")

    model_path = Path("models")
    checkpoint_path = model_path.joinpath("checkpoint")

    if not model_path.exists():
        model_path.mkdir()
    if not checkpoint_path.exists():
        checkpoint_path.mkdir()

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        print("can not use cuda!")
        cuda = False

    net = ISONet()
    criterion = nn.L1Loss(reduction="sum")
    optimizer = optim.Adam(net.parameters(), lr=lr)

    if cuda:
        net = net.to(device=device)
        criterion = criterion.to(device=device)

    scheduler = MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
    # writer = SummaryWriter('runs')

    # resume
    if resume:
        print("resuming training...")
        checkpoint = torch.load(checkpoint_path.joinpath(resume_checkpoint))
        net.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict((checkpoint["optimizer"]))
        scheduler.load_state_dict(checkpoint["scheduler"])
        resume_epoch = checkpoint["epoch"]
        start_epoch = resume_epoch
        print(f"start resume epoch {start_epoch}...")
    else:
        # init weight
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    # For save better model
    test_loss_flag = False
    best_test_loss = 0

    for epoch in range(start_epoch, 60):
        print(f"start train epoch {epoch}...")
        net.train()
        for i, (data, label) in enumerate(data_loader, 0):
            if i == 0:
                start_time = int(time.time())
            if cuda:
                data = data.to(device=device)
                label = label.to(device=device)
            label = label.unsqueeze(1)

            optimizer.zero_grad()

            output = net(data)

            loss = criterion(output, label)

            loss.backward()

            optimizer.step()

            if i % 500 == 499:
                end_time = int(time.time())
                use_time = end_time - start_time

                print(
                    f">>> epoch[{epoch}] loss[{loss:.4f}]  {i * batch_size}/{len(dataset)} lr{scheduler.get_last_lr()} ",
                    end="")
                print(f"use time [{end_time - start_time}] sec")
                start_time = end_time
                # writer.add_scalar("train_loss", loss)

        # validate
        print("eval model...")
        net.eval()

        test_loss = 0
        with torch.no_grad():
            loss_t = nn.L1Loss(reduction="sum")
            if cuda:
                loss_t = loss_t.to(device)
            for data, label in data_loader_test:
                if cuda:
                    data = data.to(device)
                    label = label.to(device)
                # expand dim
                label = label.unsqueeze_(1)
                predict = net(data)
                # sum up batch loss
                test_loss += loss_t(predict, label).item()

        test_loss /= len(data_loader_test.dataset)
        test_loss *= batch_size
        print(f'\nTest Data: Average batch[{batch_size}] loss: {test_loss:.4f}\n')
        scheduler.step()

        checkpoint = {
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "scheduler": scheduler.state_dict()
        }

        # First save
        if best_test_loss == 0:
            print("save model...")
            torch.save(net.state_dict(), model_path.joinpath(f"net_{time2str()}.pth"))
            best_test_loss = test_loss
        else:
            # Save better model
            if test_loss < best_test_loss:
                # save model
                print("Get better model,save model...")
                torch.save(net.state_dict(), model_path.joinpath(f"net_best_{time2str()}.pth"))
                best_test_loss = test_loss
        # save checkpoint
        print("save checkpoint...")
        torch.save(checkpoint, checkpoint_path.joinpath(f"net_{epoch}_{time2str()}.cpth"))
