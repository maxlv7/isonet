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

if __name__ == '__main__':

    print("load data....")
    dataset = ISONetData()
    dataset_test = ISONetData(train=False)

    data_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=64, shuffle=False)
    print("load data success...")
    cuda = False
    resume = False

    model_path = Path("models")
    if not model_path.exists():
        model_path.mkdir()

    checkpoint_path = model_path.joinpath("checkpoint")
    if not checkpoint_path:
        checkpoint_path.mkdir()

    if torch.cuda.is_available():
        cuda = True
        device = torch.cuda.current_device()

    net = ISONet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    if cuda:
        net = net.to(device=device)
        criterion = criterion.to(device=device)

    scheduler = MultiStepLR(optimizer=optimizer, milestones=[10, 30], gamma=0.1)
    writer = SummaryWriter('runs')

    # resume
    if resume:
        print("resuming training...")
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict((checkpoint["optimizer"]))
        scheduler.load_state_dict(checkpoint["scheduler"])
        resume_epoch = checkpoint["epoch"]

    # train
    start_epoch = 0
    if locals().get("resume_epoch"):
        start_epoch = locals().get("resume_epoch")

    for epoch in range(start_epoch,60):
        net.train()
        for i, (data, label) in enumerate(data_loader, 0):
            if i == 0:
                start_time = int(time.time())
            if cuda:
                data = data.to(device=device)
                label = label.to(device=device)
            optimizer.zero_grad()
            output = net(data)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            if i % 200 == 199:
                end_time = int(time.time())
                use_time = end_time-start_time

                print(f">>> epoch[{epoch + 1}] loss[{loss:.4f}]  {i * 64}/{len(dataset)} lr{scheduler.get_last_lr()}")
                writer.add_scalar("train_loss", loss)

        # validate
        print("eval model...")
        net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            loss_t = nn.CrossEntropyLoss()
            for data, label in data_loader_test:
                if cuda:
                    data = data.to(device)
                    label = label.to(device)
                    loss_t = loss_t.to(device)
                predict = net(data)
                test_loss += loss_t(predict, label).item()  # sum up batch loss
                pred = predict.argmax(dim=1)  # get the index of the max log-probability
                correct = correct + pred.eq(label.view_as(pred)).sum().item()

        test_loss /= len(data_loader_test.dataset)

        print('\n测试数据集: 平均损失为: {:.4f}, 正确率为: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader_test.dataset),
            100. * correct / len(data_loader_test.dataset)))
        scheduler.step()

        # save checkpoint
        checkpoint = {
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "scheduler": scheduler.state_dict()
        }
        torch.save(checkpoint, checkpoint_path.joinpath(f"net_{epoch}.cpth"))
        # save model
        torch.save(net.state_dict(), model_path.joinpath("net.pth"))
