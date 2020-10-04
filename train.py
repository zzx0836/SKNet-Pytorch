import torch


def train_epoch(model, optimizer, train_loader, criterion, epoch, writer=None):
    model.train()
    num = len(train_loader)
    for i, (data, label) in enumerate(train_loader):
        model.zero_grad()
        optimizer.zero_grad()
        data = data.cuda()
        label = label.cuda().long()
        result = model(data)
        loss = criterion(result, label)
        loss.backward()
        optimizer.step()
        if i%10==0:
            print('epoch {}, [{}/{}], loss {}'.format(epoch, i, num, loss))
            if writer is not None:
                writer.add_scalar('loss', loss.item(), epoch*num + i)

