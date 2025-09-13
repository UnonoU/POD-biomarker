import sys
# sys.path.append('/home/zhuyan/workspace/narcosis')  
import torch
import torch.nn.functional as F
import numpy as np
import time
import shutil
import os
# from interpreter.io_utils import save_config_file, save_checkpoint


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def train_earlyStopping(args, train_loader, val_loader, model, criterion, optimizer, scheduler, saveModel):
    bad_count = 0
    best_acc, best_loss = -1, 1000

    val_acc = 0
    val_loss = 0
    model.eval()
    for counter, (x_batch, y_batch) in enumerate(val_loader):
        x_batch = x_batch.to(args.device)
        y_batch = y_batch.to(args.device)

        logits = model(x_batch)
        # print(logits.shape)
        # print(y_batch)
        # print(logits.shape, y_batch.shape)
        loss = criterion(logits, y_batch)

        top1 = accuracy(logits, y_batch, topk=(1,))
        val_acc += top1[0]
        val_loss += loss.data.cpu().numpy()

    val_acc /= (counter + 1)
    val_loss /= (counter + 1)
    print(
        f"Epoch {-1}     Val accuracy: {val_acc.item()}    Val loss: {val_loss}")

    train_loss_history, val_loss_history = np.zeros(args.epochs_finetune), np.zeros(args.epochs_finetune)
    train_acc_history, val_acc_history = np.zeros(args.epochs_finetune), np.zeros(args.epochs_finetune)

    model_epochs, optimizer_epochs = {}, {}
    for epoch in range(args.epochs_finetune):
        start_time = time.time()
        train_acc = 0
        train_loss = 0
        model.train()
        train_confusion = torch.zeros((2, 2))
        for counter, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(args.device)
            y_batch = y_batch.to(args.device)

            logits = model(x_batch)
            # print(logits.shape, y_batch.shape)
            loss = criterion(logits, y_batch)

            top1 = accuracy(logits, y_batch, topk=(1,))
            train_confusion = train_confusion + get_confusionMat(logits, y_batch, 2)
            train_acc += top1[0]
            train_loss += loss.data.cpu().numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if np.mod(counter+1, 100) == 0:
            #     print('counter', counter+1, 'train acc', train_acc.item() / counter+1)

        scheduler.step()
        # scheduler.step(val_loss)

        train_acc /= (counter + 1)
        train_loss /= (counter + 1)

        val_acc = 0
        val_loss = 0
        # confusionMat = torch.zeros((9,9))
        confusionMat = torch.zeros((2, 2))

        model.eval()
        for counter, (x_batch, y_batch) in enumerate(val_loader):
            x_batch = x_batch.to(args.device)
            y_batch = y_batch.to(args.device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            top1 = accuracy(logits, y_batch, topk=(1,))
           # confusionMat = confusionMat + get_confusionMat(logits, y_batch, 9)
            confusionMat = confusionMat + get_confusionMat(logits, y_batch, 2)
            val_acc += top1[0]
            val_loss += loss.data.cpu().numpy()
        # print(confusionMat)


        val_acc /= (counter + 1)
        val_loss /= (counter + 1)
        print(
            f"Epoch {epoch}    Train accuracy {train_acc.item()}    Val accuracy: {val_acc.item()}    Train loss {train_loss}    Val loss: {val_loss}")
        # print('learning rate', scheduler.get_last_lr())
        train_loss_history[epoch] = train_loss
        val_loss_history[epoch] = val_loss
        train_acc_history[epoch] = train_acc
        val_acc_history[epoch] = val_acc

        model_epochs[epoch] = model
        optimizer_epochs[epoch] = optimizer

        if val_acc > best_acc:
            # 在这里添加额外的条件检查
            if epoch > 20:
                bad_count = 0
                best_loss = val_loss
                best_acc = val_acc
                best_epoch = epoch
                best_confusion = confusionMat
                best_confusion_train = train_confusion
            # 如果当前epoch不大于20，则即使性能有所提升，也不会更新最佳模型
        else:
            bad_count += 1

        if bad_count > args.max_tol:
            print('best epoch %d, train loss: %.4f, val loss: %.4f, train acc: %.3f, val acc: %.3f' % (
                best_epoch, train_loss_history[best_epoch], val_loss_history[best_epoch],
                train_acc_history[best_epoch], val_acc_history[best_epoch]))
            break

        end_time = time.time()
        print('time consumed:', end_time - start_time)

    if saveModel:
        if not os.path.exists(args.save_dir_ft):
            os.mkdir(args.save_dir_ft)
        checkpoint_name = 'finetune_checkpoint_{:04d}.pth.tar'.format(best_epoch)
        save_checkpoint({
            'epoch': best_epoch,
            'state_dict': model_epochs[best_epoch].state_dict(),
            'optimizer': optimizer_epochs[best_epoch].state_dict(),
        }, is_best=False, filename=os.path.join(args.save_dir_ft, checkpoint_name))

        checkpoint_name = 'finetune_checkpoint_{:04d}.pth.tar'.format(epoch)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model_epochs[epoch].state_dict(),
            'optimizer': optimizer_epochs[epoch].state_dict(),
        }, is_best=False, filename=os.path.join(args.save_dir_ft, checkpoint_name))      

    if bad_count <=  args.max_tol:
        print('best epoch %d, train loss: %.4f, val loss: %.4f, train acc: %.3f, val acc: %.3f' % (
                best_epoch, train_loss_history[best_epoch], val_loss_history[best_epoch],
                train_acc_history[best_epoch], val_acc_history[best_epoch]))
    return best_epoch, train_loss_history, val_loss_history, train_acc_history, val_acc_history, best_confusion, best_confusion_train, confusionMat,train_confusion


def train_earlyStopping_new(args, train_loader, val_loader, model, criterion, optimizer, scheduler, saveModel):
    bad_count = 0
    best_acc, best_loss = -1, 1000

    val_acc = 0
    val_loss = 0
    model.eval()
    for counter, (x_batch, y_batch) in enumerate(val_loader):
        x_batch = x_batch.to(args.device)
        y_batch = y_batch.to(args.device)

        logits = model(x_batch)
        # print(logits.shape, y_batch.shape)
        loss = criterion(logits, y_batch)

        top1 = accuracy(logits, y_batch, topk=(1,))
        val_acc += top1[0]
        val_loss += loss.data.cpu().numpy()

    val_acc /= (counter + 1)
    val_loss /= (counter + 1)
    print(
        f"Epoch {-1}     Val accuracy: {val_acc.item()}    Val loss: {val_loss}")

    train_loss_history, val_loss_history = np.zeros(args.epochs_finetune), np.zeros(args.epochs_finetune)
    train_acc_history, val_acc_history = np.zeros(args.epochs_finetune), np.zeros(args.epochs_finetune)

    model_epochs, optimizer_epochs = {}, {}
    logits_epochs = {}
    for epoch in range(args.epochs_finetune):
        start_time = time.time()
        train_acc = 0
        train_loss = 0
        model.train()
        for counter, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(args.device)
            y_batch = y_batch.to(args.device)

            logits = model(x_batch)
            # print(logits.shape, y_batch.shape)
            loss = criterion(logits, y_batch)

            top1 = accuracy(logits, y_batch, topk=(1,))
            train_acc += top1[0]
            train_loss += loss.data.cpu().numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if np.mod(counter+1, 100) == 0:
            #     print('counter', counter+1, 'train acc', train_acc.item() / counter+1)

        scheduler.step()

        train_acc /= (counter + 1)
        train_loss /= (counter + 1)

        val_acc = 0
        val_loss = 0
        confusionMat = torch.zeros((9,9))
        model.eval()
        logits_epochs[epoch] = np.zeros((0, 2))
        y_all = np.zeros(0)
        for counter, (x_batch, y_batch) in enumerate(val_loader):
            x_batch = x_batch.to(args.device)
            y_batch = y_batch.to(args.device)

            logits = model(x_batch)

            logits_np = logits.detach().cpu().numpy()
            y_np = y_batch.detach().cpu().numpy()
            logits_epochs[epoch] = np.concatenate((logits_epochs[epoch], logits_np), 0)
            y_all = np.concatenate((y_all, y_np))

            loss = criterion(logits, y_batch)

            top1 = accuracy(logits, y_batch, topk=(1,))
            confusionMat = confusionMat + get_confusionMat(logits, y_batch, 9)
            val_acc += top1[0]
            val_loss += loss.data.cpu().numpy()
        # print(confusionMat)

        val_acc /= (counter + 1)
        val_loss /= (counter + 1)
        print(
            f"Epoch {epoch}    Train accuracy {train_acc.item()}    Val accuracy: {val_acc.item()}    Train loss {train_loss}    Val loss: {val_loss}")
        # print('learning rate', scheduler.get_last_lr())
        train_loss_history[epoch] = train_loss
        val_loss_history[epoch] = val_loss
        train_acc_history[epoch] = train_acc
        val_acc_history[epoch] = val_acc

        model_epochs[epoch] = model
        optimizer_epochs[epoch] = optimizer

        if val_acc > best_acc:
            bad_count = 0
            best_loss = val_loss
            best_acc = val_acc
            best_epoch = epoch
            best_confusion = confusionMat
            
        else:
            bad_count += 1

        if bad_count > args.max_tol:
            print('best epoch %d, train loss: %.4f, val loss: %.4f, train acc: %.3f, val acc: %.3f' % (
                best_epoch, train_loss_history[best_epoch], val_loss_history[best_epoch],
                train_acc_history[best_epoch], val_acc_history[best_epoch]))
            break

        end_time = time.time()
        print('time consumed:', end_time - start_time)

    if saveModel:
        checkpoint_name = 'finetune_checkpoint_{:04d}.pth.tar'.format(best_epoch)
        save_checkpoint({
            'epoch': best_epoch,
            'state_dict': model_epochs[best_epoch].state_dict(),
            'optimizer': optimizer_epochs[best_epoch].state_dict(),
        }, is_best=False, filename=os.path.join(args.save_dir_ft, checkpoint_name))

    if bad_count <=  args.max_tol:
        print('best epoch %d, train loss: %.4f, val loss: %.4f, train acc: %.3f, val acc: %.3f' % (
                best_epoch, train_loss_history[best_epoch], val_loss_history[best_epoch],
                train_acc_history[best_epoch], val_acc_history[best_epoch]))
    best_logits = logits_epochs[best_epoch]
    print(np.sum(np.argmax(best_logits, axis=1)==y_all) / len(y_all))
    print(best_logits.shape)
    print(y_all.shape)
    return best_epoch, train_loss_history, val_loss_history, train_acc_history, val_acc_history, best_confusion, best_logits, y_all


def train_new(args, train_loader, val_loader, model, criterion, optimizer, scheduler, saveModel):
    bad_count = 0
    best_acc, best_loss = -1, 1000

    val_acc = 0
    val_loss = 0
    model.eval()
    for counter, (x_batch, y_batch) in enumerate(val_loader):
        x_batch = x_batch.to(args.device)
        y_batch = y_batch.to(args.device)

        logits = model(x_batch)
        # print(logits.shape, y_batch.shape)
        loss = criterion(logits, y_batch)

        top1 = accuracy(logits, y_batch, topk=(1,))
        val_acc += top1[0]
        val_loss += loss.data.cpu().numpy()

    val_acc /= (counter + 1)
    val_loss /= (counter + 1)
    print(
        f"Epoch {-1}     Val accuracy: {val_acc.item()}    Val loss: {val_loss}")

    train_loss_history, val_loss_history = np.zeros(args.epochs_finetune), np.zeros(args.epochs_finetune)
    train_acc_history, val_acc_history = np.zeros(args.epochs_finetune), np.zeros(args.epochs_finetune)

    model_epochs, optimizer_epochs = {}, {}
    logits_epochs = {}
    for epoch in range(args.epochs_finetune):
        start_time = time.time()
        train_acc = 0
        train_loss = 0
        model.train()
        for counter, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(args.device)
            y_batch = y_batch.to(args.device)

            logits = model(x_batch)
            # print(logits.shape, y_batch.shape)
            loss = criterion(logits, y_batch)

            top1 = accuracy(logits, y_batch, topk=(1,))
            train_acc += top1[0]
            train_loss += loss.data.cpu().numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if np.mod(counter+1, 100) == 0:
            #     print('counter', counter+1, 'train acc', train_acc.item() / counter+1)

        scheduler.step()

        train_acc /= (counter + 1)
        train_loss /= (counter + 1)

        val_acc = 0
        val_loss = 0
        confusionMat = torch.zeros((9,9))
        model.eval()
        logits_epochs[epoch] = np.zeros((0, 2))
        y_all = np.zeros(0)
        for counter, (x_batch, y_batch) in enumerate(val_loader):
            x_batch = x_batch.to(args.device)
            y_batch = y_batch.to(args.device)

            logits = model(x_batch)

            logits_np = logits.detach().cpu().numpy()
            y_np = y_batch.detach().cpu().numpy()
            logits_epochs[epoch] = np.concatenate((logits_epochs[epoch], logits_np), 0)
            y_all = np.concatenate((y_all, y_np))

            loss = criterion(logits, y_batch)

            top1 = accuracy(logits, y_batch, topk=(1,))
            confusionMat = confusionMat + get_confusionMat(logits, y_batch, 9)
            val_acc += top1[0]
            val_loss += loss.data.cpu().numpy()
        # print(confusionMat)

        val_acc /= (counter + 1)
        val_loss /= (counter + 1)
        print(
            f"Epoch {epoch}    Train accuracy {train_acc.item()}    Val accuracy: {val_acc.item()}    Train loss {train_loss}    Val loss: {val_loss}")
        # print('learning rate', scheduler.get_last_lr())
        train_loss_history[epoch] = train_loss
        val_loss_history[epoch] = val_loss
        train_acc_history[epoch] = train_acc
        val_acc_history[epoch] = val_acc

        model_epochs[epoch] = model
        optimizer_epochs[epoch] = optimizer

        end_time = time.time()
        print('time consumed:', end_time - start_time)

    if saveModel:
        checkpoint_name = 'finetune_checkpoint_{:04d}.pth.tar'.format(epoch)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model_epochs[epoch].state_dict(),
            'optimizer': optimizer_epochs[epoch].state_dict(),
        }, is_best=False, filename=os.path.join(args.save_dir_ft, checkpoint_name))

    best_logits = logits_epochs[epoch]
    print(np.sum(np.argmax(best_logits, axis=1)==y_all) / len(y_all))
    return epoch, train_loss_history, val_loss_history, train_acc_history, val_acc_history, confusionMat, logits_epochs, y_all

def train_earlyStopping_nono(args, train_loader, val_loader, model, criterion, optimizer, scheduler, saveModel,lr,wd):
    bad_count = 0
    best_acc, best_loss = -1, 1000

    val_acc = 0
    val_loss = 0
    model.eval()
    for counter, (x_batch, y_batch) in enumerate(val_loader):
        x_batch = x_batch.to(args.device)
        y_batch = y_batch.to(args.device)

        logits = model(x_batch)
        # print(logits.shape)
        # print(y_batch)
        # print(logits.shape, y_batch.shape)
        loss = criterion(logits, y_batch)

        top1 = accuracy(logits, y_batch, topk=(1,))
        val_acc += top1[0]
        val_loss += loss.data.cpu().numpy()

    val_acc /= (counter + 1)
    val_loss /= (counter + 1)
    print(
        f"Epoch {-1}     Val accuracy: {val_acc.item()}    Val loss: {val_loss}")

    train_loss_history, val_loss_history = np.zeros(args.epochs_finetune), np.zeros(args.epochs_finetune)
    train_acc_history, val_acc_history = np.zeros(args.epochs_finetune), np.zeros(args.epochs_finetune)

    model_epochs, optimizer_epochs = {}, {}
    for epoch in range(args.epochs_finetune):
        start_time = time.time()
        train_acc = 0
        train_loss = 0
        model.train()
        train_confusion = torch.zeros((2, 2))
        for counter, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(args.device)
            y_batch = y_batch.to(args.device)

            logits = model(x_batch)
            # print(logits.shape, y_batch.shape)
            loss = criterion(logits, y_batch)

            top1 = accuracy(logits, y_batch, topk=(1,))
            train_confusion = train_confusion + get_confusionMat(logits, y_batch, 2)
            train_acc += top1[0]
            train_loss += loss.data.cpu().numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if np.mod(counter+1, 100) == 0:
            #     print('counter', counter+1, 'train acc', train_acc.item() / counter+1)

        scheduler.step()
        # scheduler.step(val_loss)

        train_acc /= (counter + 1)
        train_loss /= (counter + 1)

        val_acc = 0
        val_loss = 0
        # confusionMat = torch.zeros((9,9))
        confusionMat = torch.zeros((2, 2))

        model.eval()
        for counter, (x_batch, y_batch) in enumerate(val_loader):
            x_batch = x_batch.to(args.device)
            y_batch = y_batch.to(args.device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            top1 = accuracy(logits, y_batch, topk=(1,))
           # confusionMat = confusionMat + get_confusionMat(logits, y_batch, 9)
            confusionMat = confusionMat + get_confusionMat(logits, y_batch, 2)
            val_acc += top1[0]
            val_loss += loss.data.cpu().numpy()
        # print(confusionMat)


        val_acc /= (counter + 1)
        val_loss /= (counter + 1)
        print(
            f"Epoch {epoch}    Train accuracy {train_acc.item()}    Val accuracy: {val_acc.item()}    Train loss {train_loss}    Val loss: {val_loss}")
        # print('learning rate', scheduler.get_last_lr())
        train_loss_history[epoch] = train_loss
        val_loss_history[epoch] = val_loss
        train_acc_history[epoch] = train_acc
        val_acc_history[epoch] = val_acc

        model_epochs[epoch] = model
        optimizer_epochs[epoch] = optimizer

        if val_acc > best_acc:
            # 在这里添加额外的条件检查
            if epoch > 20:
                bad_count = 0
                best_loss = val_loss
                best_acc = val_acc
                best_epoch = epoch
                best_confusion = confusionMat
                best_confusion_train = train_confusion
            # 如果当前epoch不大于20，则即使性能有所提升，也不会更新最佳模型
        else:
            bad_count += 1

        if bad_count > args.max_tol:
            print('best epoch %d, train loss: %.4f, val loss: %.4f, train acc: %.3f, val acc: %.3f' % (
                best_epoch, train_loss_history[best_epoch], val_loss_history[best_epoch],
                train_acc_history[best_epoch], val_acc_history[best_epoch]))
            break

        end_time = time.time()
        print('time consumed:', end_time - start_time)

    if saveModel:
        if not os.path.exists(args.save_dir_ft):
            os.mkdir(args.save_dir_ft)
        checkpoint_name = 'finetune_checkpoint_{:04d}_lr_{}_wd_{}_best.tar'.format(best_epoch,lr,wd)
        save_checkpoint({
            'epoch': best_epoch,
            'state_dict': model_epochs[best_epoch].state_dict(),
            'optimizer': optimizer_epochs[best_epoch].state_dict(),
        }, is_best=False, filename=os.path.join(args.save_dir_ft, checkpoint_name))

        checkpoint_name = 'finetune_checkpoint_{:04d}_lr_{}_wd_{}.pth.tar'.format(epoch,lr,wd)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model_epochs[epoch].state_dict(),
            'optimizer': optimizer_epochs[epoch].state_dict(),
        }, is_best=False, filename=os.path.join(args.save_dir_ft, checkpoint_name))

    if bad_count <=  args.max_tol:
        print('best epoch %d, train loss: %.4f, val loss: %.4f, train acc: %.3f, val acc: %.3f' % (
                best_epoch, train_loss_history[best_epoch], val_loss_history[best_epoch],
                train_acc_history[best_epoch], val_acc_history[best_epoch]))
    return best_epoch, train_loss_history, val_loss_history, train_acc_history, val_acc_history, best_confusion, best_confusion_train, confusionMat,train_confusion


def train_earlyStopping_valEnsemble(args, train_loader, val_loader, model, criterion, optimizer, scheduler, augTime):
    bad_count = 0
    best_acc, best_loss = -1, 1000

    val_acc = 0
    val_loss = 0
    model.eval()
    for counter, (x_batch, y_batch) in enumerate(val_loader):
        x_batch = x_batch.to(args.device)
        y_batch = y_batch.to(args.device)

        logits = model(x_batch)
        # print(logits.shape, y_batch.shape)
        loss = criterion(logits, y_batch)

        top1 = accuracy(logits, y_batch, topk=(1,))
        val_acc += top1[0]
        val_loss += loss.data.cpu().numpy()

    val_acc /= (counter + 1)
    val_loss /= (counter + 1)
    print(
        f"Epoch {-1}     Val accuracy: {val_acc.item()}    Val loss: {val_loss}")

    train_loss_history, val_loss_history = np.zeros(args.epochs_finetune), np.zeros(args.epochs_finetune)
    train_acc_history, val_acc_history = np.zeros(args.epochs_finetune), np.zeros(args.epochs_finetune)

    for epoch in range(args.epochs_finetune):
        start_time = time.time()
        train_acc = 0
        train_loss = 0
        model.train()
        for counter, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(args.device)
            y_batch = y_batch.to(args.device)

            logits = model(x_batch)
            # print(logits.shape, y_batch.shape)
            loss = criterion(logits, y_batch)

            top1 = accuracy(logits, y_batch, topk=(1,))
            train_acc += top1[0]
            train_loss += loss.data.cpu().numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if np.mod(counter+1, 100) == 0:
            #     print('counter', counter+1, 'train acc', train_acc.item() / counter+1)

        scheduler.step()

        train_acc /= (counter + 1)
        train_loss /= (counter + 1)

        val_acc = 0
        val_loss = 0
        model.eval()
        for counter, (x_batch, y_batch) in enumerate(val_loader):
            x_batch = x_batch.to(args.device)
            y_batch = y_batch.to(args.device)

            logits = model(x_batch)
            logits = logits.reshape(-1, augTime, 3).mean(axis=1)
            y_batch = y_batch[torch.arange(0, len(y_batch), augTime)]
            loss = criterion(logits, y_batch)

            top1 = accuracy(logits, y_batch, topk=(1,))
            val_acc += top1[0]
            val_loss += loss.data.cpu().numpy()

        val_acc /= (counter + 1)
        val_loss /= (counter + 1)
        print(
            f"Epoch {epoch}    Train accuracy {train_acc.item()}    Val accuracy: {val_acc.item()}    Train loss {train_loss}    Val loss: {val_loss}")
        # print('learning rate', scheduler.get_last_lr())
        train_loss_history[epoch] = train_loss
        val_loss_history[epoch] = val_loss
        train_acc_history[epoch] = train_acc
        val_acc_history[epoch] = val_acc


        if val_loss < best_loss:
            bad_count = 0
            best_loss = val_loss
            best_acc = val_acc
            best_epoch = epoch
        else:
            bad_count += 1

        if bad_count > args.max_tol:
            print('best epoch %d, train loss: %.4f, val loss: %.4f, train acc: %.3f, val acc: %.3f' % (
                best_epoch, train_loss_history[best_epoch], val_loss_history[best_epoch],
                train_acc_history[best_epoch], val_acc_history[best_epoch]))
            break

        end_time = time.time()
        print('time consumed:', end_time - start_time)

    if bad_count <=  args.max_tol:
        print('best epoch %d, train loss: %.4f, val loss: %.4f, train acc: %.3f, val acc: %.3f' % (
                best_epoch, train_loss_history[best_epoch], val_loss_history[best_epoch],
                train_acc_history[best_epoch], val_acc_history[best_epoch]))
    return best_epoch, train_loss_history, val_loss_history, train_acc_history, val_acc_history


def train(args, best_epoch, train_loader, val_loader, model, criterion, optimizer, scheduler,
          results_finetune, sub):
    for epoch in range(best_epoch + 1):
        train_acc = 0
        train_loss = 0
        model.train()
        for counter, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(args.device)
            y_batch = y_batch.to(args.device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            top1 = accuracy(logits, y_batch, topk=(1,))
            train_acc += top1[0]
            train_loss += loss.data.cpu().numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if np.mod(counter+1, 100) == 0:
            #     print('counter', counter+1, 'train acc', train_acc.item() / counter+1)
        scheduler.step()

        train_acc /= (counter + 1)
        train_loss /= (counter + 1)

        val_acc = 0
        val_loss = 0
        model.eval()
        for counter, (x_batch, y_batch) in enumerate(val_loader):
            x_batch = x_batch.to(args.device)
            y_batch = y_batch.to(args.device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            top1 = accuracy(logits, y_batch, topk=(1,))
            val_acc += top1[0]
            val_loss += loss.data.cpu().numpy()

        val_acc /= (counter + 1)
        val_loss /= (counter + 1)
        print(
            f"Epoch {epoch}    Train accuracy {train_acc.item()}    Test accuracy: {val_acc.item()}    Train loss {train_loss}    Test loss: {val_loss}")

        results_finetune['train_loss_history'][sub, epoch] = train_loss
        results_finetune['val_loss_history'][sub, epoch] = val_loss
        results_finetune['train_acc_history'][sub, epoch] = train_acc
        results_finetune['val_acc_history'][sub, epoch] = val_acc
    results_finetune['best_val_acc'][sub] = val_acc
    results_finetune['best_val_loss'][sub] = val_loss
    return results_finetune, model


def test(args, test_loader, model, criterion):
    test_acc = 0
    test_loss = 0
    for counter, (x_batch, y_batch) in enumerate(test_loader):
        x_batch = x_batch.to(args.device)
        y_batch = y_batch.to(args.device)

        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        top1 = accuracy(logits, y_batch, topk=(1,))
        test_acc += top1[0]
        test_loss += loss.data.cpu().numpy()

    test_acc /= (counter + 1)
    test_loss /= (counter + 1)
    print(f"\tTest acc {test_acc.item()}    Test loss: {test_loss}")
    return test_acc, test_loss


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # eq_out = (torch.abs(output - output[:, 0].repeat(output.shape[1], 1).t()) < 1e-5).sum(axis=1)
        # print(eq_out)
        # eq_num = (eq_out > 1).sum()
        # if eq_out.sum() > len(eq_out):
        #     print('Equal logits for different entries!')
        #     print(output[eq_out>1, :].shape, output[eq_out>1, :])

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            # if k == 1:
            #     correct_k = correct_k - eq_num
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_confusionMat(output, target, n_class):
    with torch.no_grad():
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()

        confusionMat = np.zeros((n_class, n_class))
        for i in range(n_class):
            for j in range(n_class):
                confusionMat[i, j] = torch.sum((pred==j) & (target==i))
        return confusionMat