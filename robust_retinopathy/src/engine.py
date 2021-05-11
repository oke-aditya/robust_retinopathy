# Adapter from Quickvision

import torch
from torch import nn
from torch.cuda import amp
import utils
import time
from collections import OrderedDict
from utils import accuracy


def train_step(model: nn.Module, train_loader, criterion,
               device: str, optimizer,
               scheduler=None, num_batches: int = None,
               log_interval: int = 100, grad_penalty: bool = False,
               scaler=None,):
    """
    Performs one step of training. Calculates loss, forward pass, computes gradient and returns metrics.
    Args:
        model : A pytorch CNN Model.
        train_loader : Train loader.
        criterion : Loss function to be optimized.
        device : "cuda" or "cpu"
        optimizer : Torch optimizer to train.
        scheduler : Learning rate scheduler.
        num_batches : (optional) Integer To limit training to certain number of batches.
        log_interval : (optional) Defualt 100. Integer to Log after specified batch ids in every batch.
        scaler: (optional)  Pass torch.cuda.amp.GradScaler() for fp16 precision Training.
    """

    model = model.to(device)
    start_train_step = time.time()
    metrics = OrderedDict()
    model.train()
    last_idx = len(train_loader) - 1
    batch_time_m = utils.AverageMeter()
    # data_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()
    cnt = 0
    batch_start = time.time()
    # num_updates = epoch * len(loader)

    for batch_idx, (inputs, target) in enumerate(train_loader):
        last_batch = batch_idx == last_idx
        # data_time_m.update(time.time() - batch_start)
        inputs = inputs.to(device)
        target = target.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        if scaler is not None:
            with amp.autocast():
                output = model(inputs)
                loss = criterion(output, target)
                # Scale the loss using Grad Scaler

            if grad_penalty is True:
                # Scales the loss for autograd.grad's backward pass, resulting in scaled grad_params
                scaled_grad_params = torch.autograd.grad(scaler.scale(loss),
                                                         model.parameters(), create_graph=True)
                # Creates unscaled grad_params before computing the penalty. scaled_grad_params are
                # not owned by any optimizer, so ordinary division is used instead of scaler.unscale_:
                inv_scale = 1.0 / scaler.get_scale()
                grad_params = [p * inv_scale for p in scaled_grad_params]
                # Computes the penalty term and adds it to the loss
                with amp.autocast():
                    grad_norm = 0
                    for grad in grad_params:
                        grad_norm += grad.pow(2).sum()

                    grad_norm = grad_norm.sqrt()
                    loss = loss + grad_norm

            scaler.scale(loss).backward()
            # Step using scaler.step()
            scaler.step(optimizer)
            # Update for next iteration
            scaler.update()

        else:
            output = model(inputs)
            loss = criterion(output, target)

            if grad_penalty is True:
                # Create gradients
                grad_params = torch.autograd.grad(loss, model.parameters(), create_graph=True)
                # Compute the L2 Norm as penalty and add that to loss
                grad_norm = 0
                for grad in grad_params:
                    grad_norm += grad.pow(2).sum()
                grad_norm = grad_norm.sqrt()
                loss = loss + grad_norm

            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        cnt += 1
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        top1_m.update(acc1.item(), output.size(0))
        top5_m.update(acc5.item(), output.size(0))
        losses_m.update(loss.item(), inputs.size(0))

        batch_time_m.update(time.time() - batch_start)
        batch_start = time.time()
        if last_batch or batch_idx % log_interval == 0:  # If we reach the log intervel
            print(
                "Batch Train Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                "Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  "
                "Top 1 Accuracy: {top1.val:>7.4f} ({top1.avg:>7.4f})  "
                "Top 5 Accuracy: {top5.val:>7.4f} ({top5.avg:>7.4f})".format(
                    batch_time=batch_time_m, loss=losses_m, top1=top1_m, top5=top5_m))

        if num_batches is not None:
            if cnt >= num_batches:
                end_train_step = time.time()
                metrics["loss"] = losses_m.avg
                metrics["top1"] = top1_m.avg
                metrics["top5"] = top5_m.avg
                print(f"Done till {num_batches} train batches")
                print(f"Time taken for train step = {end_train_step - start_train_step} sec")
                return metrics

    metrics["loss"] = losses_m.avg
    metrics["top1"] = top1_m.avg
    metrics["top5"] = top5_m.avg
    end_train_step = time.time()
    print(f"Time taken for train step = {end_train_step - start_train_step} sec")
    return metrics


def val_step(model: nn.Module, val_loader, criterion,
             device: str, num_batches=None,
             log_interval: int = 100):

    """
    Performs one step of validation. Calculates loss, forward pass and returns metrics.
    Args:
        model : A pytorch CNN Model.
        val_loader : Validation loader.
        criterion : Loss function to be optimized.
        device : "cuda" or "cpu"
        num_batches : (optional) Integer To limit validation to certain number of batches.
        log_interval : (optional) Defualt 100. Integer to Log after specified batch ids in every batch.
    """

    model = model.to(device)
    start_val_step = time.time()
    last_idx = len(val_loader) - 1
    batch_time_m = utils.AverageMeter()
    # data_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()
    cnt = 0
    model.eval()
    batch_start = time.time()
    metrics = OrderedDict()
    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(val_loader):
            last_batch = batch_idx == last_idx
            inputs = inputs.to(device)
            target = target.to(device)

            output = model(inputs)
            loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            reduced_loss = loss.data

            losses_m.update(reduced_loss.item(), inputs.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))
            batch_time_m.update(time.time() - batch_start)

            batch_start = time.time()

            if (last_batch or batch_idx % log_interval == 0):  # If we reach the log intervel
                print(
                    "Batch Inference Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                    "Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  "
                    "Top 1 Accuracy: {top1.val:>7.4f} ({top1.avg:>7.4f})  "
                    "Top 5 Accuracy: {top5.val:>7.4f} ({top5.avg:>7.4f})".format(
                        batch_time=batch_time_m, loss=losses_m, top1=top1_m, top5=top5_m))

            if num_batches is not None:
                if cnt >= num_batches:
                    end_val_step = time.time()
                    metrics["loss"] = losses_m.avg
                    metrics["top1"] = top1_m.avg
                    metrics["top5"] = top5_m.avg
                    print(f"Done till {num_batches} validation batches")
                    print(f"Time taken for validation step = {end_val_step - start_val_step} sec")
                    return metrics

        metrics["loss"] = losses_m.avg
        metrics["top1"] = top1_m.avg
        metrics["top5"] = top5_m.avg
        print("Finished the validation epoch")

    end_val_step = time.time()
    print(f"Time taken for validation step = {end_val_step - start_val_step} sec")
    return metrics
