from math import ceil

import torch
from torch import Tensor
from torch.autograd import Variable

from cycle_gan import CycleGAN
from data_loader import TrainDataSet
from logger import logger
from utils import ensure_dir, get_opts
from torch.utils.data import DataLoader

project_root = "/Users/patel/Documents/Coursework/ECE285/Project/GTA-to-Cityscapes-Domain-Adaptation/"#"./"
data_root = project_root + "gta/images" #"./gta/images/"
models_prefix = project_root + "saved_models/"
images_prefix = project_root + "saved_images/"

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def train_cycle_gan(data_root, semi_supervised=False):
    opt = get_opts()

    ensure_dir(models_prefix)
    ensure_dir(images_prefix)

    cycle_gan = CycleGAN(device, models_prefix, opt["lr"], opt["b1"],
                         train=True, semi_supervised=semi_supervised)
    # data = DataLoader(data_root=data_root,
    #                   image_size=(opt['img_height'], opt['img_width']),
    #                   batch_size=opt['batch_size'])
    dataset = TrainDataSet(data_root=data_root, image_size=(opt['img_height'], opt['img_width']))
    print("dataset : ", dataset)
    dataLoader = DataLoader(dataset, batch_size=1)

    total_images = len(dataset.names)
    print("Total Training Images", total_images)

    total_batches = int(ceil(total_images / opt['batch_size']))

    for epoch in range(5):
        for i, data in enumerate(dataLoader):
            real_A, real_B = data

            real_A = Variable(real_A.type(Tensor))
            real_B = Variable(real_B.type(Tensor))

            cycle_gan.set_input(real_A, real_B)
            cycle_gan.train()

            message = (
                "\r[Epoch {}/{}] [Batch {}/{}] [DA:{}, DB:{}] [GA:{}, GB:{}, cycleA:{}, cycleB:{}, G:{}]"
                    .format(epoch, opt["n_epochs"], iteration, total_batches,
                            cycle_gan.loss_disA.item(),
                            cycle_gan.loss_disB.item(),
                            cycle_gan.loss_genA.item(),
                            cycle_gan.loss_genB.item(),
                            cycle_gan.loss_cycle_A.item(),
                            cycle_gan.loss_cycle_B.item(),
                            cycle_gan.loss_G))
            print(message)
            logger.info(message)

        #     if iteration % opt['sample_interval'] == 0:
        #         cycle_gan.save_progress(images_prefix, epoch, iteration)
        # cycle_gan.save_progress(images_prefix, epoch, total_batches, save_epoch=True)



    # for epoch in range(cycle_gan.epoch_tracker.epoch, opt['n_epochs']):
    #     for iteration in range(total_batches):

    #         if (epoch == cycle_gan.epoch_tracker.epoch and
    #                     iteration < cycle_gan.epoch_tracker.iter):
    #             continue

    #         y, x = next(data.data_generator(iteration))

    #         real_A = Variable(x.type(Tensor))
    #         real_B = Variable(y.type(Tensor))

    #         cycle_gan.set_input(real_A, real_B)
    #         cycle_gan.train()

    #         message = (
    #             "\r[Epoch {}/{}] [Batch {}/{}] [DA:{}, DB:{}] [GA:{}, GB:{}, cycleA:{}, cycleB:{}, G:{}]"
    #                 .format(epoch, opt["n_epochs"], iteration, total_batches,
    #                         cycle_gan.loss_disA.item(),
    #                         cycle_gan.loss_disB.item(),
    #                         cycle_gan.loss_genA.item(),
    #                         cycle_gan.loss_genB.item(),
    #                         cycle_gan.loss_cycle_A.item(),
    #                         cycle_gan.loss_cycle_B.item(),
    #                         cycle_gan.loss_G))
    #         print(message)
    #         logger.info(message)

    #         if iteration % opt['sample_interval'] == 0:
    #             cycle_gan.save_progress(images_prefix, epoch, iteration)
    #     cycle_gan.save_progress(images_prefix, epoch, total_batches, save_epoch=True)


if __name__ == "__main__":
    train_cycle_gan(data_root)
