import torch
from torch import Tensor
from torch.autograd import Variable

# from cycle_gan import CycleGAN
# from data_loader_test import DataLoader
# from data_loader import TestDataSet
from logger import logger
# from utils import ensure_dir, get_opts
from torch.utils.data import DataLoader

import sys
sys.path.insert(1, './../')

from model.cycle_gan import CycleGAN
from data.data_loader import TestDataSet
from scripts.utils import ensure_dir, get_opts


project_root = "./../"
data_root = project_root + "data/" #"/Users/patel/Documents/Coursework/ECE285/Project/GTA-to-Cityscapes-Domain-Adaptation/my_input/"
models_prefix = project_root + "model/saved_models/test_"
images_prefix = project_root + "data/my_saved_images/"

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def test_cycle_gan(semi_supervised=True):
    opt = get_opts()

    ensure_dir(models_prefix)
    ensure_dir(images_prefix)

    cycle_gan = CycleGAN(device, models_prefix, opt["lr"], opt["b1"], train=False, 
                        semi_supervised=semi_supervised)

    dataset = TestDataSet(data_root=data_root, image_size=(opt['img_height'], opt['img_width']))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


    total_images = len(dataset.names)
    print("\n\nTotal images : ", len(dataset.test_names))
    print("Total Testing Images", total_images)

    loss_A = 0.0
    loss_B = 0.0
    name_loss_A = []
    name_loss_B = []

    for i,data in enumerate(dataloader):
        print(i, "/", total_images)
        name = dataset.names[i]
        x = data
        real_A = Variable(x.type(Tensor))

        cycle_gan.set_input(real_A, real_A)
        cycle_gan.test()
        cycle_gan.save_image(images_prefix, name)
        loss_A += cycle_gan.test_A
        loss_B += cycle_gan.test_B
        name_loss_A.append((cycle_gan.test_A, name))
        name_loss_B.append((cycle_gan.test_B, name))


    info = "Average Loss A:{} B :{}".format(loss_A/(1.0 * total_images), loss_B/(1.0 * total_images))
    print(info)
    logger.info(info)
    name_loss_A = sorted(name_loss_A)
    name_loss_B = sorted(name_loss_B)
    print("top 10 images")
    print(name_loss_A[:10])
    print(name_loss_B[:10])


if __name__ == "__main__":
    test_cycle_gan()
