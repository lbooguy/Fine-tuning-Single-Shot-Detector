import torch
import torchvision.utils
from torch.utils.data import DataLoader
from torchvision.models import VGG16_Weights
from torchvision.models.detection.ssd import det_utils
import utils
from engine import train_one_epoch, evaluate
from DATASET import RobotHand

if __name__ == '__main__':
    dataset_train = RobotHand(
        csv_file='./Image665/train_annotations_300_300/_annotations.csv',
        root="./Image665/train_300_300")
    dataset_valid = RobotHand(
        csv_file='./Image665/valid_annotations_300_300/_annotations.csv',
        root="./Image665/valid_300_300")

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=32, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)
    data_loader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=16, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torchvision.models.detection.ssd300_vgg16(weights="SSD300_VGG16_Weights.COCO_V1",
                                                      weights_backbone=VGG16_Weights.IMAGENET1K_FEATURES,
                                                      trainable_backbone_layers=2)
    num_classes = 2
    num_anchors = model.anchor_generator.num_anchors_per_location()
    in_channels = det_utils.retrieve_out_channels(model.backbone, (300, 300))
    model.head.classification_head = torchvision.models.detection.ssd.SSDClassificationHead(in_channels=in_channels,
                                                                                            num_anchors=num_anchors,
                                                                                            num_classes=num_classes)

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    num_epochs = 50
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the valid dataset
        evaluate(model, data_loader_valid, device=device)
        torch.save(model, "./SSD_trainable_backbone_layers_2.pt")
    torch.save(model, "./ssd_" + str(num_epochs) + "_epochs" + ".pt")
