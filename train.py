import torch
import torch.optim as optim
import torch.nn as nn
import model_factory as model_no
import adversarial as ad
import numpy as np
import argparse
from data_list import ImageList
import pre_process as prep
import criterion_factory as cf
import sampler

def test_target(loader, model):
    with torch.no_grad():
        start_test = True
        iter_val = [iter(loader['val' + str(i)]) for i in range(10)]
        for i in range(len(loader['val0'])):
            data = [iter_val[j].next() for j in range(10)]
            inputs = [data[j][0] for j in range(10)]
            labels = data[0][1]
            for j in range(10):
                inputs[j] = inputs[j].cuda()
            labels = labels.cuda()
            outputs = []
            for j in range(10):
                _, output = model(inputs[j])
                outputs.append(output)
            outputs = sum(outputs)
            if start_test:
                all_output = outputs.data.float()
                all_label = labels.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)
        _, predict = torch.max(all_output, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy


def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma, power, init_lr=0.001):
    lr = init_lr * (1 + gamma * iter_num) ** (-power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        i += 1
    return optimizer

class discriminatorDANN(nn.Module):
    def __init__(self, feature_len):
        super(discriminatorDANN, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.ad_layer1 = nn.Linear(feature_len, 1024)
        self.ad_layer1.weight.data.normal_(0, 0.01)
        self.ad_layer1.bias.data.fill_(0.0)
        self.fc1 = nn.Sequential(self.ad_layer1, nn.ReLU(), nn.Dropout(0.5))

        self.ad_layer2 = nn.Linear(1024, 1024)
        self.ad_layer2.weight.data.normal_(0, 0.01)
        self.ad_layer2.bias.data.fill_(0.0)

        self.ad_layer3 = nn.Linear(1024, 1)
        self.ad_layer3.weight.data.normal_(0, 0.3)
        self.ad_layer3.bias.data.fill_(0.0)
        self.fc2_3 = nn.Sequential(self.ad_layer2, nn.ReLU(), nn.Dropout(0.5), self.ad_layer3, nn.Sigmoid())

    def forward(self, x, y):
        f2 = self.fc1(x)
        f = self.fc2_3(f2)
        return f


class discriminatorCDAN(nn.Module):
    def __init__(self, feature_len):
        super(discriminatorCDAN, self).__init__()
        self.ad_layer1 = nn.Linear(feature_len * 31, 1024)
        self.ad_layer1.weight.data.normal_(0, 0.01)
        self.ad_layer1.bias.data.fill_(0.0)
        self.fc1 = nn.Sequential(self.ad_layer1, nn.ReLU(), nn.Dropout(0.5))

        self.ad_layer2 = nn.Linear(1024, 1024)
        self.ad_layer2.weight.data.normal_(0, 0.01)
        self.ad_layer2.bias.data.fill_(0.0)
        self.ad_layer3 = nn.Linear(1024, 1)
        self.ad_layer3.weight.data.normal_(0, 0.3)
        self.ad_layer3.bias.data.fill_(0.0)
        self.fc2_3 = nn.Sequential(self.ad_layer2, nn.ReLU(), nn.Dropout(0.5), self.ad_layer3, nn.Sigmoid())

    def forward(self, x, y):
        op_out = torch.bmm(y.unsqueeze(2), x.unsqueeze(1))
        ad_in = op_out.view(-1, y.size(1) * x.size(1))
        f2 = self.fc1(ad_in)
        f = self.fc2_3(f2)
        return f


class predictor(nn.Module):
    def __init__(self, feature_len, cate_num):
        super(predictor, self).__init__()
        self.classifier = nn.Linear(feature_len, cate_num)
        self.classifier.weight.data.normal_(0, 0.01)
        self.classifier.bias.data.fill_(0.0)

    def forward(self, features):
        activations = self.classifier(features)
        return (activations)


class net(nn.Module):
    def __init__(self, feature_len):
        super(net, self).__init__()
        self.model_fc = model_no.Resnet50Fc()
        self.bottleneck_0 = nn.Linear(feature_len, 256)
        self.bottleneck_0.weight.data.normal_(0, 0.005)
        self.bottleneck_0.bias.data.fill_(0.1)
        self.bottleneck_layer = nn.Sequential(self.bottleneck_0, nn.ReLU(), nn.Dropout(0.5))
        self.classifier_layer = predictor(256, num_categories)

    def forward(self, x):
        features = self.model_fc(x)
        out_bottleneck = self.bottleneck_layer(features)
        logits = self.classifier_layer(out_bottleneck)
        return (out_bottleneck, logits)

    def get_parameters(self):
        parameter_list = [{"params": self.model_fc.parameters(), "lr_mult": 0.1}, \
                          {"params": self.bottleneck_layer.parameters(), "lr_mult": 1}, \
                          {"params": self.classifier_layer.parameters(), "lr_mult": 1}]
        return parameter_list


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Transfer Learning')

    parser.add_argument('--source', type=str, nargs='?', default='c', help="source dataset")
    parser.add_argument('--target', type=str, nargs='?', default='p', help="target dataset")
    parser.add_argument('--lr', type=float, nargs='?', default=0.03, help="target dataset")
    parser.add_argument('--max_iteration', type=float, nargs='?', default=12500, help="target dataset")
    parser.add_argument('--k', type=int, default=3, help="k")
    parser.add_argument('--test_batch_size', type=int, default=4, help="test batch size")
    parser.add_argument('--n_samples', type=int, default=2, help='number of samples from each src class')
    parser.add_argument('--multi_gpu', type=int, default=0, help="use dataparallel if 1")
    parser.add_argument('--msc_coeff', type=float, default=1.0, help="coeff for similarity loss")
    parser.add_argument('--pre_train', type=int, default=0, help="number of iterations to pretrain for")
    parser.add_argument('--ila_switch_iter', type=int, default=0,
                        help="number of iterations when only DA loss works and sim doesn't")
    parser.add_argument('--method', type=str, nargs='?', default='DANN', choices=['DANN', 'CDAN'])
    parser.add_argument('--mu', type=int, default=80,
                        help="these many target samples are used finally, eg. 2/3 of batch") #mu in number

    args = parser.parse_args() 

    args.multi_gpu = bool(args.multi_gpu)
    num_categories = 31

    file_path = {
        "i": "/vulcan-pvc1/ml_for_da_pan_base/fg_dataset_list/bird31_ina_list_2017.txt",
        "n": "/vulcan-pvc1/ml_for_da_pan_base/fg_dataset_list/bird31_nabirds_list.txt",
        "c": "/vulcan-pvc1/ml_for_da_pan_base/fg_dataset_list/bird31_cub2011.txt",
    }

    dataset_source = file_path[args.source]
    dataset_target = dataset_test = file_path[args.target]

    batch_size = {"train": args.n_samples * num_categories, "val": args.n_samples * num_categories, "test": args.test_batch_size}
    for i in range(10):
        batch_size["val" + str(i)] = 4

    src_train_sampler = sampler.get_sampler({
        'path'              : dataset_source,
        'n_classes'         : num_categories,
        'n_samples'         : args.n_samples,
    })

    dataset_loaders = {}

    dataset_list = ImageList(open(dataset_source).readlines(),
                             transform=prep.image_train(resize_size=256, crop_size=224))

    dataset_loaders["train"] = torch.utils.data.DataLoader(dataset_list, batch_sampler=src_train_sampler, \
                                                               shuffle=False, num_workers=16)

    dataset_list = ImageList(open(dataset_target).readlines(),
                             transform=prep.image_train(resize_size=256, crop_size=224))
    dataset_loaders["val"] = torch.utils.data.DataLoader(dataset_list, batch_size=batch_size['train'], shuffle=True,
                                                         num_workers=16, drop_last=True)


    prep_dict_test = prep.image_test_10crop(resize_size=256, crop_size=224)
    for i in range(10):
        dataset_list = ImageList(open(dataset_test).readlines(), transform=prep_dict_test["val" + str(i)])
        dataset_loaders["val" + str(i)] = torch.utils.data.DataLoader(dataset_list,
                                                                      batch_size=batch_size["val" + str(i)],
                                                                      shuffle=False, num_workers=16)

    # network construction
    feature_len = 2048
    my_net = net(feature_len)
    my_net = my_net.cuda()

    if args.multi_gpu:
        print('USING MULTI GPU')
        my_net = nn.DataParallel(my_net)
        my_net_accr = my_net.module
    else: 
        print('NOT USING MULTI GPU')
        my_net_accr = my_net

    my_net.train(True)

    if args.method == 'DANN':
        my_discriminator = discriminatorDANN(256)
    elif args.method == 'CDAN':
        my_discriminator = discriminatorCDAN(256)
    else:
        raise Exception('{} not implemented'.format(args.method))

    # domain discriminator
    my_discriminator = my_discriminator.cuda()
    my_discriminator.train(True)

    # gradient reversal layer
    my_grl = ad.AdversarialLayer()

    msc_config = {
        'k'      : args.k,
        'm'      : args.n_samples,
        'mu'   : args.mu,
    }
    msc_module = cf.MSCLoss(msc_config)
    msc_module = msc_module.cuda()

    # criterion and optimizer
    criterion = {
        "classifier" : nn.CrossEntropyLoss(),
        "adversarial": nn.BCELoss()
    }

    optimizer_dict = [
        {"params": filter(lambda p: p.requires_grad, my_net_accr.model_fc.parameters()), "lr": 0.1},
        {"params": filter(lambda p: p.requires_grad, my_net_accr.bottleneck_layer.parameters()), "lr": 1},
        {"params": filter(lambda p: p.requires_grad, my_net_accr.classifier_layer.parameters()), "lr": 1},
        {"params": filter(lambda p: p.requires_grad, my_discriminator.parameters()), "lr": 1}  # ,
    ]

    optimizer = optim.SGD(optimizer_dict, lr=0.1, momentum=0.9, weight_decay=0.0005)

    optimizer_dict_pre = [
        {"params": filter(lambda p: p.requires_grad, my_net_accr.model_fc.parameters()), "lr": 0.1},
        {"params": filter(lambda p: p.requires_grad, my_net_accr.bottleneck_layer.parameters()), "lr": 1},
        {"params": filter(lambda p: p.requires_grad, my_net_accr.classifier_layer.parameters()), "lr": 1}
    ]

    optimizer_pre = optim.SGD(optimizer_dict_pre, lr=0.1, momentum=0.9, weight_decay=0.0005)

    param_lr_pre = []
    for param_group in optimizer_pre.param_groups:
        param_lr_pre.append(param_group["lr"])

    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])

    len_source = len(dataset_loaders["train"]) - 1
    len_target = len(dataset_loaders["val"]) - 1
    iter_source = iter(dataset_loaders["train"])
    iter_target = iter(dataset_loaders["val"])

    # pre_train
    for pt in range(1, args.pre_train + 1):
        print('pre_train iter {}'.format(pt))
        my_net.train(True)
        optimizer_pre = inv_lr_scheduler(param_lr_pre, optimizer_pre, pt, init_lr=args.lr, gamma=0.001, power=0.75)
        optimizer_pre.zero_grad()
        if pt % len_source == 0:
            iter_source = iter(dataset_loaders["train"])
        data_source = iter_source.next()
        inputs_source, labels_source = data_source
        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()

        all_features, logits = my_net(inputs_source)
        classifier_loss = criterion["classifier"](logits, labels_source)
        classifier_loss.backward()
        optimizer_pre.step()

    len_source = len(dataset_loaders["train"]) - 1
    len_target = len(dataset_loaders["val"]) - 1
    iter_source = iter(dataset_loaders["train"])
    iter_target = iter(dataset_loaders["val"])

    for iter_num in range(1, args.max_iteration + 1):
        my_net.train(True)
        optimizer = inv_lr_scheduler(param_lr, optimizer, iter_num, init_lr=args.lr, gamma=0.001, power=0.75)
        optimizer.zero_grad()

        if iter_num % len_source == 0:
            iter_source = iter(dataset_loaders["train"])
        if iter_num % len_target == 0:
            iter_target = iter(dataset_loaders["val"])
        data_source = iter_source.next()
        data_target = iter_target.next()
        inputs_source, labels_source = data_source
        inputs_target, labels_target = data_target
        inputs = torch.cat((inputs_source, inputs_target), dim=0)
        inputs = inputs.cuda()
        labels_target = labels_target.cuda()

        labels_source = labels_source.cuda()
        domain_labels = torch.from_numpy(
            np.array([[1], ] * len(inputs_source)+ [[0], ] * len(inputs_target))).float()
        domain_labels = domain_labels.cuda()

        all_features, logits = my_net(inputs)
        src_logits = logits[:len(inputs_source)]
        src_features = all_features[:len(inputs_source)]
        tgt_features = all_features[len(inputs_source):]
        classifier_loss = criterion["classifier"](src_logits, labels_source)

        max_iter = args.max_iteration
        domain_predicted = my_discriminator(my_grl.apply(all_features), nn.softmax(dim=1)(logits).detach())
        transfer_loss = nn.bceloss()(domain_predicted, domain_labels)

        total_loss = classifier_loss + transfer_loss

        if iter_num > args.ila_switch_iter:
            msc_loss = msc_module(src_features, labels_source, tgt_features)
            total_loss += (args.msc_coeff * msc_loss)

        total_loss.backward()
        optimizer.step()

        # test
        test_interval = 500
        if iter_num % test_interval == 0:
            my_net.eval()
            test_acc = test_target(dataset_loaders, my_net)
            stats = 'iter: {:05d}, test_acc:{:.4f}\n'.format(iter_num, test_acc)
            print(stats)