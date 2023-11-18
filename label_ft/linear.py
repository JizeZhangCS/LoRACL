import time
import tqdm
import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch.distributed as dist

from utils.meters import AverageMeter, ProgressMeter, _collect_lora_grad, accuracy
from utils.ddp import concat_all_gather_interlace
from utils.valid import ret_metrics, cls_metrics

def valid(train_loader, iid_loader, ood_loader, model, criterion, epoch, args):
    model.eval()
    print("current used valid percentage:  " + str(args.valid_precent))
    num_train_samples = int(len(train_loader.dataset) * args.valid_precent)
    features_dim = 512 if args.arch == 'resnet18' else 2048
    device = next(model.parameters()).device
    is_root = dist.get_rank() == 0
    
    train_features = np.zeros((num_train_samples, features_dim), dtype=np.float16)
    train_labels = np.zeros(num_train_samples, dtype=np.int64)
    end_ind = 0
    for ind, (x, y) in tqdm(enumerate(train_loader)):
        with torch.no_grad():
            # features: output of layer4, before avgpool
            x = model.conv1(x.to(device))
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            features = model.layer4(x)
            features = torch.mean(features, dim=[-2, -1])
            features = nn.functional.normalize(features, dim=1)
            features = concat_all_gather_interlace(features)
            y = concat_all_gather_interlace(y.to(device))
            if not is_root:
                if (ind+1) * len(features) >= num_train_samples:
                    break
                continue
        y = y.cpu().numpy().astype(np.float16)
        features = features.cpu().numpy().astype(np.float16)
        begin_ind = end_ind
        end_ind = min(begin_ind + len(features), len(train_features))   # "drop_last=False", preventing shape issues, also for valid percent cutting
        train_features[begin_ind:end_ind, :] = features[:end_ind - begin_ind]
        train_labels[begin_ind:end_ind] = y[:end_ind - begin_ind]
        if end_ind >= num_train_samples:
            break

    if is_root:
        if args.classifier == 'sgd':
            cls = SGDClassifier(max_iter=1000, n_jobs=16, tol=1e-3).fit(train_features, train_labels)
        elif args.classifier == 'logistic':
            cls = LogisticRegression(max_iter=1000, n_jobs=16, tol=1e-3).fit(train_features, train_labels)
        elif args.classifier == 'retrieval':
            cls = NearestNeighbors(n_neighbors=min(20, train_features.shape[0]), algorithm='auto',
                                   n_jobs=-1, metric='correlation').fit(train_features)
        else:
            raise NotImplementedError()
    else:
        cls = None
        
    def _run_cls(dst_domain_pth, test_loader):
        print(f"Target domain {dst_domain_pth}")

        batch_time = AverageMeter('Time', ':6.5f')
        acc1 = AverageMeter('Acc@1', ':6.5f')
        acc10 = AverageMeter('Acc@10', ':6.5f')
        acc20 = AverageMeter('Acc@20', ':6.5f')
        precision1 = AverageMeter('p@1', ':6.5f')
        precision10 = AverageMeter('p@10', ':6.5f')
        precision20 = AverageMeter('p@20', ':6.5f')
        precision5 = AverageMeter('p@5', ':6.3f')
        precision15 = AverageMeter('p@15', ':6.3f')

        progress = ProgressMeter(
            len(test_loader),
            [batch_time, acc1, acc10, acc20, precision1, precision5, precision10, precision15, precision20],
            prefix=f"Train on {args.data} Test on {dst_domain_pth}")
        end = time.time()
        num_test_samples = len(test_loader.dataset)
        print(f'num_test_samples: {num_test_samples}')
        total_samples = 0
        all_features = []
        all_y = []
        for ind, (x, y) in enumerate(test_loader):
            with torch.no_grad():
                x = model.conv1(x.to(device))
                x = model.bn1(x)
                x = model.relu(x)
                x = model.maxpool(x)
                x = model.layer1(x)
                x = model.layer2(x)
                x = model.layer3(x)
                features = model.layer4(x)
                features = torch.mean(features, dim=[-2, -1])
                features = nn.functional.normalize(features, dim=1)
                features = concat_all_gather_interlace(features)
                y = concat_all_gather_interlace(y.to(device)).cpu()
                if not is_root:
                    continue
            features = features.cpu().numpy().astype(np.float16)
            y = y.numpy()
            total_samples += len(y)
            if total_samples > num_test_samples:
                diff = total_samples - num_test_samples
                features = features[:-diff]
                y = y[:-diff]
                all_features.append(features)
                all_y.append(y)
                print(f'diff {diff}')
            all_features.append(features)
            all_y.append(y)
            if args.classifier == 'retrieval':
                a1, a10, a20, p1, p5, p10, p15, p20 = ret_metrics(features, y, train_labels, cls)
            else:
                a1, a10, a20, p1, p5, p10, p15, p20 = cls_metrics(features, y, cls)
            acc1.update(a1, len(y))
            acc10.update(a10, len(y))
            acc20.update(a20, len(y))
            precision1.update(p1, len(y))
            precision10.update(p10, len(y))
            precision20.update(p20, len(y))
            precision5.update(p5, len(y))
            precision15.update(p15, len(y))
            batch_time.update(time.time() - end)
            end = time.time()

            if ind % 10 == 0:
                progress.display(ind)

        print('Final batch:')
        progress.display(len(test_loader))
        if args.gpu==0:
            print('Results:')
            print(','.join([
                            f'acc1={acc1.avg}',
                            f'acc10={acc10.avg}',
                            f'acc20={acc20.avg}',
                            f'precision1={precision1.avg}',
                            f'precision5={precision5.avg}',
                            f'precision10={precision10.avg}',
                            f'precision15={precision15.avg}',
                            f'precision20={precision20.avg}'
                            ]))
    _run_cls(args.iid_val_data, iid_loader)
    _run_cls(args.ood_val_data, ood_loader)