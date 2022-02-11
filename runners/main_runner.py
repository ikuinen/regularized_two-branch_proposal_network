import collections
import logging
import os

import numpy as np
import torch
from fairseq.utils import move_to_cuda

from utils import AverageMeter, TimeMeter


# | loss 0.5513 | mIoU 0.3620 | IoU@0.1 0.7207 | IoU@0.3 0.5845 | IoU@0.5 0.3227 | IoU@0.7 0.1349 | IoU@0.9 0.0481 |
# | loss 0.5797 | mIoU 0.3655 | IoU@0.1 0.7277 | IoU@0.3 0.5893 | IoU@0.5 0.3262 | IoU@0.7 0.1358 | IoU@0.9 0.0481 |
# | loss 0.5477 | mIoU 0.3673 | IoU@0.1 0.7327 | IoU@0.3 0.5940 | IoU@0.5 0.3262 | IoU@0.7 0.1339 | IoU@0.9 0.0488 |

class MainRunner:
    def __init__(self, args):
        self.args = args
        self._build_dataset()
        self._build_model()
        if 'train' in args:
            self._build_optimizer()
            self.num_updates = 0

        self.num_clips = args['dataset']['max_num_frames'] // args['dataset']['target_stride']
        # self.props = self.test_set.props

    def train(self):

        for bias in [0.0]:
            logging.info('bias = {}.'.format(bias))
            for epoch in range(1, 20):
                logging.info('Start Epoch {}'.format(epoch))
                self.model_saved_path = self.args['train']['model_saved_path']
                os.makedirs(self.model_saved_path, mode=0o755, exist_ok=True)
                save_path = os.path.join(self.model_saved_path, 'model-{}.pt'.format(epoch))

                self._train_one_epoch(epoch, bias=bias)
                self._save_model(save_path)
                self.eval(bias=bias)
                self.eval(bias=bias, top_n=5, thresh=0.45)
                logging.info('=' * 60)

            print('-' * 120)
        logging.info('Done.')

    def _train_one_epoch(self, epoch, **kwargs):
        self.model.train()

        def print_log():
            msg = 'Epoch {}, Batch {}, lr = {:.5f}, '.format(epoch, bid, curr_lr)
            for k, v in loss_meter.items():
                msg += '{} = {:.4f}, '.format(k, v.avg)
                v.reset()
            msg += '{:.3f} seconds/batch'.format(1.0 / time_meter.avg)
            logging.info(msg)

        from models.weakly.loss import weakly_supervised_loss

        display_n_batches, bid = 50, 0
        time_meter = TimeMeter()
        loss_meter = collections.defaultdict(lambda: AverageMeter())
        if self.args['dataset']['dataset'] == 'ActivityNet':
            num_cands = 153
            fp = open('append.log', encoding='utf8', mode='a')
        else:
            num_cands = 200
            fp = open('append2.log', encoding='utf8', mode='a')
            # 31self.args['train']['topK'] = 48
        # if self.num_updates < 50:
        #     self.args['train']['topK'] = 195
        # else:
        #     self.args['train']['topK'] = 48
        for bid, batch in enumerate(self.train_loader, 1):
            self.optimizer.zero_grad()
            net_input = move_to_cuda(batch['net_input'])
            net_input['props'] = net_input['props'].expand(len(self.device_ids), -1, -1)
            net_input['props_graph'] = net_input['props_graph'].expand(len(self.device_ids), -1, -1)
            output = self.model(**net_input, get_negative=True, **kwargs)
            # Act: 153, 32,
            # Char: 100, 32
            loss, _ = weakly_supervised_loss(pos_score=output['score'],
                                             neg_score1=output['inter_neg']['neg_score'],
                                             neg_score2=output['intra_neg']['neg_score'],
                                             neg_weight2=output['intra_neg']['weight'],
                                             weight_gt=net_input['frame_gt'],
                                             props=net_input['props'][0],
                                             log_fp=fp, num_cands=num_cands,
                                             loss_meter=loss_meter, **self.args['train'])
            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)

            # update
            self.optimizer.step()
            self.num_updates += 1
            curr_lr = self.lr_scheduler.step_update(self.num_updates)
            time_meter.update()
            loss_meter['loss'].update(loss.item())

            if bid % display_n_batches == 0:
                print_log()

        if bid % display_n_batches != 0:
            print_log()

        fp.write('=' * 60 + '\n')
        fp.flush()
        fp.close()

    def eval(self, top_n=1, thresh=None, **kwargs):
        self.model.eval()
        metrics_logger = collections.defaultdict(lambda: AverageMeter())
        from models.weakly.loss import bce_rescale_loss

        with torch.no_grad():
            for bid, batch in enumerate(self.test_loader, 1):
                net_input = move_to_cuda(batch['net_input'])
                net_input['props'] = net_input['props'].expand(len(self.device_ids), -1, -1)
                net_input['props_graph'] = net_input['props_graph'].expand(len(self.device_ids), -1, -1)
                # forward
                output = self.model(**net_input, get_negative=True, **kwargs)
                durations = np.asarray([i[1] for i in batch['raw']])
                gt = np.asarray([i[2] for i in batch['raw']])

                loss, prob = bce_rescale_loss(output['score'], net_input['map_gt'])
                from models.weakly.loss import weakly_supervised_loss_fuck
                neg, pos = weakly_supervised_loss_fuck(pos_score=output['score'],
                                                       neg_score1=output['inter_neg']['neg_score'],
                                                       neg_score2=output['intra_neg']['neg_score'],
                                                       neg_weight2=output['intra_neg']['weight'],
                                                       weight_gt=net_input['frame_gt'],
                                                       props=net_input['props'][0],
                                                       log_fp=None, num_cands=153,
                                                       loss_meter=None, **self.args['train'])
                metrics_logger['loss'].update(loss.item())
                bsz = prob.size(0)
                prob = np.reshape(prob.cpu().numpy(), [bsz, -1])
                idx = np.argmax(prob, -1)

                idx1 = np.argmax(np.reshape(output['intra_neg']['neg_score'].cpu().numpy(), [bsz, -1]), -1)
                selected_props = self.test_set.props[idx]  # [bsz, 2]
                neg_props = self.test_set.props[idx1]

                weight = output['intra_neg']['weight'].cpu().numpy()[:, :, 0]
                if top_n > 1:
                    num_clips = self.num_clips
                    sort_idx = np.argsort(-prob, -1)
                    cand_props = list(self.test_set.props[sort_idx])  # [bsz, cand_props, 2]
                    top_n_selected_props = [selected_props]

                    for it in range(1, top_n):
                        ptr_props = top_n_selected_props[-1]
                        selected_props = []
                        for i in range(bsz):
                            p2 = cand_props[i]
                            p1 = np.repeat(np.expand_dims(ptr_props[i], 0),
                                           p2.shape[0], 0)

                            iou = calculate_IoU_batch2((p1[:, 0], p1[:, 1]), (p2[:, 0], p2[:, 1]))
                            keep = iou <= thresh
                            # print(keep.shape, cand_props[i].shape)
                            cand_props[i] = cand_props[i][keep]
                            # print(cand_props[i].shape)
                            selected_props.append(cand_props[i][0])
                        top_n_selected_props[-1] = top_n_selected_props[-1] * durations[:, np.newaxis] / num_clips
                        # print(np.asarray(selected_props).shape, selected_props[0].shape)
                        top_n_selected_props.append(np.asarray(selected_props))

                    top_n_selected_props[-1] = top_n_selected_props[-1] * durations[:, np.newaxis] / num_clips
                    # top_n_selected_props = np.asarray(top_n_selected_props)
                    res = top_n_metric(top_n_selected_props, gt)
                else:
                    ori_props = selected_props
                    selected_props = selected_props * durations[:, np.newaxis] / self.num_clips
                    neg_props = neg_props * durations[:, np.newaxis] / self.num_clips

                    res, iou = top_1_metric(selected_props, gt)
                    neg_res, neg_iou = top_1_metric(neg_props, gt)
                for k, v in res.items():
                    metrics_logger[k].update(v, bsz)

        for k, v in metrics_logger.items():
            print('| {} {:.4f}'.format(k, v.avg), end=' ')
        print('|')
        return metrics_logger

    def _build_dataset(self):
        import datasets as da
        from gensim.models import KeyedVectors
        from torch.utils.data import DataLoader
        args = self.args['dataset']
        cls = getattr(da, args['dataset'], None)
        vocab = KeyedVectors.load_word2vec_format(args['vocab_path'], binary=True)
        self.train_set = cls(data_path=args['train_data'], vocab=vocab, args=args, is_training=True)
        self.test_set = cls(data_path=args['test_data'], vocab=vocab, args=args)
        self.val_set = cls(data_path=args['val_data'], vocab=vocab, args=args) if args['val_data'] else None
        logging.info('train: {} samples, test: {} samples'.format(len(self.train_set), len(self.test_set)))
        batch_size = self.args['train']['batch_size']
        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True,
                                       collate_fn=self.train_set.collate_data, num_workers=1)
        self.test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=False,
                                      collate_fn=self.test_set.collate_data,
                                      num_workers=4)
        self.val_loader = DataLoader(self.val_set, batch_size=batch_size, shuffle=False,
                                     collate_fn=self.val_set.collate_data,
                                     num_workers=4) if args['val_data'] else None

    def _build_model(self):
        model_config = self.args['model']
        import models

        device_ids = list(range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))))
        logging.info('GPU: {}'.format(device_ids))
        self.model = getattr(models, model_config['name'], None)(model_config)
        self.model = self.model.cuda(device_ids[0])
        print(self.model)
        self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
        self.device_ids = device_ids

    def _build_optimizer(self):
        from optimizers import AdamOptimizer
        from optimizers.lr_schedulers import InverseSquareRootSchedule
        parameters = list(self.model.parameters())
        args = self.args['train']
        self.optimizer = AdamOptimizer(args, parameters)
        self.lr_scheduler = InverseSquareRootSchedule(args, self.optimizer)

    def _save_model(self, path):
        state_dict = {
            'num_updates': self.num_updates,
            'config': self.args,
            'model_parameters': self.model.state_dict(),
        }
        torch.save(state_dict, path)
        logging.info('save model to {}, num_updates {}.'.format(path, self.num_updates))

    def _load_model(self, path):
        # path = os.path.join(self.args.model_saved_path, name)
        state_dict = torch.load(path)
        self.num_updates = state_dict['num_updates']
        self.lr_scheduler.step_update(self.num_updates)
        parameters = state_dict['model_parameters']
        self.model.load_state_dict(parameters)
        logging.info('load model from {}, num_updates {}.'.format(path, self.num_updates))


def calculate_IoU(i0, i1):
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))

    if union[1] - union[0] < -1e-5:
        return 0
    iou = 1.0 * (inter[1] - inter[0] + 1e-10) / (union[1] - union[0] + 1e-10)
    return iou if iou >= 0.0 else 0.0


def calculate_IoU_batch1(i0, i1):
    union = (np.min(np.stack([i0[0], i1[0]], 0), 0), np.max(np.stack([i0[1], i1[1]], 0), 0))
    inter = (np.max(np.stack([i0[0], i1[0]], 0), 0), np.min(np.stack([i0[1], i1[1]], 0), 0))
    # iou = 1.0 * (inter[1] - inter[0] + 1e-10) / (union[1] - union[0] + 1e-10)
    iou = 1.0 * (inter[1] - inter[0] + 1) / (union[1] - union[0] + 1)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou


def calculate_IoU_batch2(i0, i1):
    union = (np.min(np.stack([i0[0], i1[0]], 0), 0), np.max(np.stack([i0[1], i1[1]], 0), 0))
    inter = (np.max(np.stack([i0[0], i1[0]], 0), 0), np.min(np.stack([i0[1], i1[1]], 0), 0))
    # iou = 1.0 * (inter[1] - inter[0] + 1e-10) / (union[1] - union[0] + 1e-10)
    iou = 1.0 * (inter[1] - inter[0] + 1e-10) / (union[1] - union[0] + 1e-10)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou


# [nb, 2], [nb, 2]
def top_n_metric(preds, label):
    result = {}
    bsz = preds[0].shape[0]
    top_iou = []
    for pred in preds:
        iou = calculate_IoU_batch1((pred[:, 0], pred[:, 1]), (label[:, 0], label[:, 1]))
        top_iou.append(iou)
    iou = np.max(np.stack(top_iou, 1), 1)
    result['mIoU'] = np.mean(iou)
    for i in range(1, 10, 2):
        result['IoU@0.{}'.format(i)] = 1.0 * np.sum(iou >= i / 10) / bsz
    return result


def top_1_metric(pred, label):
    result = {}
    bsz = pred.shape[0]
    iou = calculate_IoU_batch2((pred[:, 0], pred[:, 1]), (label[:, 0], label[:, 1]))
    result['mIoU'] = np.mean(iou)
    for i in range(1, 10, 2):
        result['IoU@0.{}'.format(i)] = 1.0 * np.sum(iou >= i / 10) / bsz
    return result, iou
