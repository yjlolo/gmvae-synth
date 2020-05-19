import numpy as np
import torch
from base import BaseTrainer


class GMVAE_Trainer(BaseTrainer):
    def __init__(self, model, loss, metrics, optimizer, resume, config,
                 data_loader, label_portion=1,
                 valid_data_loader=None, lr_scheduler=None, train_logger=None):
        super(GMVAE_Trainer, self).__init__(model, loss, metrics, optimizer, resume, config, train_logger)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        assert label_portion <= 1 and label_portion >= 0
        self.label_portion = label_portion
        self._sample_label()

    def _sample_label(self):
        if self.label_portion == 0:
            self.label_idx = None
        elif self.label_portion == 1:
            self.label_idx = self.data_loader.train_idx
        else:
            dict_cat_inst_idx = self.data_loader.dataset.dict_cat_inst_idx
            dict_cat_inst_ratio = self.data_loader.dataset.dict_cat_inst_ratio
            train_idx = self.data_loader.train_idx
            n_train = len(train_idx)
            n_label = int(n_train * self.label_portion)
            # figure out idx of training data for each instrument
            for k, v in dict_cat_inst_idx.items():
                dict_cat_inst_idx[k] = np.intersect1d(v, train_idx)
            # get idx of the luckily selected samples that will have labels
            label_idx = np.array([])
            for k, v in dict_cat_inst_ratio.items():
                np.random.seed(111)
                label_idx = np.hstack([label_idx, np.random.choice(dict_cat_inst_idx[k], int(v * n_label), replace=False)])
            self.label_idx = label_idx

    def _compute_loss(self, epoch, x, x_predict, q_mu, q_logvar, log_q_y_logit, q_y, y,
                      is_train=True, label_idx=None,
                      pitch_input=None, pitch_mu=None, pitch_logvar=None, pitch_logit=None):
        logpx_z = -1 * self.loss['loss_recon'](epoch, x_predict, x)
        kld_y, h_y = self.loss['loss_class'](epoch, log_q_y_logit, q_y, self.model.n_class)
        neg_kld_y = -1 * kld_y
        neg_kld_z = -1 * self.loss['loss_latent'](epoch, q_mu, q_logvar, q_y,
                                                  self.model.mu_lookup, self.model.logvar_lookup)
        # the instrument label loss will be zero if unsupervised
        label_loss = self.loss['loss_class_label'](epoch, log_q_y_logit, y, is_train=is_train, label_idx=label_idx)

        logpx_z = torch.mean(logpx_z, dim=0)
        neg_kld_y = torch.mean(neg_kld_y, dim=0)
        neg_kld_z = torch.mean(neg_kld_z, dim=0)
        h_y = torch.mean(h_y, dim=0)
        label_loss = torch.mean(label_loss, dim=0)
        lower_bound = (logpx_z + neg_kld_y + neg_kld_z)

        if self.model.is_pitch_condition:
            assert pitch_input is not None
            assert pitch_mu is not None
            assert pitch_logvar is not None
            pitch_kl_loss = self.loss['loss_pitch_emb'](epoch, self.model.pitch_mu_lookup, self.model.pitch_logvar_lookup,
                                                        pitch_mu, pitch_logvar, pitch_input)
            pitch_kl_loss = torch.mean(pitch_kl_loss, dim=0)
        else:
            pitch_kl_loss = torch.zeros(1)
            pitch_kl_loss.require_grad = False

        if self.model.is_pitch_discriminate:
            pitch_classify_loss = self.loss['loss_pitch_discriminate'](epoch, pitch_logit, pitch_input)
            pitch_classify_loss = torch.mean(pitch_classify_loss, dim=0)
        else:
            pitch_classify_loss = torch.zeros(1)
            pitch_classify_loss.require_grad = False


        total_loss = -1 * lower_bound + label_loss + pitch_kl_loss + pitch_classify_loss


        return total_loss, lower_bound, logpx_z, neg_kld_y, neg_kld_z, h_y, label_loss,\
            pitch_kl_loss, pitch_classify_loss

    def _data_pipe(self, data, target, data_idx):
        if self.label_idx is not None:
            batch_label_idx = np.nonzero(np.in1d(data_idx, self.label_idx))
        else:
            batch_label_idx = None
        y_ins, y_pitch_class, y_pitch, y_dyn = target[0], target[1], target[2], target[3]
        pitch_input = y_pitch.to(self.device)
        dyn_input = y_dyn.to(self.device)
        target = y_ins.to(self.device)
        data = data.to(self.device)
        n_band, context_size = data.size(2), data.size(3)
        data = data.view(-1, n_band, 1, context_size).squeeze(2)
        return pitch_input, dyn_input, data, target, batch_label_idx

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
        return acc_metrics

    def _train_epoch(self, epoch):
        self.model.train()

        total_loss = 0
        total_lowerbound = 0
        total_logpx_z = 0
        total_neg_kld_y = 0
        total_neg_kld_z = 0
        total_h_y = 0
        total_label_loss = 0
        total_pitch_kl_loss = 0
        total_pitch_classify_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, (data, target, idx) in enumerate(self.data_loader):
            pitch_input, dyn_input, data, target, batch_label_idx = self._data_pipe(data, target, idx)

            x_predict, mu, logvar, z, log_q_y_logit, q_y, ind, pitch_mu, pitch_logvar, pitch_z, pitch_logit =\
                self.model(data)

            loss, lower_bound, logpx_z, neg_kld_y, neg_kld_z, h_y, label_loss, pitch_kl_loss, pitch_classify_loss = \
                self._compute_loss(epoch, data, x_predict, mu, logvar, log_q_y_logit, q_y, target,
                                   pitch_input=pitch_input, pitch_mu=pitch_mu,
                                   pitch_logvar=pitch_logvar, pitch_logit=pitch_logit,
                                   is_train=True, label_idx=batch_label_idx)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_lowerbound += lower_bound.item()
            total_logpx_z += logpx_z.item()
            total_neg_kld_y += neg_kld_y.item()
            total_neg_kld_z += neg_kld_z.item()
            total_h_y += h_y.item()
            total_label_loss += label_loss.item()
            total_pitch_kl_loss += pitch_kl_loss.item()
            total_pitch_classify_loss += pitch_classify_loss.item()
            total_metrics += self._eval_metrics(log_q_y_logit, target)

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item()))

        total_loss /= len(self.data_loader)
        total_lowerbound /= len(self.data_loader)
        total_logpx_z /= len(self.data_loader)
        total_neg_kld_y /= len(self.data_loader)
        total_neg_kld_z /= len(self.data_loader)
        total_h_y /= len(self.data_loader)
        total_label_loss /= len(self.data_loader)
        total_pitch_kl_loss /= len(self.data_loader)
        total_pitch_classify_loss /= len(self.data_loader)
        total_metrics = (total_metrics / len(self.data_loader)).tolist()

        self.writer.set_step(epoch, 'train')
        self.writer.add_scalar('total_loss', total_loss)
        self.writer.add_scalar('ELBO', total_lowerbound)
        self.writer.add_scalar('logpx_z', total_logpx_z)
        self.writer.add_scalar('neg_kld_z', total_neg_kld_z)
        self.writer.add_scalar('h_y', total_h_y)
        self.writer.add_scalar('label_CE', total_label_loss)
        self.writer.add_scalar('pitch_emb', total_pitch_kl_loss)
        self.writer.add_scalar('pitch_classify', total_pitch_classify_loss)

        for k, m in enumerate(total_metrics):
            self.writer.add_scalar('metric_%d' % k, m)

        log = {
            'loss': total_loss,
            'lower_bound': total_lowerbound,
            'logpx_z': total_logpx_z,
            'neg_kld_y': total_neg_kld_y,
            'neg_kld_z': total_neg_kld_z,
            'h_y': total_h_y,
            'label_loss': total_label_loss,
            'pitch_emb': total_pitch_kl_loss,
            'pitch_classify': total_pitch_classify_loss,
            'metrics': total_metrics
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_lowerbound = 0
        total_val_logpx_z = 0
        total_val_neg_kld_y = 0
        total_val_neg_kld_z = 0
        total_val_h_y = 0
        total_val_label_loss = 0
        total_val_pitch_kl_loss = 0
        total_val_pitch_classify_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, (data, target, idx) in enumerate(self.valid_data_loader):
                pitch_input, dyn_input, data, target, batch_label_idx = self._data_pipe(data, target, idx)

                x_predict, mu, logvar, z, log_q_y_logit, q_y, ind, pitch_mu, pitch_logvar, pitch_z, pitch_logit =\
                    self.model(data)

                loss, lower_bound, logpx_z, neg_kld_y, neg_kld_z, h_y, label_loss, pitch_kl_loss, pitch_classify_loss = \
                    self._compute_loss(epoch, data, x_predict, mu, logvar, log_q_y_logit, q_y, target,
                                       pitch_input=pitch_input, pitch_mu=pitch_mu,
                                       pitch_logvar=pitch_logvar, pitch_logit=pitch_logit,
                                       is_train=False, label_idx=batch_label_idx)

                total_val_loss += loss.item()
                total_val_lowerbound += lower_bound.item()
                total_val_logpx_z += logpx_z.item()
                total_val_neg_kld_y += neg_kld_y.item()
                total_val_neg_kld_z += neg_kld_z.item()
                total_val_h_y += h_y.item()
                total_val_label_loss += label_loss.item()
                total_val_pitch_kl_loss += pitch_kl_loss.item()
                total_val_pitch_classify_loss += pitch_classify_loss.item()
                total_val_metrics += self._eval_metrics(log_q_y_logit, target)

            total_val_loss /= len(self.valid_data_loader)
            total_val_lowerbound /= len(self.valid_data_loader)
            total_val_logpx_z /= len(self.valid_data_loader)
            total_val_neg_kld_y /= len(self.valid_data_loader)
            total_val_neg_kld_z /= len(self.valid_data_loader)
            total_val_h_y /= len(self.valid_data_loader)
            total_val_label_loss /= len(self.valid_data_loader)
            total_val_pitch_kl_loss /= len(self.valid_data_loader)
            total_val_pitch_classify_loss /= len(self.valid_data_loader)
            total_val_metrics = (total_val_metrics / len(self.valid_data_loader)).tolist()

            self.writer.set_step(epoch, 'valid')
            self.writer.add_scalar('total_loss', total_val_loss)
            self.writer.add_scalar('ELBO', total_val_lowerbound)
            self.writer.add_scalar('logpx_z', total_val_logpx_z)
            self.writer.add_scalar('neg_kld_z', total_val_neg_kld_z)
            self.writer.add_scalar('h_y', total_val_h_y)
            self.writer.add_scalar('label_CE', total_val_label_loss)
            self.writer.add_scalar('pitch_emb', total_val_pitch_kl_loss)
            self.writer.add_scalar('pitch_classify', total_val_pitch_classify_loss)
            for k, m in enumerate(total_val_metrics):
                self.writer.add_scalar('metric_%d' % k, m)

        return {
            'val_loss': total_val_loss,
            'val_lower_bound': total_val_lowerbound,
            'val_logpx_z': total_val_logpx_z,
            'val_neg_kld_y': total_val_neg_kld_y,
            'val_neg_kld_z': total_val_neg_kld_z,
            'val_h_y': total_val_h_y,
            'val_label_loss': total_val_label_loss,
            'val_pitch_emb': total_val_pitch_kl_loss,
            'val_pitch_classify': total_val_pitch_classify_loss,
            'val_metrics': total_val_metrics,
        }
