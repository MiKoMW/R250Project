from __future__ import unicode_literals, print_function, division

import os
import time
import argparse
import shutil

import torch
import numpy as np
from model import Model
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, Adagrad
import tensorflow as tf

import config
from batcher import Batcher
from data import Vocab
from utils import calc_running_avg_loss
from train_util import get_input_from_batch, get_output_from_batch, compute_reward, gen_preds
from eval import Evaluate

use_cuda = config.use_gpu and torch.cuda.is_available()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


# TODO: ADDED for dev
tf.compat.v1.disable_eager_execution()

class Train(object):
    def __init__(self):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.train_data_path, self.vocab, mode='train',
                               batch_size=config.batch_size, single_pass=False)
        time.sleep(5)

        # print("BATCH")
        # print(self.batcher)

        if not os.path.exists(config.log_root):
            os.mkdir(config.log_root)

        self.model_dir = os.path.join(config.log_root, 'train_model')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.eval_log = os.path.join(config.log_root, 'eval_log')
        if not os.path.exists(self.eval_log):
            os.mkdir(self.eval_log)
        self.summary_writer = tf.compat.v1.summary.FileWriter(self.eval_log)


    def save_model(self, running_avg_loss, iter, mode):
        state = {
            'iter': iter,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'reduce_state_dict': self.model.reduce_state.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_loss
        }
        if mode == 'train':
            save_model_dir = self.model_dir
        else:
            best_model_dir = os.path.join(config.log_root, 'best_model')
            if not os.path.exists(best_model_dir):
                os.mkdir(best_model_dir)
            save_model_dir = best_model_dir

        if len(os.listdir(save_model_dir))>0:
            if mode == 'train':
                time.sleep(2)
            else:
                shutil.rmtree(save_model_dir)
                time.sleep(2)
                os.mkdir(save_model_dir)
        train_model_path = os.path.join(save_model_dir, 'model_best_%d'%(iter))
        torch.save(state, train_model_path)
        return train_model_path


    def setup_train(self, model_file_path=None):
        self.model = Model(model_file_path)

        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()) + \
                 list(self.model.reduce_state.parameters())
        initial_lr = config.lr_coverage if config.is_coverage else config.lr
        if config.mode == 'MLE':
            self.optimizer = Adagrad(params, lr=0.15, initial_accumulator_value=0.1)
        else:
            self.optimizer = Adam(params, lr=initial_lr)

        start_iter, start_loss = 0, 0

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']
        return start_iter, start_loss


    def train_one_batch(self, batch, alpha, beta):

        #
        # print("BATCH")
        # print(batch)


        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch, use_cuda)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, use_cuda)

        #
        # print("ENC_BATCH")
        # print(len(enc_batch))
        # print(len(enc_batch[0]))
        # print((enc_batch[0]))
        #
        # print("enc_padding_mask")
        # print(enc_padding_mask)
        # print(len(enc_padding_mask))
        # print(len(enc_padding_mask[0]))

        # print("enc_lens")
        # print(enc_lens)
        # print("enc_batch_extend_vocab")
        # print(enc_batch_extend_vocab)

        self.optimizer.zero_grad()

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)

        # print("encoder_outputs")
        # print(encoder_outputs.siz)


        s_t_1 = self.model.reduce_state(encoder_hidden)

        nll_list= []

        # sample_size

        gen_summary = torch.LongTensor(config.batch_size*[config.sample_size*[[2]]]) # B x S x 1

        # print("gen_summary")
        # print(gen_summary.size())
        # print(gen_summary)


        if use_cuda: gen_summary = gen_summary.cuda()
        preds_y = gen_summary.squeeze(2) # B x S


        # TODO: Print Gold Here!!!!
        # print("preds_y")
        # print(preds_y.size())
        # print(preds_y)
        # print(self.vocab.size())
        # print("temp")
        # from data import outputids2words
        # temp = outputids2words(list(map(lambda x : x.item(), enc_batch[1])),self.vocab,None)
        # print(" ".join(temp))
        # # for item in dec_batch[1]:
        # #     temp = self.vocab.id2word(item.item())
        # #     from data import outputids2words(dec_batch[1])
        # #     print(temp)

        from data import outputids2words

        # print("dec_batch")
        # print(dec_batch[0])
        # temp = outputids2words(list(map(lambda x : x.item(), dec_batch[0])),self.vocab,None)
        # print(temp)
        # print()
        # print("target_batch")
        # print(target_batch[0])
        # temp = outputids2words(list(map(lambda x : x.item(), target_batch[0])),self.vocab,None)
        # print(temp)
        # print()


        for di in range(min(config.max_dec_steps, dec_batch.size(1))):
            # Select the current input word
            p1 = np.random.uniform()
            if p1 < alpha: # use ground truth word
                y_t_1 = dec_batch[:, di]
            else: # use decoded word
                y_t_1 = preds_y[:, 0]

            # print("y_t_1")
            # # print(y_t_1)
            # print("dec_batch")
            # print(dec_batch)
            final_dist, s_t_1,  c_t_1, attn_dist, p_gen, next_coverage = self.model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask,
                                                        c_t_1, extra_zeros, enc_batch_extend_vocab,
                                                        coverage, di)

            # Select the current output word
            p2 = np.random.uniform()
            if p2 < beta: # sample the ground truth word
                target = target_batch[:, di]
                sampled_batch = torch.stack(config.sample_size*[target], 1) # B x S
            else: # randomly sample a word with given probabilities
                sampled_batch = torch.multinomial(final_dist, config.sample_size, replacement=True) # B x S

            # Compute the NLL
            probs = torch.gather(final_dist, 1, sampled_batch).squeeze()
            step_nll = -torch.log(probs + config.eps)

            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_nll = step_nll + config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage
            nll_list.append(step_nll)

            # Store the decoded words in preds_y
            preds_y = gen_preds(sampled_batch, use_cuda)
            # Add the decoded words into gen_summary (mixed with ground truth and decoded words)
            gen_summary = torch.cat((gen_summary, preds_y.unsqueeze(2)), 2) # B x S x L

        # compute the REINFORCE score        
        nll = torch.sum(torch.stack(nll_list, 2), 2)  # B x S
        all_rewards, avg_reward = compute_reward(batch, gen_summary, self.vocab, config.mode, use_cuda) # B x S, 1
        batch_loss = torch.sum(nll * all_rewards, dim=1)  # B
        loss = torch.mean(batch_loss)

        loss.backward()

        self.norm = clip_grad_norm_(self.model.encoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.decoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.reduce_state.parameters(), config.max_grad_norm)

        self.optimizer.step()
        return loss.item(), avg_reward.item()



    def train_one_batch_mixer(self, batch, alpha_in, beta_in, mixer_fisrt_N_steps_use_XENT):

        #
        # print("BATCH")
        # print(batch)


        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch, use_cuda)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, use_cuda)


        # start from here.
        self.optimizer.zero_grad()

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)

        # print("encoder_outputs")
        # print(encoder_outputs.siz)


        s_t_1 = self.model.reduce_state(encoder_hidden)

        nll_list= []

        # sample_size 是啥？

        gen_summary = torch.LongTensor(config.batch_size*[config.sample_size*[[2]]]) # B x S x 1

        # print("gen_summary")
        # print(gen_summary.size())
        # print(gen_summary)


        if use_cuda: gen_summary = gen_summary.cuda()
        preds_y = gen_summary.squeeze(2) # B x S


        # TODO: Print Gold Here!!!!
        # print("preds_y")
        # print(preds_y.size())
        # print(preds_y)
        # print(self.vocab.size())
        # print("temp")
        # from data import outputids2words
        # temp = outputids2words(list(map(lambda x : x.item(), enc_batch[1])),self.vocab,None)
        # print(" ".join(temp))
        # # for item in dec_batch[1]:
        # #     temp = self.vocab.id2word(item.item())
        # #     from data import outputids2words(dec_batch[1])
        # #     print(temp)

        from data import outputids2words

        # print("dec_batch")
        # print(dec_batch[0])
        # temp = outputids2words(list(map(lambda x : x.item(), dec_batch[0])),self.vocab,None)
        # print(temp)
        # print()
        # print("target_batch")
        # print(target_batch[0])
        # temp = outputids2words(list(map(lambda x : x.item(), target_batch[0])),self.vocab,None)
        # print(temp)
        # print()

        # print("START")
        for di in range(min(config.max_dec_steps, dec_batch.size(1))):
            # Select the current input word
            p1 = np.random.uniform()


            # Mixer schedule goes here
            if di < mixer_fisrt_N_steps_use_XENT:
                alpha = 1.
                beta = 1.
            else:
                alpha = alpha_in
                beta = beta_in

            # print("alphdabeta")
            # print(alpha)
            # print(beta)
            if p1 < alpha: # use ground truth word
                y_t_1 = dec_batch[:, di]
            else: # use decoded word
                y_t_1 = preds_y[:, 0]



            # print("y_t_1")
            # # print(y_t_1)
            # print("dec_batch")
            # print(dec_batch)
            final_dist, s_t_1,  c_t_1, attn_dist, p_gen, next_coverage = self.model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask,
                                                        c_t_1, extra_zeros, enc_batch_extend_vocab,
                                                        coverage, di)

            # Select the current output word
            p2 = np.random.uniform()
            if p2 < beta: # sample the ground truth word
                target = target_batch[:, di]
                sampled_batch = torch.stack(config.sample_size*[target], 1) # B x S
            else: # randomly sample a word with given probabilities
                sampled_batch = torch.multinomial(final_dist, config.sample_size, replacement=True) # B x S

            # Compute the NLL
            probs = torch.gather(final_dist, 1, sampled_batch).squeeze()
            step_nll = -torch.log(probs + config.eps)

            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_nll = step_nll + config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage
            nll_list.append(step_nll)

            # Store the decoded words in preds_y
            preds_y = gen_preds(sampled_batch, use_cuda)
            # Add the decoded words into gen_summary (mixed with ground truth and decoded words)
            gen_summary = torch.cat((gen_summary, preds_y.unsqueeze(2)), 2) # B x S x L

        # compute the REINFORCE score
        nll = torch.sum(torch.stack(nll_list, 2), 2)  # B x S
        all_rewards, avg_reward = compute_reward(batch, gen_summary, self.vocab, config.mode, use_cuda) # B x S, 1
        batch_loss = torch.sum(nll * all_rewards, dim=1)  # B
        loss = torch.mean(batch_loss)
        if config.increasing_rl:
            temp_loss = loss * (40 - mixer_fisrt_N_steps_use_XENT)/40
            # print(mixer_fisrt_N_steps_use_XENT)
        else:
            temp_loss = loss
        temp_loss.backward()

        # loss.backward()

        self.norm = clip_grad_norm_(self.model.encoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.decoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.reduce_state.parameters(), config.max_grad_norm)

        self.optimizer.step()
        return loss.item(), avg_reward.item()


    def train_one_batch_mixer_first(self, batch, alpha_input, beta_input, mixer_fisrt_N_steps_use_XENT):

        #
        # print("BATCH")
        # print(batch)


        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch, use_cuda)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, use_cuda)


        # start from here.
        self.optimizer.zero_grad()

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)

        # print("encoder_outputs")
        # print(encoder_outputs.siz)


        s_t_1 = self.model.reduce_state(encoder_hidden)

        nll_list= []

        # sample_size 是啥？

        gen_summary = torch.LongTensor(config.batch_size*[config.sample_size*[[2]]]) # B x S x 1

        # print("gen_summary")
        # print(gen_summary.size())
        # print(gen_summary)


        if use_cuda: gen_summary = gen_summary.cuda()
        preds_y = gen_summary.squeeze(2) # B x S


        # TODO: Print Gold Here!!!!
        # print("preds_y")
        # print(preds_y.size())
        # print(preds_y)
        # print(self.vocab.size())
        # print("temp")
        # from data import outputids2words
        # temp = outputids2words(list(map(lambda x : x.item(), enc_batch[1])),self.vocab,None)
        # print(" ".join(temp))
        # # for item in dec_batch[1]:
        # #     temp = self.vocab.id2word(item.item())
        # #     from data import outputids2words(dec_batch[1])
        # #     print(temp)

        from data import outputids2words

        # print("dec_batch")
        # print(dec_batch[0])
        # temp = outputids2words(list(map(lambda x : x.item(), dec_batch[0])),self.vocab,None)
        # print(temp)
        # print()
        # print("target_batch")
        # print(target_batch[0])
        # temp = outputids2words(list(map(lambda x : x.item(), target_batch[0])),self.vocab,None)
        # print(temp)
        # print()

        do_XENT = False
        consider_Done = True
        loss_1 = 0
        # print("START")
        for di in range(min(mixer_fisrt_N_steps_use_XENT ,min(config.max_dec_steps, dec_batch.size(1)))):
            do_XENT = True
            consider_Done = False
            #min(config.max_dec_steps, dec_batch.size(1))

            p1 = np.random.uniform()
            alpha = 1.
            beta = 1.

            if p1 < alpha: # use ground truth word
                y_t_1 = dec_batch[:, di]
            else: # use decoded word
                y_t_1 = preds_y[:, 0]

            final_dist, s_t_1,  c_t_1, attn_dist, p_gen, next_coverage = self.model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask,
                                                        c_t_1, extra_zeros, enc_batch_extend_vocab,
                                                        coverage, di)

            p2 = np.random.uniform()
            if p2 < beta: # sample the ground truth word
                target = target_batch[:, di]
                sampled_batch = torch.stack(config.sample_size*[target], 1) # B x S
            else: # randomly sample a word with given probabilities
                sampled_batch = torch.multinomial(final_dist, config.sample_size, replacement=True) # B x S

            probs = torch.gather(final_dist, 1, sampled_batch).squeeze()
            step_nll = -torch.log(probs + config.eps)

            nll_list.append(step_nll)

            # Store the decoded words in preds_y
            preds_y = gen_preds(sampled_batch, use_cuda)
            # Add the decoded words into gen_summary (mixed with ground truth and decoded words)
            gen_summary = torch.cat((gen_summary, preds_y.unsqueeze(2)), 2) # B x S x L
        if do_XENT:
            # compute the REINFORCE score
            nll = torch.sum(torch.stack(nll_list, 2), 2)  # B x S
            all_rewards, avg_reward = compute_reward(batch, gen_summary, self.vocab, "MLE", use_cuda)  # B x S, 1
            batch_loss = torch.sum(nll * all_rewards, dim=1)  # B
            loss_1 = torch.mean(batch_loss)

        loss_2 = 0

        do_mixer = False
        nll_list= []
        # print("START")
        for di in range(mixer_fisrt_N_steps_use_XENT, min(config.max_dec_steps, dec_batch.size(1))):

            do_mixer = True

            p1 = np.random.uniform()
            alpha = alpha_input
            beta = beta_input

            if p1 < alpha:  # use ground truth word
                y_t_1 = dec_batch[:, di]
            else:  # use decoded word
                y_t_1 = preds_y[:, 0]

            final_dist, s_t_1, c_t_1, attn_dist, p_gen, next_coverage = self.model.decoder(y_t_1, s_t_1,
                                                                                           encoder_outputs,
                                                                                           encoder_feature,
                                                                                           enc_padding_mask,
                                                                                           c_t_1, extra_zeros,
                                                                                           enc_batch_extend_vocab,
                                                                                           coverage, di)
            p2 = np.random.uniform()
            if p2 < beta:  # sample the ground truth word
                target = target_batch[:, di]
                sampled_batch = torch.stack(config.sample_size * [target], 1)  # B x S
            else:  # randomly sample a word with given probabilities
                sampled_batch = torch.multinomial(final_dist, config.sample_size, replacement=True)  # B x S

            probs = torch.gather(final_dist, 1, sampled_batch).squeeze()
            step_nll = -torch.log(probs + config.eps)

            nll_list.append(step_nll)

            # Store the decoded words in preds_y
            preds_y = gen_preds(sampled_batch, use_cuda)
            # Add the decoded words into gen_summary (mixed with ground truth and decoded words)
            gen_summary = torch.cat((gen_summary, preds_y.unsqueeze(2)), 2)  # B x S x L
        if(do_mixer):
            nll = torch.sum(torch.stack(nll_list, 2), 2)  # B x S
            all_rewards, avg_reward = compute_reward(batch, gen_summary, self.vocab, config.mode, use_cuda)  # B x S, 1
            batch_loss = torch.sum(nll * all_rewards, dim=1)  # B
            loss_2 = torch.mean(batch_loss)
        loss = (loss_1 + loss_2)
        if config.increasing_rl:
            temp_loss = loss * (40 - mixer_fisrt_N_steps_use_XENT)/40
            # print(mixer_fisrt_N_steps_use_XENT)
        else:
            temp_loss = loss
        temp_loss.backward()

        # loss.backward()
        # loss.backward()
        self.norm = clip_grad_norm_(self.model.encoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.decoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.reduce_state.parameters(), config.max_grad_norm)

        self.optimizer.step()



        return loss.item(), avg_reward.item()


    # def trainIters_mixer(self, n_iters, model_file_path=None):
    #     iter, running_avg_loss = self.setup_train(model_file_path)
    #     min_val_loss = np.inf
    #
    #     alpha = config.alpha
    #     beta = config.beta
    #     k1 = config.k1
    #     k2 = config.k2
    #     delay = 0
    #     earlyStopping_counter = 0
    #
    #     isMixer = config.isMixer
    #     mixer_T = config.mixer_T
    #     mixer_delta = config.mixer_delta
    #     mixer_N_XENT_step = config.mixer_N_XENT_step
    #     mixer_N_XENTRL_step = config.mixer_N_XENTRL_step
    #
    #     while iter < n_iters:
    #         if config.mode == 'RL':
    #             alpha = 0.
    #             beta = 0.
    #         elif config.mode == 'GTI':
    #             alpha = 1.
    #             beta = 0.
    #         elif config.mode == 'SO':
    #             alpha = 1.
    #             beta = k2/(k2+np.exp((iter-delay)/k2))
    #         elif config.mode == 'SIO':
    #             alpha *= k1
    #             if alpha < 0.01:
    #                 beta = k2/(k2+np.exp((iter-delay)/k2))
    #             else:
    #                 beta = 1.
    #                 delay += 1
    #         elif config.mode == 'DAGGER':
    #             alpha *= k1
    #             beta = 1.
    #         elif config.mode == 'DAGGER*':
    #             alpha = config.alpha
    #             beta = 1.
    #         # elif config.mode == "MIXER":
    #         #     alpha = 1.
    #         #     beta = 1.
    #         #     isMixer = True
    #         else:
    #             alpha = 1.
    #             beta = 1.
    #
    #         batch = self.batcher.next_batch()
    #
    #         if(isMixer):
    #             # More mixer code
    #             if iter <= mixer_N_XENT_step:
    #                 mixer_fisrt_N_steps_use_XENT = mixer_T
    #             else:
    #                 temp_over_step = iter - mixer_N_XENT_step
    #                 temp_done_XENTR_turn = int(temp_over_step / mixer_N_XENTRL_step)
    #                 mixer_fisrt_N_steps_use_XENT = mixer_T - (temp_done_XENTR_turn * mixer_delta) - mixer_delta
    #                 if (mixer_fisrt_N_steps_use_XENT < 0):
    #                     mixer_fisrt_N_steps_use_XENT = 0
    #             loss, avg_reward = self.train_one_batch_mixer_mle(batch, alpha, beta, mixer_fisrt_N_steps_use_XENT)
    #
    #         else:
    #             loss, avg_reward = self.train_one_batch(batch, alpha, beta)
    #
    #         running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
    #         iter += 1
    #
    #         if iter % config.print_interval == 0:
    #             print('steps %d, current_loss: %f, avg_reward: %f' % (iter, loss, avg_reward))
    #
    #         if iter % config.save_model_iter == 0:
    #             model_file_path = self.save_model(running_avg_loss, iter, mode='train')
    #             evl_model = Evaluate(model_file_path)
    #             val_avg_loss = evl_model.run_eval()
    #             if val_avg_loss < min_val_loss:
    #                 min_val_loss = val_avg_loss
    #                 best_model_file_path = self.save_model(running_avg_loss, iter, mode='eval')
    #                 print('Save best model at %s' % best_model_file_path)
    #                 earlyStopping_counter = 0
    #             else:
    #                 if(iter >= config.min_earlyStopping):
    #                     earlyStopping_counter+=1
    #             print('steps %d, train_loss: %f, val_loss: %f' % (iter, loss, val_avg_loss))
    #             # write val_loss into tensorboard
    #             loss_sum = tf.compat.v1.Summary()
    #             loss_sum.value.add(tag='val_avg_loss', simple_value=val_avg_loss)
    #             self.summary_writer.add_summary(loss_sum, global_step=iter)
    #             self.summary_writer.flush()
    #
    #             if(earlyStopping_counter > 2):
    #                 print("EARLY STOP at " + (str(iter)))
    #                 break




    def trainIters(self, n_iters, model_file_path=None):
        iter, running_avg_loss = self.setup_train(model_file_path)
        min_val_loss = np.inf

        alpha = config.alpha
        beta = config.beta
        k1 = config.k1
        k2 = config.k2
        delay = 0
        earlyStopping_counter = 0

        # TODO: Setup the mixer curriculum learning scheduling
        isMixer = config.isMixer
        mixer_T = config.mixer_T
        mixer_delta = config.mixer_delta
        mixer_N_XENT_step = config.mixer_N_XENT_step
        mixer_N_XENTRL_step = config.mixer_N_XENTRL_step

        while iter < n_iters:
            if config.mode == 'RL':
                alpha = 0.
                beta = 0.
            elif config.mode == 'GTI':
                alpha = 1.
                beta = 0.
            elif config.mode == 'SO':
                alpha = 1.
                beta = k2/(k2+np.exp((iter-delay)/k2))
            elif config.mode == 'SIO':
                alpha *= k1
                if alpha < 0.01:
                    beta = k2/(k2+np.exp((iter-delay)/k2))
                else:
                    beta = 1.
                    delay += 1
            elif config.mode == 'DAGGER':
                alpha *= +k1
                beta = 1.
            elif config.mode == 'DAGGER*':
                alpha = config.alpha
                beta = 1.
            # elif config.mode == "MIXER":
            #     alpha = 1.
            #     beta = 1.
            #     isMixer = True
            else:
                alpha = 1.
                beta = 1.

            batch = self.batcher.next_batch()

            if(isMixer):
                # More mixer code
                if iter <= mixer_N_XENT_step:
                    mixer_fisrt_N_steps_use_XENT = mixer_T
                else:
                    temp_over_step = iter - mixer_N_XENT_step
                    temp_done_XENTR_turn = int(temp_over_step / mixer_N_XENTRL_step)
                    mixer_fisrt_N_steps_use_XENT = mixer_T - (temp_done_XENTR_turn * mixer_delta) - mixer_delta
                    if (mixer_fisrt_N_steps_use_XENT < 0):
                        mixer_fisrt_N_steps_use_XENT = 0
                loss, avg_reward = self.train_one_batch_mixer(batch, alpha, beta, mixer_fisrt_N_steps_use_XENT)
                # loss, avg_reward = self.train_one_batch_mixer_first(batch, alpha, beta, mixer_fisrt_N_steps_use_XENT)

            else:
                loss, avg_reward = self.train_one_batch(batch, alpha, beta)

            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
            iter += 1

            if iter % config.print_interval == 0:
                print('steps %d, current_loss: %f, avg_reward: %f' % (iter, loss, avg_reward))

            if iter % config.save_model_iter == 0:
                model_file_path = self.save_model(running_avg_loss, iter, mode='train')
                evl_model = Evaluate(model_file_path)
                val_avg_loss = evl_model.run_eval()
                if val_avg_loss < min_val_loss:
                    min_val_loss = val_avg_loss
                    best_model_file_path = self.save_model(running_avg_loss, iter, mode='eval')
                    print('Save best model at %s' % best_model_file_path)
                    earlyStopping_counter = 0
                else:
                    if(iter >= config.min_earlyStopping):
                        earlyStopping_counter+=1
                print('steps %d, train_loss: %f, val_loss: %f' % (iter, loss, val_avg_loss))
                # write val_loss into tensorboard
                loss_sum = tf.compat.v1.Summary()
                loss_sum.value.add(tag='val_avg_loss', simple_value=val_avg_loss)
                self.summary_writer.add_summary(loss_sum, global_step=iter)
                self.summary_writer.flush()

                if(earlyStopping_counter > 1):
                    print("EARLY STOP at " + (str(iter)))
                    break






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument("-m",
                        dest="model_file_path",
                        required=False,
                        default=None,
                        help="Model file for retraining (default: None).")
    args = parser.parse_args()

    train_processor = Train()
    train_processor.trainIters(config.max_iterations, args.model_file_path)
