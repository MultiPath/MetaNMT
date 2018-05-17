# encoding: utf-8
import torch
import numpy as np
from torchtext import data
from torchtext import datasets

import revtok
import logging
import random
import argparse
import os
import copy
import sys
import math
import time

from ez_train import export, valid_model
from decode import decode_model
from model import Transformer, UniversalTransformer, INF, TINY, softmax
from utils import NormalField, NormalTranslationDataset, TripleTranslationDataset, ParallelDataset, LazyParallelDataset, merge_cache
from utils import Metrics, Best, computeGLEU, computeBLEU, NormalBucketIterator
from time import gmtime, strftime
from tqdm import tqdm, trange
from torch.autograd import Variable


# all the hyper-parameters
parser = argparse.ArgumentParser(description='Train a Transformer-Like Model.')

# dataset settings
parser.add_argument('--data_prefix', type=str, default='/data0/data/transformer_data/')
parser.add_argument('--vocab_prefix', type=str, default='/data0/data/transformer_data/')
parser.add_argument('--workspace_prefix', type=str, default='./')
parser.add_argument('--dataset',   type=str, default='iwslt', help='"the name of dataset"')
parser.add_argument('-s', '--src', type=str, default='ro',  help='meta-testing target language.')
parser.add_argument('-t', '--trg', type=str, default='en',  help='meta-testing target language.')
parser.add_argument('-a', '--aux', nargs='+', type=str,  default='es it pt fr',  help='meta-testing target language.')

parser.add_argument('--load_vocab',   action='store_true', help='load a pre-computed vocabulary')
parser.add_argument('--use_revtok',   action='store_true', help='use reversible tokenization')
parser.add_argument('--remove_eos',   action='store_true', help='possibly remove <eos> tokens for FastTransformer')
parser.add_argument('--test_set',     type=str, default=None,  help='which test set to use')
parser.add_argument('--max_len',      type=int, default=None,  help='limit the train set sentences to this many tokens')

# model basic settings
parser.add_argument('--prefix', type=str, default='[time]',      help='prefix to denote the model, nothing or [time]')
parser.add_argument('--params', type=str, default='james-iwslt', help='pamarater sets: james-iwslt, t2t-base, etc')

# model ablation settings
parser.add_argument('--causal_enc', action='store_true', help='use unidirectional encoder (useful for real-time translation)')
parser.add_argument('--causal',   action='store_true', help='use causal attention')
parser.add_argument('--diag',     action='store_true', help='ignore diagonal attention when doing self-attention.')
parser.add_argument('--use_wo',   action='store_true', help='use output weight matrix in multihead attention')
parser.add_argument('--share_embeddings',     action='store_true', help='share embeddings between encoder and decoder')
parser.add_argument('--positional_attention', action='store_true', help='incorporate positional information in key/value')

# running setting
parser.add_argument('--mode',    type=str, default='train',  help='train, test or build')
parser.add_argument('--gpu',     type=int, default=0,        help='GPU to use or -1 for CPU')
parser.add_argument('--seed',    type=int, default=19920206, help='seed for randomness')

# universal neural machine translation
parser.add_argument('--max_token', action='store_true', help='enable only using the nearest universal tokens')
parser.add_argument('--universal', action='store_true', help='enable embedding sharing in the universal space')
parser.add_argument('--inter_size', type=int, default=1, help='hack: inorder to increase the batch-size.')
parser.add_argument('--share_universal_embedding', action='store_true', help='share the embedding matrix with target. Currently only supports English.')
parser.add_argument('--finetune_dataset',  type=str, default=None)
parser.add_argument('--finetune', action='store_true', help='add an action as finetuning. used for RO dataset.')
parser.add_argument('--universal_options', default=[], nargs='*',
                    choices=['argmax', 'st', 'fixed_A', 'trainable_universal_tokens', 'refined_V'], help='list servers, storage, or both (default: %(default)s)')

parser.add_argument('--meta_learning', action='store_true', help='meta-learning for low resource neural machine translation')
parser.add_argument('--meta_approx_2nd', action='store_true', help='2nd order derivative approximation in meta-learning')
parser.add_argument('--approx_lr',type=float, default=0.1, help='step-size 2nd order derivative approximation in meta-learning')

# meta-learning
parser.add_argument('--cross_meta_learning',  action='store_true', help='randomly sample two languaeges: train on one, test on another.')
parser.add_argument('--no_meta_training',     action='store_true', help='no meta learning. directly training everything jointly.')
parser.add_argument('--sequential_learning',  action='store_true', help='default using a parallel training paradiam. However, it is another option to make training sequential.')
parser.add_argument('--valid_steps',   type=int, default=5,        help='repeating training for 5 epoches')
parser.add_argument('--inner_steps',   type=int, default=32,       help='every ** words for one meta-update (for default 160k)')
parser.add_argument('--valid_epochs',  type=int, default=4,        help='every ** words for one meta-update (for default 160k)')
parser.add_argument('--eval-every',    type=int, default=1024,     help='run dev every')
parser.add_argument('--eval-every-examples', type=int, default=-1, help='alternative to eval every (batches)')
parser.add_argument('--save_every',    type=int, default=10000,   help='save the best checkpoint every 50k updates')
parser.add_argument('--maximum_steps', type=int, default=2000000, help='maximum steps you take to train a model')
parser.add_argument('--batch_size',    type=int, default=2048,    help='# of tokens processed per batch')
parser.add_argument('--valid_batch_size', type=int, default=2048,    help='# of tokens processed per batch')
parser.add_argument('--optimizer',     type=str, default='Adam')
parser.add_argument('--disable_lr_schedule', action='store_true', help='disable the transformer-style learning rate')

parser.add_argument('--distillation', action='store_true', help='knowledge distillation at sequence level')
parser.add_argument('--finetuning',   action='store_true', help='knowledge distillation at word level')
parser.add_argument('--support_size',  type=int, default=16000, help='number of tokens as the support set.')
parser.add_argument('--finetune_params', type=str, default='fast', choices=['fast', 'emb_enc', 'emb'])

# decoding
parser.add_argument('--length_ratio',  type=int,   default=2, help='maximum lengths of decoding')
parser.add_argument('--decode_mode',   type=str,   default='argmax', help='decoding mode: argmax, mean, sample, noisy, search')
parser.add_argument('--beam_size',     type=int,   default=1, help='beam-size used in Beamsearch, default using greedy decoding')
parser.add_argument('--f_size',        type=int,   default=1, help='heap size for sampling/searching in the fertility space')
parser.add_argument('--alpha',         type=float, default=1, help='length normalization weights')
parser.add_argument('--temperature',   type=float, default=1, help='smoothing temperature for noisy decodig')
parser.add_argument('--rerank_by_bleu', action='store_true', help='use the teacher model for reranking')

# model saving/reloading, output translations
parser.add_argument('--load_from',     type=str, default=None, help='load from checkpoint')
parser.add_argument('--resume',        action='store_true', help='when loading from the saved model, it resumes from that.')
parser.add_argument('--share_encoder', action='store_true', help='use teacher-encoder to initialize student')

parser.add_argument('--no_bpe',        action='store_true', help='output files without BPE')
parser.add_argument('--no_write',      action='store_true', help='do not write the decoding into the decoding files.')
parser.add_argument('--output_fer',    action='store_true', help='decoding and output fertilities')

# debugging
parser.add_argument('--debug',       action='store_true', help='debug mode: no saving or tensorboard')
parser.add_argument('--tensorboard', action='store_true', help='use TensorBoard')


args = parser.parse_args()
if args.prefix == '[time]':
    args.prefix = strftime("%m.%d_%H.%M.", gmtime())

# check the path
if not os.path.exists(args.workspace_prefix):
    os.mkdir(args.workspace_prefix)

def build_path(args, name):
    prefix = args.workspace_prefix
    pathname = os.path.join(prefix, name)
    if not os.path.exists(pathname):
        os.mkdir(pathname)
    args.__dict__.update({name + "_dir": pathname})
    return pathname

build_path(args, "resume")
build_path(args, "models")
build_path(args, "runs")
build_path(args, "logs")


# setup logger settings
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

fh = logging.FileHandler('{}/log-{}.txt'.format(args.logs_dir, args.prefix))
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(fh)

# setup random seeds

def set_random(seed=None):
    if seed is None:
        seed = args.seed
    #print("seed {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_random()
seeds = random.sample(range(20000000), 500000)

# ----------------------------------------------------------------------------------------------------------------- #
data_prefix = args.data_prefix

# setup data-field
DataField = NormalField
TRG   = DataField(init_token='<init>', eos_token='<eos>', batch_first=True)
SRC   = DataField(batch_first=True) if not args.share_embeddings else TRG
SP    = 4
max_src_size = -1

# ----------------------------------------------------------------------------------------------------- #
vocab_src = []
vocab_trg = []
U = torch.zeros(SP, 300)
word_count, word_vector = torch.load(args.vocab_prefix + args.src + '.pt')
vocab_src += word_count
U = torch.cat([U, word_vector.float()], dim=0)

# build vocab online
SRC.build_vocab_from_vocab(vocab_src)
target_word_count, target_word_vector = torch.load(args.vocab_prefix + args.trg + '.pt')
vocab_trg += target_word_count
V = target_word_vector.float()[:20000, :]

# build vocab online
TRG.build_vocab_from_vocab(vocab_trg)

if args.gpu > -1:
    U = U.cuda(args.gpu)
    V = V.cuda(args.gpu)

args.__dict__.update({'U': U, 'V': V, 'max_src_size': U.size(0), 'unitok_size': V.size(0)})

# setup many datasets (need to manaually setup --- Meta-Learning settings.
logger.info('start loading the dataset')

# evaluation does not need aux-languages
DEV_BLEU, TEST_BLEU = [], []

for sample in range(5):

    if "meta" in args.dataset:
        working_path = data_prefix + "{}/eval/{}-{}/".format(args.dataset, args.src, args.trg)

        dev_set = 'dev'
        test_set = 'test'
        train_set = 'train.{}.{}'.format(args.support_size, sample)

        train_data, dev_data, test_data = ParallelDataset.splits(path=working_path, train=train_set,
            validation=dev_set, test=test_set, exts=('.src', '.trg'), fields=[('src', SRC), ('trg', TRG)])
        decoding_path = working_path + '{}.' + args.src + '-' + args.trg + '.new'

    else:
        raise NotImplementedError

    logger.info('load dataset done.. src: {}, trg: {}, U: {}, V: {}'.format(len(SRC.vocab), len(TRG.vocab), U.size(0), V.size(0)))
    args.__dict__.update({'trg_vocab': len(TRG.vocab), 'src_vocab': len(SRC.vocab)})

    # build dynamic batching ---
    def dyn_batch_with_padding(new, i, sofar):
        prev_max_len = sofar / (i - 1) if i > 1 else 0
        if args.distillation:
            return max(len(new.src), len(new.trg), len(new.dec), prev_max_len) * i
        else:
            return max(len(new.src), len(new.trg),  prev_max_len) * i

    def dyn_batch_without_padding(new, i, sofar):
        if args.distillation:
            return sofar + max(len(new.src), len(new.trg), len(new.dec))
        else:
            return sofar + max(len(new.src), len(new.trg))

    if args.batch_size == 1:  # speed-test: one sentence per batch.
        batch_size_fn = lambda new, count, sofar: count
    else:
        batch_size_fn = dyn_batch_with_padding # dyn_batch_without_padding

    train_real, dev_real, test_real = data.BucketIterator.splits(
        (train_data, dev_data, test_data),
        batch_sizes=(args.batch_size, args.valid_batch_size, args.valid_batch_size),
        device=args.gpu, shuffle=True,
        batch_size_fn=batch_size_fn, repeat=None if args.mode == 'train' else False)
    logger.info("build the dataset. done!")

    # ----------------------------------------------------------------------------------------------------------------- #
    # model hyper-params:
    logger.info('use default parameters of t2t-base')
    hparams = {'d_model': 512, 'd_hidden': 512, 'n_layers': 6,
                'n_heads': 8, 'drop_ratio': 0.1, 'warmup': 16000} # ~32
    args.__dict__.update(hparams)

    # ----------------------------------------------------------------------------------------------------------------- #
    # show the arg:

    hp_str = (f"{args.dataset}_default_"
            f"{args.load_from}_"
            f"{args.finetune_dataset}")

    logger.info(f'Starting with HPARAMS: {hp_str}')
    model_name = args.models_dir + '/' + 'eval.' + args.prefix + hp_str

    # build the model
    model = UniversalTransformer(SRC, TRG, args)

    # logger.info(str(model))
    if args.load_from is not None:
        with torch.cuda.device(args.gpu):
            # model.load_state_dict(torch.load(args.models_dir + '/' + args.load_from + '.pt',
            # map_location=lambda storage, loc: storage.cuda()))  # load the pretrained models.
            model.load_fast_weights(torch.load(args.resume_dir + '/' + args.load_from + '.pt',
            map_location=lambda storage, loc: storage.cuda()), type='meta')  # load the pretrained models.

    # use cuda
    if args.gpu > -1:
        model.cuda(args.gpu)

    # additional information
    args.__dict__.update({'model_name': model_name, 'hp_str': hp_str,  'logger': logger})

    # tensorboard writer
    if args.tensorboard and (not args.debug):
        from tensorboardX import SummaryWriter
        writer = SummaryWriter('{}/{}'.format(args.runs_dir, 'eval.' + args.prefix + args.hp_str))
    else:
        writer = None

    # show the arg:
    arg_str = "args:\n"
    for w in sorted(args.__dict__.keys()):
        if (w is not "U") and (w is not "V") and (w is not "Freq"):
            arg_str += "{}:\t{}\n".format(w, args.__dict__[w])
    logger.info(arg_str)

    # ----------------------------------------------------------------------------------------------------------------- #
    #
    # Starting Meta-Learning for Low-Resource Neural Machine Transaltion
    #
    # ----------------------------------------------------------------------------------------------------------------- #

    # if resume training
    if (args.load_from is not None) and (args.resume):
        with torch.cuda.device(args.gpu):   # very important.
            offset, opt_states = torch.load(args.resume_dir + '/' + args.load_from + '.pt.states',
                                            map_location=lambda storage, loc: storage.cuda())
    else:
        offset = 0

    print('offset {}'.format(offset))

    # ---- updates ------ #
    iters = offset
    eposides = 0
    tokens = 0
    time0 = time.time()

    def get_learning_rate(i, lr0=0.1, disable=False):
        if not disable:
            return lr0 * 10 / math.sqrt(args.d_model) * min(1 / math.sqrt(i), i / (args.warmup * math.sqrt(args.warmup)))
        return 0.00002


    def inner_loop(args, data, model, weights=None, iters=0, inner_steps=None, self_opt=None, use_prog_bar=True, inner_loop_data=None):

        set_random(seeds[iters])
        model.train()

        data_loader, data_name = data
        flag = isinstance(data_loader, (list,))

        if inner_steps is None:
            inner_steps = args.inner_steps

        if use_prog_bar:
            progressbar = tqdm(total=inner_steps, desc='start training for {}'.format(data_name))

        if weights is not None:
            model.load_fast_weights(weights)

        if self_opt is None:
            self_opt = torch.optim.Adam([p for p in model.get_parameters(type=args.finetune_params)
                                        if p.requires_grad], betas=(0.9, 0.98), eps=1e-9) # reset the optimizer

        step = 0
        for i in range(inner_steps):
            self_opt.param_groups[0]['lr'] = get_learning_rate(iters + i + 1, disable=args.disable_lr_schedule)
            self_opt.zero_grad()
            loss_inner = 0
            bs_inner = 0
            for j in range(args.inter_size):
                if not flag:
                    train_batch = next(iter(data_loader))
                else:
                    train_batch = data_loader[step]
                step += 1


                if inner_loop_data is not None:
                    inner_loop_data.append(train_batch)

                inputs, input_masks, targets, target_masks, sources, source_masks, encoding, batch_size = model.quick_prepare(train_batch)

                loss = model.cost(targets, target_masks, out=model(encoding, source_masks, inputs, input_masks)) / args.inter_size
                loss.backward()


                loss_inner = loss_inner + loss
                bs_inner = bs_inner + batch_size * max(inputs.size(1), targets.size(1))

            # update the fast-weights
            self_opt.step()
            info = '  Inner-loop[{}]: loss={:.3f}, lr={:.8f}, batch_size={}'.format(data_name, export(loss_inner), self_opt.param_groups[0]['lr'], bs_inner)


            if use_prog_bar:
                progressbar.update(1)
                progressbar.set_description(info)

        if use_prog_bar:
            progressbar.close()
        return model.save_fast_weights()



    # ----- meta-validation ----- #
    dev_iters = iters
    weights = model.save_fast_weights()
    self_opt = torch.optim.Adam([p for p in model.get_parameters(type=args.finetune_params)
                                if p.requires_grad], betas=(0.9, 0.98), eps=1e-9)
    corpus_bleu = -1

    # training start..
    best = Best(max, 'corpus_bleu', 'i', model=model, opt=self_opt, path=args.model_name, gpu=args.gpu)
    dev_metrics = Metrics('dev', 'loss', 'gleu')

    outputs_data = valid_model(args, model, dev_real, dev_metrics, print_out=False)
    corpus_bleu0 = outputs_data['corpus_bleu']
    fast_weights = [(weights, corpus_bleu0)]

    if args.tensorboard and (not args.debug):
        writer.add_scalar('dev/BLEU_corpus_', outputs_data['corpus_bleu'], dev_iters)

    for j in range(args.valid_epochs):
        args.logger.info("Fine-tuning epoch: {}".format(j))
        dev_metrics.reset()

        inner_loop(args, (train_real, "ro"), model, None, dev_iters, inner_steps=args.valid_steps, self_opt=self_opt)
        dev_iters += args.inner_steps

        outputs_data = valid_model(args, model, dev_real, dev_metrics, print_out=False)
        if args.tensorboard and (not args.debug):
            writer.add_scalar('dev/Loss', dev_metrics.loss, dev_iters)
            writer.add_scalar('dev/BLEU_corpus_', outputs_data['corpus_bleu'], dev_iters)

        if outputs_data['corpus_bleu'] > corpus_bleu:
            corpus_bleu = outputs_data['corpus_bleu']

        args.logger.info('model:' + args.prefix + args.hp_str + "\n")
        args.logger.info('used: {}s'.format(time.time() - time0) + "\n")
        fast_weights.append([model.save_fast_weights(), outputs_data['corpus_bleu']])

    if args.tensorboard and (not args.debug):
        writer.add_scalar('dev/zero_shot_BLEU', corpus_bleu0, iters)
        writer.add_scalar('dev/fine_tune_BLEU', corpus_bleu, iters)

    fast_weights = sorted(fast_weights, key=lambda a: a[1], reverse=True)

    args.logger.info('validation done.\n')
    model.load_fast_weights(fast_weights[0][0])         # --- comming back to normal

    best.accumulate(corpus_bleu, iters)
    args.logger.info('the best model is achieved at {},  corpus BLEU={}'.format(best.i, best.corpus_bleu))
    args.logger.info('perform Beam-search on |test set|:')

    dev_out = valid_model(args, model, dev_real, print_out=False, beam=4)
    tst_out = valid_model(args, model, test_real, print_out=False, beam=4)

    DEV_BLEU.append(dev_out['corpus_bleu'])
    args.logger.info('used: {}s'.format(time.time() - time0) + "\n")
    TEST_BLEU.append(tst_out['corpus_bleu'])
    args.logger.info('used: {}s'.format(time.time() - time0) + "\n")
    args.logger.info('Done.')


print('DEV', np.mean(DEV_BLEU))
print('TST', np.mean(TEST_BLEU))
