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
parser.add_argument('--cross_rate', type=float, default=1,        help='randomly flipping cross objective or normal objective')
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
parser.add_argument('--optimizer',     type=str, default='Adam')
parser.add_argument('--disable_lr_schedule', action='store_true', help='disable the transformer-style learning rate')

parser.add_argument('--distillation', action='store_true', help='knowledge distillation at sequence level')
parser.add_argument('--finetuning',   action='store_true', help='knowledge distillation at word level')

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
SRCs  = [DataField(batch_first=True) for _ in [args.src] + args.aux]   # all the source languages have seperate data-fields
SP    = 4
max_src_size = -1
data_id = dict()

# setup many datasets (need to manaually setup --- Meta-Learning settings.
logger.info('start loading the dataset')

if "meta" in args.dataset:
    working_path = data_prefix + "{}/{}-{}/".format(args.dataset, args.src, args.trg)

    test_set = 'dev.tok'
    if args.finetune_dataset is None:
        train_set = 'finetune.tok'
    else:
        train_set = args.finetune_dataset

    train_data, dev_data = LazyParallelDataset.splits(path=working_path, train=train_set,
        validation=test_set, exts=('.src', '.trg'), fields=[('src', SRCs[0]), ('trg', TRG)])

    aux_data = [LazyParallelDataset(path=working_path + dataset, exts=('.src', '.trg'),
                fields=[('src', SRCs[i + 1]), ('trg', TRG)], lazy=True, max_len=100) for i, dataset in enumerate(args.aux)]
    decoding_path = working_path + '{}.' + args.src + '-' + args.trg + '.new'

    # ----------------------------------------------------------------------------------------------------- #
    vocab_src = []
    vocab_trg = []
    Us = []  # set of keys for univeral embeddings

    for i, lan in enumerate([args.src] + args.aux):
        word_count, word_vector = torch.load(args.vocab_prefix + lan + '.pt')
        U = torch.cat([torch.zeros(SP, 300), word_vector.float()], dim=0)
        SRCs[i].build_vocab_from_vocab(word_count)  # build vocab online
        Us.append(U)
        data_id[lan] = i 

        if U.size(0) > max_src_size:
            max_src_size = U.size(0)
        

    target_word_count, target_word_vector = torch.load(args.vocab_prefix + args.trg + '.pt')
    V = target_word_vector.float()[:20000, :]

    # build vocab online
    TRG.build_vocab_from_vocab(target_word_count)


    if args.gpu > -1:
        for i in range(len(Us)):
            Us[i] = Variable(Us[i]).cuda(args.gpu)
        V = V.cuda(args.gpu)

    args.__dict__.update({'U': None,  'V': V, 'max_src_size': max_src_size, 'unitok_size': V.size(0)})

else:
    raise NotImplementedError

logger.info('load dataset done.. trg: {},  V: {}'.format(len(TRG.vocab), V.size(0)))
for i, S in enumerate(SRCs):
    logger.info('{}: {}'.format(args.src if i == 0 else args.aux[i -1], len(S.vocab)))

args.__dict__.update({'trg_vocab': len(TRG.vocab), 'src_vocab': max_src_size})

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

train_real, dev_real = data.BucketIterator.splits(
    (train_data, dev_data), batch_sizes=(args.batch_size, args.batch_size), device=args.gpu, shuffle=False,
    batch_size_fn=batch_size_fn, repeat=None if args.mode == 'train' else False)
aux_reals = [data.BucketIterator(dataset, batch_size=args.batch_size, device=args.gpu, train=True, batch_size_fn=batch_size_fn, shuffle=False)
            for dataset in aux_data]
logger.info("build the dataset. done!")


# ----------------------------------------------------------------------------------------------------------------- #
# model hyper-params:
logger.info('use default parameters of t2t-base')
hparams = {'d_model': 512, 'd_hidden': 512, 'n_layers': 6,
            'n_heads': 8, 'drop_ratio': 0.1, 'warmup': 16000} # ~32
args.__dict__.update(hparams)

# ----------------------------------------------------------------------------------------------------------------- #
# show the arg:

# hp_str = (f"{args.dataset}_subword_"
#           f"{args.d_model}_{args.d_hidden}_{args.n_layers}_{args.n_heads}_"
#           f"{args.drop_ratio:.3f}_{args.warmup}_{'universal_' if args.universal else ''}_meta")
languages = ''.join([args.src, '-', args.trg, '-'] + args.aux)
hp_str = (f"{args.dataset}_default_"
          f"{languages}_"
          f"{'universal' if args.universal else ''}_"
          f"{'' if args.no_meta_training else 'meta'}_"
          f"{'2apx' if args.meta_approx_2nd else ''}_"
          f"{'cross' if args.cross_meta_learning else ''}_"
          f"{args.inter_size*args.batch_size}_{args.inner_steps}")

logger.info(f'Starting with HPARAMS: {hp_str}')
model_name = args.models_dir + '/' + args.prefix + hp_str

# build the model
model = UniversalTransformer(SRCs[0], TRG, args)

# logger.info(str(model))
if args.load_from is not None:
    with torch.cuda.device(args.gpu):
        model.load_state_dict(torch.load(args.models_dir + '/' + args.load_from + '.pt',
        map_location=lambda storage, loc: storage.cuda()))  # load the pretrained models.


# use cuda
if args.gpu > -1:
    model.cuda(args.gpu)

# additional information
args.__dict__.update({'model_name': model_name, 'hp_str': hp_str,  'logger': logger,  'n_lang': len(args.aux)})

# tensorboard writer
if args.tensorboard and (not args.debug):
    from tensorboardX import SummaryWriter
    writer = SummaryWriter('{}/{}'.format(args.runs_dir, args.prefix + args.hp_str))
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

# optimizer
meta_opt = torch.optim.Adam([p for p in model.get_parameters(type='meta' if not args.no_meta_training else 'full')
                            if p.requires_grad], betas=(0.9, 0.98), eps=1e-9)
if args.meta_approx_2nd:
    sgd_opt = torch.optim.SGD([p for p in model.get_parameters(type='meta0') if p.requires_grad], lr=args.approx_lr)

 # if resume training
if (args.load_from is not None) and (args.resume):
    with torch.cuda.device(args.gpu):   # very important.
        offset, opt_states = torch.load(args.models_dir + '/' + args.load_from + '.pt.states',
                                        map_location=lambda storage, loc: storage.cuda())
        meta_opt.load_state_dict(opt_states)
else:
    offset = 0

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
    lang_id = data_id[data_name]
    lang_U  = Us[lang_id]

    flag = isinstance(data_loader, (list,))

    if inner_steps is None:
        inner_steps = args.inner_steps

    if use_prog_bar:
        progressbar = tqdm(total=inner_steps, desc='start training for {}'.format(data_name))

    if weights is not None:
        model.load_fast_weights(weights)

    if self_opt is None:
        self_opt = torch.optim.Adam([p for p in model.get_parameters(type='fast') if p.requires_grad], betas=(0.9, 0.98), eps=1e-9) # reset the optimizer

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

            inputs, input_masks, targets, target_masks, sources, source_masks, encoding, batch_size = model.quick_prepare(train_batch, U=lang_U)
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


# training start..
best = Best(max, 'corpus_bleu', 'i', model=model, opt=meta_opt, path=args.model_name, gpu=args.gpu)
train_metrics = Metrics('train', 'loss', 'real', 'fake')
dev_metrics = Metrics('dev', 'loss', 'gleu', 'real_loss', 'fake_loss', 'distance', 'alter_loss', 'distance2', 'fertility_loss', 'corpus_gleu')

# overlall progress-ba
progressbar = tqdm(total=args.eval_every, desc='start training')


while True:

    # ----- saving the checkpoint ----- #
    if iters % args.save_every == 0:
        args.logger.info('save (back-up) checkpoints at iter={}'.format(iters))
        with torch.cuda.device(args.gpu):
            torch.save(best.model.state_dict(), '{}_iter={}.pt'.format(args.model_name, iters))
            torch.save([iters, best.opt.state_dict()], '{}_iter={}.pt.states'.format(args.model_name, iters))

    # ----- meta-validation ----- #
    if iters % args.eval_every == 0:

        progressbar.close()
        dev_iters = iters
        weights = model.save_fast_weights()
        lang_U = Us[0]

        fast_weights = weights
        self_opt = torch.optim.Adam([p for p in model.get_parameters(type='fast') if p.requires_grad], betas=(0.9, 0.98), eps=1e-9)
        corpus_bleu = -1

        outputs_data = valid_model(args, model, dev_real, dev_metrics, print_out=True, U=lang_U)
        corpus_bleu0 = outputs_data['corpus_bleu']

        if args.tensorboard and (not args.debug):
            writer.add_scalar('dev/BLEU_corpus_', outputs_data['corpus_bleu'], dev_iters)

        for j in range(args.valid_epochs):
            args.logger.info("Fine-tuning epoch: {}".format(j))
            dev_metrics.reset()

            inner_loop(args, (train_real, args.src), model, None, dev_iters, inner_steps=args.valid_steps, self_opt=self_opt)
            dev_iters += args.inner_steps

            if j > 1:
                outputs_data = valid_model(args, model, dev_real, dev_metrics, print_out=True, U=lang_U)
                if args.tensorboard and (not args.debug):
                    writer.add_scalar('dev/Loss', dev_metrics.loss, dev_iters)
                    writer.add_scalar('dev/BLEU_corpus_', outputs_data['corpus_bleu'], dev_iters)

                if outputs_data['corpus_bleu'] > corpus_bleu:
                    corpus_bleu = outputs_data['corpus_bleu']

                args.logger.info('model:' + args.prefix + args.hp_str + "\n")

        if args.tensorboard and (not args.debug):
            writer.add_scalar('dev/zero_shot_BLEU', corpus_bleu0, iters)
            writer.add_scalar('dev/fine_tune_BLEU', corpus_bleu, iters)


        args.logger.info('validation done.\n')
        model.load_fast_weights(weights)         # --- comming back to normal

        # -- restart the progressbar --
        progressbar = tqdm(total=args.eval_every, desc='start training')

        if not args.debug:
            best.accumulate(corpus_bleu, iters)
            args.logger.info('the best model is achieved at {},  corpus BLEU={}'.format(
                best.i, best.corpus_bleu))

    # ----- meta-training ------- #
    model.train()
    if iters > args.maximum_steps:
        args.logger.info('reach the maximum updating steps.')
        break

    # ----- inner-loop ------
    selected = random.randint(0, args.n_lang - 1)  # randomly pick one language pair
    if args.cross_meta_learning:
        
        p = random.random()
        if p < args.cross_rate:
            selected2 = random.randint(0, args.n_lang - 1)
            cross_flag = True
        else:
            selected2 = selected
            cross_flag = False

    if not args.no_meta_training:  # ----- only meta-learning requires inner-loop
        inner_loop_data = []
        weights = model.save_fast_weights()  # in case the data has been changed...
        fast_weights = inner_loop(args, (aux_reals[selected], args.aux[selected]), model, iters = iters, use_prog_bar=False, inner_loop_data=inner_loop_data)

    # ------ outer-loop -----
    meta_opt.param_groups[0]['lr'] = get_learning_rate(iters + 1, disable=args.disable_lr_schedule)
    meta_opt.zero_grad()

    loss_outer = 0
    bs_outter = 0

    if args.cross_meta_learning and cross_flag:
        model.encoder.out.weight.data[SP:, :].zero_() # zero out the embeddings as we come to a new language (no-self embeddings)

    for j in range(args.inter_size):

        if not args.cross_meta_learning:
            meta_train_batch = next(iter(aux_reals[selected]))
            lang_U = Us[selected + 1]
        else:
            meta_train_batch = next(iter(aux_reals[selected2]))
            lang_U = Us[selected2 + 1]

        inputs, input_masks, targets, target_masks, sources, source_masks, encoding, batch_size = model.quick_prepare(meta_train_batch, U=lang_U)
        loss = model.cost(targets, target_masks, out=model(encoding, source_masks, inputs, input_masks)) / args.inter_size
        loss.backward()

        loss_outer = loss_outer + loss
        bs_outter = bs_outter + batch_size * max(inputs.size(1), targets.size(1))

    # update the meta-parameters
    if not args.no_meta_training:
        model.load_fast_weights(weights)
        if args.meta_approx_2nd:
            meta_grad = model.save_fast_gradients('meta')

            sgd_opt.param_groups[0]['lr'] = args.approx_lr
            sgd_opt.step()   # --- update the parameters using SGD ---
            # print(model.grad_sum(meta_grad))
            fast_weights2 = inner_loop(args, (inner_loop_data, args.aux[selected]), model, iters = iters, use_prog_bar=False) # inner loop agains

            # sgd_opt.param_groups[0]['lr'] = -args.approx_lr
            # model.load_fast_weights(weights)
            # model.load_fast_gradients(meta_grad, 'meta')
            # sgd_opt.step()
            # fast_weights3 = inner_loop(args, (inner_loop_data, args.aux[selected]), model, iters = iters, use_prog_bar=False)

            # compute new gradient:
            for name in meta_grad:
                if name in fast_weights:
                    meta_grad[name].add_(0.01 * (fast_weights[name] - fast_weights2[name]) / args.approx_lr)

            # print(model.grad_sum(meta_grad))
            model.load_fast_gradients(meta_grad, 'meta')
            model.load_fast_weights(weights)

    meta_opt.step()
    info = 'Outer: loss={:.3f}, lr={:.8f}, batch_size={}, eposides={}'.format(
        export(loss_outer), meta_opt.param_groups[0]['lr'], bs_outter, iters)
    progressbar.update(1)
    progressbar.set_description(info)
    tokens = tokens + bs_outter

    if args.tensorboard and (not args.debug):
        writer.add_scalar('train/Loss', export(loss_outer), iters + 1)


    # ---- zero the self-embedding matrix
    if not args.no_meta_training:
        model.encoder.out.weight.data[SP:, :].zero_() # ignore the first special tokens.


    iters = iters + 1
    eposides = eposides + 1

    def hms(sec_elapsed):
        h = int(sec_elapsed / (60 * 60))
        m = int((sec_elapsed % (60 * 60)) / 60)
        s = sec_elapsed % 60.
        return "{}:{:>02}:{:>05.2f}".format(h, m, s)
    # args.logger.info("Training {} tokens / {} batches / {} episodes, ends with: {}\n".format(tokens, iters, eposides, hms(time.time() - time0)))

args.logger.info('Done.')
