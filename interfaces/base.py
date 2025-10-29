import logging
import torch
import sys
import os
import string
from collections import OrderedDict
from model.ABINet import abinet
import ptflops
from model import recognizer
from model import moran
from model import crnn
from dataset import lmdbDataset_real,alignCollate_realWTL, alignCollate_realWTLAMask,lmdbDataset_realIC15
from model.parseq.parseq import PARSeq
from model.MATRN import matrn
from utils.labelmaps import get_vocabulary
from model.StyleSRN import StyleSRN
sys.path.append('../')
from utils import ssim_psnr, utils_moran, utils_crnn


class TextBase(object):
    def __init__(self, config, args):
        super(TextBase, self).__init__()
        self.config = config
        self.args = args
        self.tpg_text_emb={ "CRNN" : 37, "PARSeq" : 95}
        self.scale_factor = self.config.TEST.down_sample_scale
        self.align_collate = alignCollate_realWTLAMask
        self.align_collate_val = alignCollate_realWTL
        if self.args.general:
            self.load_dataset_val = lmdbDataset_realIC15
        else:
            self.load_dataset_val = lmdbDataset_real
        self.resume = config.TEST.resume
        self.batch_size = self.config.TEST.batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        alpha_dict = {
            'digit': string.digits,
            'lower': string.digits + string.ascii_lowercase,
            'upper': string.digits + string.ascii_letters,
            'all': string.digits + string.ascii_letters + string.punctuation,
            'chinese': open("al_chinese.txt", "r",encoding='UTF-8').readlines()[0].replace("\n", "")
        }
        self.voc_type = self.config.TEST.voc_type
        self.alphabet = alpha_dict[self.voc_type]
        self.max_len = config.TEST.max_len
        self.vis_dir = self.config.TEST.VAL.vis_dir
        self.ckpt_path = os.path.join('output_dir', self.vis_dir)
        self.cal_psnr = ssim_psnr.calculate_psnr
        self.cal_ssim = ssim_psnr.SSIM()
        self.cal_psnr_weighted = ssim_psnr.weighted_calculate_psnr
        self.cal_ssim_weighted = ssim_psnr.SSIM_WEIGHTED()
        self.mask = self.args.mask
        alphabet_moran = ':'.join(string.digits+string.ascii_lowercase+'$')
        self.converter_moran = utils_moran.strLabelConverterForAttention(alphabet_moran, ':')
        self.converter_crnn = utils_crnn.strLabelConverter(string.digits + string.ascii_lowercase)

    def get_val_data(self):
        cfg = self.config.TEST
        assert isinstance(cfg.VAL.val_data_dir, list)
        dataset_list = []
        loader_list = []
        for data_dir_ in cfg.VAL.val_data_dir:
            val_dataset, val_loader = self.get_test_data(data_dir_)
            dataset_list.append(val_dataset)
            loader_list.append(val_loader)
        return dataset_list, loader_list

    def get_test_data(self, dir_):
        cfg = self.config.TEST

        test_dataset = self.load_dataset_val(root=dir_,
                                         voc_type=cfg.voc_type,
                                         max_len=cfg.max_len,
                                         test=True,
                                         rotate=False
                                         )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=int(cfg.workers),
            collate_fn=self.align_collate_val(imgH=cfg.height, imgW=cfg.width, down_sample_scale=cfg.down_sample_scale,
                                          mask=self.mask, train=False),
            drop_last=False)
        return test_dataset, test_loader


    def generator_init(self):
        cfg = self.config.TEST
        resume = self.resume
        model = StyleSRN(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                               STN=self.args.STN, mask=self.mask, srb_nums=self.args.srb,n_eca=self.args.n_eca,text_emb=self.tpg_text_emb[self.args.tpg]
                              )
        model = model.to(self.device)

        if not resume == '':
            logging.info('loading pre-trained model from %s ' % resume)
            if os.path.isdir(resume):
                print("resume:", resume)
                model_dict = torch.load(resume)['state_dict_G']
                model.load_state_dict(model_dict, strict=True)
            else:
                loaded_state_dict = torch.load(resume)['state_dict_G']
                model.load_state_dict(loaded_state_dict, strict=True)
        
        channel_size = 4
        macs, params = ptflops.get_model_complexity_info(model, (
            channel_size, cfg.height // cfg.down_sample_scale, cfg.width // cfg.down_sample_scale), as_strings=True,
                                                         print_per_layer_stat=False, verbose=True)
        print("---------------- SR Module -----------------")
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        print("--------------------------------------------")
        
        return model

    def parse_abinet_data(self, imgs_input):
        return imgs_input # 对输入图像做预处理。这里的abinet不需要做任何操作

    def parse_parseq_data(self, imgs_input):
        return imgs_input

    def ABINet_init(self):
        cfg = self.config.ABINet
        ABINET = abinet.ABINet_model(cfg)
        ABINET = ABINET.to(self.device)
        ABINET = torch.nn.DataParallel(ABINET, device_ids=range(self.config.TEST.ngpu))
        for p in ABINET.parameters():
            p.requires_grad = False
        ABINET.eval()
        return ABINET


    def MATRN_init(self):
        cfg = self.config.MATRN
        ABINET = matrn.MATRN(cfg)
        ABINET = ABINET.to(self.device)
        ABINET = torch.nn.DataParallel(ABINET, device_ids=range(self.config.TEST.ngpu))
        for p in ABINET.parameters():
            p.requires_grad = False
        ABINET.eval()
        return ABINET

    def PARSeq_init(self):
        cfg = self.config.PARSeq
        ABINET = PARSeq(cfg)
        ABINET = ABINET.to(self.device)
        ABINET = torch.nn.DataParallel(ABINET, device_ids=range(self.config.TEST.ngpu))
        for p in ABINET.parameters():
            p.requires_grad = False
        ABINET.eval()
        return ABINET


    def MORAN_init(self):
        cfg = self.config.TEST
        alphabet = ':'.join(string.digits+string.ascii_lowercase+'$')
        MORAN = moran.MORAN(1, len(alphabet.split(':')), 256, 32, 100, BidirDecoder=True,
                            inputDataType='torch.cuda.FloatTensor', CUDA=True)
        model_path = self.config.TEST.VAL.moran_pretrained
        print('loading pre-trained moran model from %s' % model_path)
        state_dict = torch.load(model_path)
        MORAN_state_dict_rename = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")  # remove `module.`
            MORAN_state_dict_rename[name] = v
        MORAN.load_state_dict(MORAN_state_dict_rename)
        MORAN = MORAN.to(self.device)
        MORAN = torch.nn.DataParallel(MORAN, device_ids=range(cfg.ngpu))
        for p in MORAN.parameters():
            p.requires_grad = False
        MORAN.eval()
        return MORAN

    def parse_moran_data(self, imgs_input):
        batch_size = imgs_input.shape[0]

        in_width = self.config.TEST.width if self.config.TEST.width != 128 else 100
        imgs_input = torch.nn.functional.interpolate(imgs_input, (32, in_width), mode='bicubic')
        R = imgs_input[:, 0:1, :, :]
        G = imgs_input[:, 1:2, :, :]
        B = imgs_input[:, 2:3, :, :]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        text = torch.LongTensor(batch_size * 5)
        length = torch.IntTensor(batch_size)
        max_iter = 20
        t, l = self.converter_moran.encode(['0' * max_iter] * batch_size)
        utils_moran.loadData(text, t)
        utils_moran.loadData(length, l)
        return tensor, length, text, text

    def CRNN_init(self, recognizer_path=None, opt=None):
        model = crnn.CRNN(32, 1, 37, 256)
        model = model.to(self.device)
        print("recognizer_path:", recognizer_path)
        cfg = self.config.TEST
        aster_info = VocabInfo(cfg.voc_type)
        model_path = recognizer_path if not recognizer_path is None else self.config.TEST.VAL.crnn_pretrained
        print('loading pretrained crnn model from %s' % model_path)
        stat_dict = torch.load(model_path)
        if recognizer_path is None:
            model.load_state_dict(stat_dict)
        else:
            if type(stat_dict) == OrderedDict:
                print("The dict:")
                model.load_state_dict(stat_dict)
            else:
                print("The model:")
                model = stat_dict
        return model, aster_info

    def CRNNRes18_init(self, recognizer_path=None, opt=None):
        model = crnn.CRNN_ResNet18(32, 1, 37, 256)
        model = model.to(self.device)
        cfg = self.config.TEST
        aster_info = VocabInfo(cfg.voc_type)
        model_path = recognizer_path if not recognizer_path is None else self.config.TEST.VAL.crnn_pretrained
        print('loading pretrained crnn model from %s' % model_path)
        stat_dict = torch.load(model_path)
        if recognizer_path is None:
            if stat_dict == model.state_dict():
                model.load_state_dict(stat_dict)
        else:
            model = stat_dict
        return model, aster_info

    def parse_crnn_data(self, imgs_input_, ratio_keep=False):
        in_width = self.config.TEST.width if self.config.TEST.width != 128 else 100
        if ratio_keep:
            real_height, real_width = imgs_input_.shape[2:]
            ratio = real_width / float(real_height)

            if ratio > 3:
                in_width = int(ratio * 32)
        imgs_input = torch.nn.functional.interpolate(imgs_input_, (32, in_width), mode='bicubic')

        R = imgs_input[:, 0:1, :, :]
        G = imgs_input[:, 1:2, :, :]
        B = imgs_input[:, 2:3, :, :]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        return tensor


    def Aster_init(self):
        cfg = self.config.TEST
        aster_info = VocabInfo(cfg.voc_type)
        aster = recognizer.RecognizerBuilder(arch='ResNet_ASTER', rec_num_classes=aster_info.rec_num_classes,
                                             sDim=512, attDim=512, max_len_labels=aster_info.max_len,
                                             eos=aster_info.char2id[aster_info.EOS], STN_ON=True)
        aster.load_state_dict(torch.load(self.config.TEST.VAL.rec_pretrained)['state_dict'])
        print('load pred_trained aster model from %s' % self.config.TEST.VAL.rec_pretrained)
        aster = aster.to(self.device)
        aster = torch.nn.DataParallel(aster, device_ids=range(cfg.ngpu))
        aster.eval()
        return aster, aster_info

    def parse_aster_data(self, imgs_input):
        cfg = self.config.TEST
        aster_info = VocabInfo(cfg.voc_type)
        input_dict = {}
        images_input = imgs_input.to(self.device)
        input_dict['images'] = images_input * 2 - 1
        batch_size = images_input.shape[0]
        input_dict['rec_targets'] = torch.IntTensor(batch_size, aster_info.max_len).fill_(1)
        input_dict['rec_lengths'] = [aster_info.max_len] * batch_size
        return input_dict


class VocabInfo(object):
    def __init__(self, voc_type):
        super(VocabInfo, self).__init__()
        self.voc_type = voc_type
        assert voc_type in ['digit', 'lower', 'upper', 'all', 'chinese']
        self.EOS = 'EOS'
        self.max_len = 100
        self.PADDING = 'PADDING'
        self.UNKNOWN = 'UNKNOWN'
        self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
        self.char2id = dict(zip(self.voc, range(len(self.voc))))
        self.id2char = dict(zip(range(len(self.voc)), self.voc))
        self.rec_num_classes = len(self.voc)
