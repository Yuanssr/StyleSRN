import torch
from datetime import datetime
from utils import ssim_psnr
from interfaces import base
from utils.metrics import get_string_aster, get_string_crnn,get_string_abinet,get_string_parseq
from utils.util import str_filt
import time
import logging
from dataset.dataset import Lable2Tensor
import lpips
import setup
from model.parseq.parseq_tokenizer import get_parseq_tokenize

curr_device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
parseq_tokenizer = get_parseq_tokenize()
abi_charset = setup.CharsetMapper()
label2tensor = Lable2Tensor()
ssim = ssim_psnr.SSIM()
lpips_vgg = lpips.LPIPS(net="vgg").to(curr_device)

class TextSR(base.TextBase):

    def model_inference(self, images_lr, model,tpg_bible):

        data_in_lr = tpg_bible[self.args.tpg]["data_in_fn"](images_lr[:, :3, :, :])
        out_dict=tpg_bible[self.args.tpg]["model"](data_in_lr)
        if self.args.tpg == "CRNN":
            label_vecs_logits = out_dict
            label_vecs = torch.nn.functional.softmax(label_vecs_logits, -1)
            label_vecs_final = label_vecs.permute(1, 2, 0).unsqueeze(2)
        elif self.args.tpg == "PARSeq":
            label_vecs_logits = out_dict['logits']
            label_vecs = torch.nn.functional.softmax(label_vecs_logits, -1)
            label_vecs_final = label_vecs.permute(0, 2, 1).unsqueeze(2)
        ret_dict = {}
        ret_dict['duration']= 0.
        images_sr = []
        before = time.time()
        cascade_images= model(images_lr,label_vecs_final.detach())
        after = time.time()
        # print("fps:", (after - before))
        ret_dict["duration"] += (after - before)
        images_sr.append(cascade_images)
        ret_dict["images_sr"] = images_sr

        return ret_dict

    def eval(self, model, val_loader, aster, aster_info,tpg_bible):
        metric_dict = {
            'psnr_lr': [],
            'ssim_lr': [],
            'ssi_lr': [],
            'cnt_psnr_lr': [],
            'cnt_ssim_lr': [],
            'cnt_ssi_lr': [],
            'psnr': [],
            'ssim': [],
            'ssi': [],
            'cnt_psnr': [],
            'cnt_ssim': [],
            'cnt_ssi': [],
            'accuracy': 0.0,
            'psnr_avg': 0.0,
            'ssim_avg': 0.0,
            'ssi_avg': 0.0,
            'edis_LR': [],
            'edis_SR': [],
            'edis_HR': [],
            'LPIPS_VGG_LR': [],
            'LPIPS_VGG_SR': []
        }
        n_correct_lr = 0
        n_correct_hr = 0
        sum_images = 0
        counters = {0: 0}
        image_counter = 0
        rec_str = ""
        sr_infer_time = 0
        self.string_process_fn = aster[0]["string_process"]

        with torch.no_grad():

            for i, data in (enumerate(val_loader)):

                images_hrraw, images_lrraw, images_HRy, images_lry, label_strs, label_vecs_gt = data
                # print("label_strs:", label_strs)
                images_lr = images_lrraw.to(self.device)
                images_hr = images_hrraw.to(self.device)
                val_batch_size = images_lr.shape[0]

                ret_dict = self.model_inference(images_lr, model, tpg_bible)

                sr_infer_time += ret_dict["duration"]

                images_sr = ret_dict["images_sr"]
                # == 首先解析输入图像为文本识别器的输入形式 =======
                aster_dict_lr = aster[0]["data_in_fn"](images_lr[:, :3, :, :])
                aster_dict_hr = aster[0]["data_in_fn"](images_hr[:, :3, :, :])
                # ==== 之后是将解析后的图像用于文本识别  这里只对LR和HR进行识别====
                if self.args.test_model == "MORAN":
                    # LR
                    aster_output_lr = aster[0]["model"](
                        aster_dict_lr[0],
                        aster_dict_lr[1],
                        aster_dict_lr[2],
                        aster_dict_lr[3],
                        test=True,
                        debug=True
                    )
                    # HR
                    aster_output_hr = aster[0]["model"](
                        aster_dict_hr[0],
                        aster_dict_hr[1],
                        aster_dict_hr[2],
                        aster_dict_hr[3],
                        test=True,
                        debug=True
                    )
                else:
                    aster_output_lr = aster[0]["model"](aster_dict_lr)
                    aster_output_hr = aster[0]["model"](aster_dict_hr)

                # ============对SR进行图像解析和通过文本识别器 ==============
                if type(images_sr) == list:
                    predict_result_sr = []
                    image = images_sr[0]
                    aster_dict_sr = aster[0]["data_in_fn"](image[:, :3, :, :])

                    if self.args.test_model == "MORAN":
                        # aster_output_sr = aster[0]["model"](*aster_dict_sr)
                        aster_output_sr = aster[0]["model"](
                            aster_dict_sr[0],
                            aster_dict_sr[1],
                            aster_dict_sr[2],
                            aster_dict_sr[3],
                            test=True,
                            debug=True
                        )
                    else:
                        aster_output_sr = aster[0]["model"](aster_dict_sr)

                    # outputs_sr = aster_output_sr.permute(1, 0, 2).contiguous()

                    if self.args.test_model == "CRNN":

                        predict_result_sr_ = self.string_process_fn(aster_output_sr)

                    elif self.args.test_model == "ASTER":
                        predict_result_sr_, _ = self.string_process_fn(
                            aster_output_sr['output']['pred_rec'],
                            aster_dict_sr['rec_targets'],
                            dataset=aster_info
                        )
                    elif self.args.test_model == "MORAN":
                        preds, preds_reverse = aster_output_sr[0]
                        _, preds = preds.max(1)
                        sim_preds = self.converter_moran.decode(preds.data, aster_dict_sr[1].data)
                        predict_result_sr_ = [pred.split('$')[0] for pred in sim_preds]

                    elif self.args.test_model in ["ABINet", 'MATRN']:
                        # 对识别字典做处理
                        predict_result_sr_ = aster[0]['string_process'](aster_output_sr, abi_charset)[0]

                    elif self.args.test_model in ['PARSeq']:
                        # 对parseq的识别字典logits做处理
                        predict_result_sr_ = aster[0]['string_process'](aster_output_sr, parseq_tokenizer)

                    predict_result_sr.append(predict_result_sr_)

                    img_lr = torch.nn.functional.interpolate(images_lr, images_hr.shape[-2:], mode="bicubic")
                    img_sr = torch.nn.functional.interpolate(images_sr[-1], images_hr.shape[-2:], mode="bicubic")

                    metric_dict['psnr'].append(self.cal_psnr(img_sr[:, :3], images_hr[:, :3]))
                    metric_dict['ssim'].append(self.cal_ssim(img_sr[:, :3], images_hr[:, :3]))

                    metric_dict["LPIPS_VGG_SR"].append(
                        lpips_vgg(img_sr[:, :3], images_hr[:, :3]).data[0].reshape(-1)[0])

                    metric_dict['psnr_lr'].append(self.cal_psnr(img_lr[:, :3], images_hr[:, :3]))
                    metric_dict['ssim_lr'].append(self.cal_ssim(img_lr[:, :3], images_hr[:, :3]))


                    metric_dict["LPIPS_VGG_LR"].append(
                        lpips_vgg(img_lr[:, :3], images_hr[:, :3]).data[0].reshape(-1)[0])

                if self.args.test_model == "CRNN":  # 之前是只对SR的识别结果进行的处理，这里将HR和LR同样进行处理

                    predict_result_lr = self.string_process_fn(aster_output_lr)

                    predict_result_hr = self.string_process_fn(aster_output_hr)
                elif self.args.test_model == "ASTER":
                    predict_result_lr, _ = self.string_process_fn(
                        aster_output_lr['output']['pred_rec'],
                        aster_dict_lr['rec_targets'],
                        dataset=aster_info
                    )
                    predict_result_hr, _ = self.string_process_fn(
                        aster_output_hr['output']['pred_rec'],
                        aster_dict_hr['rec_targets'],
                        dataset=aster_info
                    )
                elif self.args.test_model == "MORAN":
                    ### LR ###
                    preds, preds_reverse = aster_output_lr[0]
                    _, preds = preds.max(1)
                    sim_preds = self.converter_moran.decode(preds.data, aster_dict_lr[1].data)
                    predict_result_lr = [pred.split('$')[0] for pred in sim_preds]

                    ### HR ###
                    preds, preds_reverse = aster_output_hr[0]
                    _, preds = preds.max(1)
                    sim_preds = self.converter_moran.decode(preds.data, aster_dict_hr[1].data)
                    predict_result_hr = [pred.split('$')[0] for pred in sim_preds]

                elif self.args.test_model in ['ABINet', 'MATRN']:
                    predict_result_lr = self.string_process_fn(aster_output_lr, abi_charset)[0]
                    predict_result_hr = self.string_process_fn(aster_output_hr, abi_charset)[0]
                elif self.args.test_model in ['PARSeq']:
                    predict_result_lr = self.string_process_fn(aster_output_lr, parseq_tokenizer)
                    predict_result_hr = self.string_process_fn(aster_output_hr, parseq_tokenizer)
                filter_mode = 'lower'

                for batch_i in range(images_lr.shape[0]):
                    label = label_strs[batch_i]
                    image_counter += 1
                    rec_str += str(image_counter) + ".jpg," + label + "\n"

                    if str_filt(predict_result_sr[0][batch_i], filter_mode) == str_filt(label, filter_mode):
                        counters[0] += 1
                    if str_filt(predict_result_lr[batch_i], filter_mode) == str_filt(label, filter_mode):
                        n_correct_lr += 1
                    if str_filt(predict_result_hr[batch_i], filter_mode) == str_filt(label, filter_mode):
                        n_correct_hr += 1
                sum_images += val_batch_size
            torch.cuda.empty_cache()

            # 已经把整个测试集跑完
            psnr_avg = sum(metric_dict['psnr']) / (len(metric_dict['psnr']) + 1e-10)
            ssim_avg = sum(metric_dict['ssim']) / (len(metric_dict['ssim']) + 1e-10)

            psnr_avg_lr = sum(metric_dict['psnr_lr']) / (len(metric_dict['psnr_lr']) + 1e-10)
            ssim_avg_lr = sum(metric_dict['ssim_lr']) / (len(metric_dict['ssim_lr']) + 1e-10)

            lpips_vgg_lr = sum(metric_dict["LPIPS_VGG_LR"]) / (len(metric_dict['LPIPS_VGG_LR']) + 1e-10)
            lpips_vgg_sr = sum(metric_dict["LPIPS_VGG_SR"]) / (len(metric_dict['LPIPS_VGG_SR']) + 1e-10)

            logging.info('[{}]\t'
                         'loss_rec {:.3f}| loss_im {:.3f}\t'
                         'PSNR {:.2f} | SSIM {:.4f}\t'
                         'LPIPS {:.4f}  '
                         .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                 0, 0,
                                 float(psnr_avg), float(ssim_avg), float(lpips_vgg_sr)))

            logging.info('[{}]\t'
                         'PSNR_LR {:.2f} | SSIM_LR {:.4f}\t'
                         'LPIPS_LR {:.4f} '
                         .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                 float(psnr_avg_lr), float(ssim_avg_lr), float(lpips_vgg_lr)))

            logging.info('save display images')

            accuracy = round(counters[0] / sum_images, 4)  # loader已经跑完了，把全部的测试集图片和全部正确个数相除

            accuracy_lr = round(n_correct_lr / sum_images, 4)
            accuracy_hr = round(n_correct_hr / sum_images, 4)
            psnr_avg = round(psnr_avg.item(), 6)
            ssim_avg = round(ssim_avg.item(), 6)

            logging.info('sr_accuray_iter' + ': %.2f%%' % (accuracy * 100))

            logging.info('lr_accuray: %.2f%%' % (accuracy_lr * 100))
            logging.info('hr_accuray: %.2f%%' % (accuracy_hr * 100))
            metric_dict['accuracy'] = accuracy
            metric_dict['psnr_avg'] = psnr_avg
            metric_dict['ssim_avg'] = ssim_avg

            inference_time = sum_images / sr_infer_time
            logging.info("AVG inference:{}".format(inference_time))
            logging.info("sum_images:{}".format(sum_images))

            return metric_dict

    def test(self):

        total_acc = {}
        total_psnr = {}
        total_ssim = {}
        tpg_bible= {}
        if self.args.tpg == "CRNN":
            crnn, aster_info = self.CRNN_init()
            crnn.eval()
            tpg_bible["CRNN"] = {
                        'model': crnn,
                        'data_in_fn': self.parse_crnn_data,
                        'string_process': get_string_crnn
                        }
        elif self.args.tpg == 'PARSeq':
            parseq = self.PARSeq_init()
            tpg_bible["PARSeq"] = {
                'model': parseq,
                'data_in_fn': self.parse_parseq_data,
                'string_process': get_string_parseq
            }

        val_dataset_list, val_loader_list = self.get_val_data()
        model = self.generator_init()
        logging.info('Using text recognizer {}'.format(self.args.test_model))
        test_bible = {}
        aster_info = None
        if self.args.test_model == "CRNN":
            crnn, aster_info = self.CRNN_init()
            crnn.eval()
            test_bible["CRNN"] = {
                'model': crnn,
                'data_in_fn': self.parse_crnn_data,
                'string_process': get_string_crnn
            }
        elif self.args.test_model == "ASTER":
            aster_real, aster_real_info = self.Aster_init()  # init ASTER model
            aster_info = aster_real_info
            test_bible["ASTER"] = {
                'model': aster_real,
                'data_in_fn': self.parse_aster_data,
                'string_process': get_string_aster
            }

        elif self.args.test_model == "MORAN":
            moran = self.MORAN_init()
            aster_info = None
            if isinstance(moran, torch.nn.DataParallel):
                moran.device_ids = [0]
            test_bible["MORAN"] = {
                'model': moran,
                'data_in_fn': self.parse_moran_data,
                'string_process': get_string_crnn
            }
        elif self.args.test_model == 'ABINet':
            abinet = self.ABINet_init()
            test_bible["ABINet"] = {
                'model': abinet,
                'data_in_fn': self.parse_abinet_data,
                'string_process': get_string_abinet
            }
        elif self.args.test_model == 'MATRN':
            matrn = self.MATRN_init()
            test_bible["MATRN"] = {
                'model': matrn,
                'data_in_fn': self.parse_abinet_data,
                'string_process': get_string_abinet
            }
        elif self.args.test_model == 'PARSeq':
            parseq = self.PARSeq_init()
            test_bible["PARSeq"] = {
                'model': parseq,
                'data_in_fn': self.parse_parseq_data,
                'string_process': get_string_parseq
            }

        for k, val_loader in enumerate(val_loader_list):
            data_name = self.config.TEST.VAL.val_data_dir[k].split('/')[-1]
            logging.info('testing %s' % data_name)
            model.eval()
            for p in model.parameters():
                p.requires_grad = False

            metrics_dict = self.eval(
                model,
                val_loader,
                [test_bible[self.args.test_model], None, None],
                aster_info,
                tpg_bible,
            )
            acc = float(metrics_dict['accuracy'])
            psnr = float(metrics_dict['psnr_avg'])
            ssim = float(metrics_dict['ssim_avg'])
            total_acc[data_name] = acc
            total_psnr[data_name] = psnr
            total_ssim[data_name] = ssim
            logging.info('best_%s = %.2f%%' % (data_name, acc * 100))
        if not self.args.general:
            logging.info('avg_acc(Easy,Medium,Hard) is {:.3f}'.format(100 * (total_acc['easy'] * 1619
                                                                             + total_acc['medium'] * 1411
                                                                             + total_acc['hard'] * 1343) / (
                                                                                  1343 + 1411 + 1619)))
            logging.info('avg_psnr(Easy,Medium,Hard) is {:.4f}'.format((total_psnr['easy'] * 1619
                                                                        + total_psnr['medium'] * 1411
                                                                        + total_psnr['hard'] * 1343) / (
                                                                               1343 + 1411 + 1619)))
            logging.info('avg_ssim(Easy,Medium,Hard) is {:.4f}'.format((total_ssim['easy'] * 1619
                                                                        + total_ssim['medium'] * 1411
                                                                        + total_ssim['hard'] * 1343) / (
                                                                               1343 + 1411 + 1619)))
            logging.info('Test with recognizer {} finished!'.format(self.args.test_model))