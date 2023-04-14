import argparse
import logging
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

from models.models import AE_student
from utils import seed, logger, general
from dataset import Mvtec
from metric import calculate_IoU

# tensorboard command
# tensorboard --logdir=PATH
# tensorboard --logdir=results\\00003_train_2023y_4m_13d_18h_52m\\

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default=os.path.join('.', 'results'), help='dir path to save resutls')
    parser.add_argument('--dataset_path', type=str, default='.\\dataset\\mvtec_anomaly_detection\\capsule', help='dataset path(need to train, valid and test)')
    #parser.add_argument('--model_weights', type=str, default='results\\00031_train_2022y_10m_19d_10h_11m\\best_loss.pt', help='to load weights')
    parser.add_argument('--model_weights', type=str, default=None, help='to load weights')
    parser.add_argument('--ckpt_term', type=int, default=100, help='checkpoint term to save')
    parser.add_argument('--num_workers', type=int, default=4, help='set num_workers')
    parser.add_argument('--device', type=str, default='cuda:0', help='your device : cpu or cuda:0')
    parser.add_argument('--seed', type=int, default=1234, help='fix seed')

    ###    
    parser.add_argument('--batch_size', type=int, default=16, help='set batch size')
    parser.add_argument('--lr', type=float, default=1E-3, help='set learning rate')
    parser.add_argument('--epochs', type=int, default=1000, help='set epoch')
    parser.add_argument('--scheduler', type=str, default=None, help='set scheduler')
    opt = parser.parse_args()
    return opt

def set_logger(path):
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler(os.path.join(path, 'log_train.log'))
    #formatter = logging.Formatter('%(asctime)s | %(levelname)s : %(message)s')
    formatter = logging.Formatter('%(message)s')
    fileHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)
    logger.setLevel(level=logging.DEBUG)

    return logger


def train():
    threshold = 0.99
    opt = get_opt()
    mode = 'train'
    seed.seed_everything(opt.seed)
    device = opt.device
    save_dir = general.get_save_dir_name(opt, mode, makedir=True)
    
    # log
    logger = set_logger(save_dir)
    logger.info('< info >')
    logger.info('dir path : %s'%save_dir)
    logger.info('< option >')
    for k, v in opt._get_kwargs():
        logger.info('%s : %s'%(k, v))


    # tensorboard
    writer = SummaryWriter(os.path.join(save_dir, 'tensorboard'))

    # load dataset
    train_dataset = Mvtec(opt.dataset_path, mode='train')
    train_generator = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    valid_dataset = Mvtec(opt.dataset_path, mode='test')
    valid_generator = DataLoader(dataset=valid_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    # load model
    model_student = AE_student()

    # optimizer
    optimizer = optim.Adam(model_student.parameters(), lr=opt.lr)

    # scheduler
    scheduler = None
    if opt.scheduler == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # loss
    #loss = nn.L1Loss()
    loss = nn.MSELoss()

    # load model
    start_epoch = 1
    if opt.model_weights is not None and  os.path.exists(opt.model_weights):
        ckpt = torch.load(opt.model_weights)
        start_epoch += ckpt['epoch']
        model_student.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])

    model_student.to(device)
    loss.to(device)

    timer = {'total':general.Timer(), 'train':general.Timer(), 'test':general.Timer()} 

    # train
    logger.info('---------- train start ----------')
    best_loss = 1000
    best_iou = 0.0
    with timer['total']:
        for i, epoch in enumerate(range(start_epoch, opt.epochs+1)):
            with timer['train']:
                model_student.train()
                loss_train = 0.0
                for idx, batch in tqdm(enumerate(train_generator)):
                    x, y = batch[0].to(device), batch[1].to(device)
                    
                    optimizer.zero_grad()
                    preds = model_student(x)
                    loss_batch = loss(preds, x)
                    loss_batch.backward()
                    optimizer.step()
                    loss_train += loss_batch.detach().item()

                loss_train /= (idx+1)
            writer.add_scalar('Loss/train', loss_train, epoch)

            with timer['test']:
                model_student.eval()
                loss_valid = 0.0
                iou_valid = []
                for idx, batch in tqdm(enumerate(valid_generator)):
                    x, y = batch[0].to(device), batch[1].to(device)

                    with torch.no_grad():
                        preds = model_student(x)
                        label_preds = general.postprocessing(x, preds, threshold)
                        loss_batch = loss(label_preds, y)
                        loss_valid += loss_batch.detach().item()
                        iou = calculate_IoU(y.detach(), label_preds.detach())
                        iou_valid += iou.detach().cpu().numpy().tolist()

                    if idx==0:
                        if epoch == 1:
                            general.save_img_from_tensor(os.path.join(save_dir, '[%04d]input_img.png'%epoch), x[0])
                            general.save_img_from_tensor(os.path.join(save_dir, '[%04d]input_lbl.png'%epoch), y[0])
                        general.save_img_from_tensor(os.path.join(save_dir, '[%04d]pred_img.png'%epoch), preds[0])
                        general.save_img_from_tensor(os.path.join(save_dir, '[%04d]pred_lbl.png'%epoch), label_preds[0])

                loss_valid /= (idx+1)
                iou_valid = sum(iou_valid)/(len(iou_valid)+1e-6)
            writer.add_scalar('Loss/valid', loss_valid, epoch)
            writer.add_scalar('IoU/valid', iou_valid, epoch)

            logger.info('%d/%d << train loss : %.5f | valid loss: %.5f | valid iou: %.5f>>  train time: %.1f | valid time: %.1f sec'%(epoch, opt.epochs, loss_train, loss_valid, iou_valid, timer['train'].time, timer['test'].time))

            # scheduler step
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            if scheduler is not None:
                scheduler.step()

            # best loss model save
            if best_loss > loss_valid:
                best_loss = loss_valid
                best_loss_epoch = epoch
                ckpt = {
                    'epoch' : epoch,
                    'model' : model_student.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'opt' : vars(opt),
                }                
                torch.save(ckpt, os.path.join(save_dir, 'best_loss.pt'))

            # best iou model save
            if best_iou > iou_valid:
                best_iou = iou_valid
                best_iou_epoch = epoch
                ckpt = {
                    'epoch' : epoch,
                    'model' : model_student.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'opt' : vars(opt),
                }                
                torch.save(ckpt, os.path.join(save_dir, 'best_iou.pt'))

            #ckpt save  
            if  epoch % opt.ckpt_term == 0: 
                ckpt = {
                    'epoch' : epoch,
                    'model' : model_student.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'opt' : vars(opt),
                }                
                torch.save(ckpt, os.path.join(save_dir, 'ckpt', 'epoch{%d}.pt'%epoch))
        

    writer.flush()
    writer.close()
    logger.info('---------- result ----------')
    logger.info('train epoch : %d'%opt.epochs)
    logger.info('total time : %.1f sec'%(timer['total'].time))
    logger.info('best loss epoch : %d '%(best_loss_epoch))
    logger.info('best iou epoch : %d '%(best_iou_epoch))
    logger.info('best loss : %f '%(best_loss))


if __name__ == '__main__':
    train()
