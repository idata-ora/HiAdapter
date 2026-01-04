from datetime import datetime
import argparse
import os


ROOT_DIR='HiAdapter-main'
subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
SNAPSHOT_DIR = ROOT_DIR+'/checkpoints/model/'+subdir+'/'

def parse_args():

    parser = argparse.ArgumentParser(description='Train network')

    parser.add_argument(
        '--test', 
        action='store_true', 
        help='If set, the program will run in prediction mode'
    )



    parser.add_argument('--frozen', default=1, type=int)
    parser.add_argument('--data_name', default='SPIDER-colorectal', type=str)  #
    parser.add_argument('--model_name', default='uni', type=str)    # ctranspath    conch    uni
    parser.add_argument('--batch_size', type=int, default=16)  
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--num_classes', default=13, type=int) 
    parser.add_argument('--sampler', default=0, type=int)
    parser.add_argument('--file_list_folder', default='split', type=str)
    parser.add_argument('--data_dir', default='trainval', type=str)




    parser.add_argument('--ppcl', default=1, type=int)
    parser.add_argument('--alpha', default=8, type=int) 
    parser.add_argument('--weight', default=0.00001, type=int) 
    parser.add_argument('--iternum', default=3123, type=int) 
    parser.add_argument('--hidden_size', default=1024, type=int) 
    parser.add_argument('--save_proxies', default='', type=str)

    parser.add_argument('--SI', default=1, type=int)
    parser.add_argument('--cdeep_fusion', default=1, type=int)
    parser.add_argument('--SI_folder', default='SI', type=str)

    parser.add_argument('--MA', default=1, type=int)
    parser.add_argument('--MA_folder', default='nuclei-non', type=str) #nuclei-non
    parser.add_argument('--deep_fusion_layers', nargs='+', type=int, default=[0, 6, 12, 18],
                    help='List of layer indices for deep fusion.')

    



    parser.add_argument('--folds', default='0', type=str) # 0-1-2-3-4
    parser.add_argument('--gpus', default=1,type=int)    
    parser.add_argument('--workers', default=0,type=int) 
    parser.add_argument('--loss_type',default='ce',type=str) # ce focal
    parser.add_argument('--lr',type=float, default=5e-4)
    # parser.add_argument('--weight_decay',type=float, default=5e-2)  # 1e-2

    parser.add_argument('--augment', type=int, default=1)
    parser.add_argument('--augment_index', type=str, default='0-1-2')


    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--record_epoch', type=int, default=0)
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--snapshot_dir',type=str, default=SNAPSHOT_DIR)

    parser.add_argument('--batch_size_for_test', type=int, default=1)
    parser.add_argument('--img_h', type=int, default=224)
    parser.add_argument('--img_w', type=int, default=224)

    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    return args

