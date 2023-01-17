import os
from pprint import pformat
from pathlib import Path
from argparse import ArgumentParser
import yaml
import numpy as np

import time
import sys
import timeit

import torch
import ignite
from ignite.utils import setup_logger
from ignite.utils import manual_seed
import pandas as pd
import logging
from logging import Logger
from torch.utils.data import DataLoader, Dataset
from typing import List, Any, Optional
from joblib import Parallel, delayed
from tqdm import tqdm

from osgeo import gdal_array

from PIL import Image

import rasterio
rasterio.warnings.simplefilter('ignore')

from libtiff import TIFF
import libtiff
libtiff.libtiff_ctypes.suppress_warnings()

from benchmarking.misc.init_benchmarking import get_dataset
from concurrent_dataloader.dataloader_mod.dataloader import DataLoader as DataLoaderParallel
from concurrent_dataloader.dataloader_mod.worker import _worker_loop as _worker_loop_parallel


class base_stats():
    def __init__(self):
        self.min = None
        self.max = None
        self.ave = None
        self.count = 0
        self.var = None
        self.sum = None
        self.sum_sq = None
        
    def step(self, params):
        if self.min is None:
            self.min = [p for p in params]
        else:
            self.min = [min(p,o) for p,o in zip(params, self.min)]

        if self.max is None:
            self.max = [p for p in params]
        else:
            self.max = [max(p,o) for p,o in zip(params, self.max)]

        if self.ave is None:
            self.ave = [p for p in params]
            self.sum = [p for p in params]
            self.count = 1
        else:
            self.sum = [p+o for p,o in zip (params, self.sum)]
            self.count += 1
            self.ave = [o/self.count for o in self.sum]

        if self.var is None:            
            self.sum_sq = [ p*p for p in params]
            self.var = [ ss/self.count - s*s for ss, s in zip(self.sum_sq, self.ave)]
        else:
            self.sum_sq = [ o + p*p for p, o in zip(params, self.sum_sq)]
            self.var = [ ss/self.count - s*s for ss, s in zip(self.sum_sq, self.ave)]
            
    def get(self):
        res = {}
        res['min'] = self.min
        res['max'] = self.max
        res['ave'] = self.ave
        res['var'] = self.var
        res['std'] = [p**0.5 for p in self.var]
        
        return res      

def setup_logging(config: Any) -> Logger:
    """Setup logger with `ignite.utils.setup_logger()`.

    Parameters
    ----------
    config
        config object. config has to contain `verbose` and `output_dir` attribute.

    Returns
    -------
    logger
        an instance of `Logger`
    """
    # green = "\033[32m"
    # reset = "\033[0m"
    logger = setup_logger(
        name="[ignite]",
        level=logging.DEBUG if config.debug else logging.INFO,
        format="%(name)s: %(message)s",
        filepath=os.path.join(config.output_dir, "training-info.log"),
    )
    return logger


class Struct(object):
    """
        Helper to go from dict to field like structs]
    """

    def __init__(self, data):
        for name, value in data.items():
            setattr(self, str(name), self._wrap(value))

    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)):
            return type(value)([self._wrap(v) for v in value])
        else:
            return Struct(value) if isinstance(value, dict) else value


class Config(object):

    def __init__(self, conf_file):
        file = open(conf_file, 'r')
        self.infos = yaml.load(file, Loader=yaml.FullLoader)
        file.close()
        for name, value in self.infos.items():
            setattr(self, str(name), self._wrap(value))

    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)):
            return type(value)([self._wrap(v) for v in value])
        else:
            return value
            # return Struct(value) if  isinstance(value, dict) else value

    def dump_config_to_file(self,filename):
        f = open(filename, "w")
        yaml.dump(self.infos, f)
        f.close()


class StackDataset(Dataset):
    """Reads in images, transforms pixel values, and serves a
    dictionary containing chip ids, image tensors, and
    label masks (where available).
    """

    def __init__(self,
                 dataset: pd.DataFrame,
                 img_folder: str,
                 logger:Optional[Any] = None,
                 use_gdal:Optional[bool] = False,
                 use_PIL:Optional[bool] = False,
                 use_TIFF:Optional[bool] = False,
                 use_RIO:Optional[bool] = False,
    ):
        """
        Instantiate the CloudDataset class.

        Args:
            dataset (pd.DataFrame): a dataframe with a row for each chip. There must be a column for chip_id,
                and a column with the path to the TIF for each of bands
            bands (list[str]): list of the bands included in the data
            labels (pd.DataFrame, optional): a dataframe with, for each chip, columns for chip_id
                and the path to the label TIF with ground truth cloud cover
        """
        super().__init__()
        self.base_dataset = dataset
        self.img_folder = img_folder
        self.use_gdal = use_gdal
        self.use_PIL = use_PIL
        self.use_TIFF = use_TIFF
        self.use_RIO = use_RIO        
        self.dataset = self.base_dataset
        self.logger = logger

        assert any ([self.use_gdal, self.use_PIL, self.use_TIFF, self.use_RIO]), ' At least one backend should be on '

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):

        row = self.dataset.iloc[idx]
        img_path = os.path.join(self.img_folder, row["in"].replace('in:', ''))
        #start = time.monotonic()
        if self.use_gdal:
            x_arr = gdal_array.LoadFile(img_path)
            #x_arr = np.random.randn(512,512,6)
        elif self.use_PIL:
            im = Image.open(img_path, mode='r')
            im.load()
            x_arr = np.array( im , dtype=np.float32)
        elif self.use_TIFF:
            tif = TIFF.open(img_path, mode='r')
            x_arr = tif.read_image()
            tif = None
        else:
            with rasterio.open(img_path) as src:
                x_arr = src.read()
            #x_arr = np.random.randn(512,512,6)
            #x_arr = np.random.randn(256,256,3)
        #end = time.monotonic()
        #self.logger.info(f' \t \t raster io fetch time is {end - start}')
        start = time.monotonic()
        x_arr = x_arr.astype("float32")
        #end = time.monotonic()
        #self.logger.info(f' \t \t converting to float32 {end - start}')
        #self.logger.info(f' \t \t raster size is {x_arr.shape}')
        #start = time.monotonic()
        #x_arr = x_arr.transpose(1, 2, 0)
        x_arr = x_arr / eval(row["range_max"].replace("range_max:", ''))
        end = time.monotonic()
        #self.logger.info(f' \t \t transpose + scale time is {end - start}')

        return x_arr


def measure_grab_time(dataset_iterator):
    start = time.monotonic()
    #x,y = next(dataset_iterator)
    x = next(dataset_iterator)
    end = time.monotonic()
    return end-start



def bench_joblib(config, train_csv, img_folder, logger, use_gdal=False):
    #
    #      BENCH GDAL
    #
    dataset_train = StackDataset(dataset=train_csv,
                                 img_folder=img_folder,
                                 logger=logger,
                                 use_gdal = use_gdal)

    it = iter(dataset_train)

    waiting_time_list = []

    backend = 'threading'

    start_batching = time.monotonic()
    num_workers = config.num_workers
    num_batches = config.num_batches
    batch_size = config.batch_size

    method = 'GDAL' if use_gdal else 'RASTERIO'
    
    logger.info(f' *********  XP WITH MANUAL {method}// DATA ACCESS ************************************** ')

    for batch in tqdm(range(num_batches)):
        start = time.monotonic()
        waiting_time_list = Parallel(n_jobs=num_workers, backend=backend, pre_dispatch=2*num_workers)(
            delayed(measure_grab_time)(it) for _ in range(batch_size)
        )
        end = time.monotonic()
        #ave = np.mean(waiting_time_list)
        #logger.info(f' *********************************************** ')
        #logger.info(f' overall time spent on batch {batch} is {end-start} -- ave { (end-start)/batch_size}')
        #logger.info(f' Threading efficiency is {ave*batch_size / (end-start)}')

    end_batching = time.monotonic()
    #logger.info(f' *********************************************** ')
    #logger.info(f' *********************************************** ')
    logger.info(f' \t\t overall time spent on last {num_batches} batches is {end_batching-start_batching} -- ave { (end_batching - start_batching)/num_batches}')

    # clean
    it = None
    dataset_train=None



def bench_dataloader(config, train_csv, img_folder, logger):
    #
    #      BENCH DATALOADER
    #

    num_workers = config.num_workers
    num_batches = config.num_batches
    batch_size = config.batch_size

    # dataset
    dataset_train = StackDataset(dataset=train_csv,
                                 img_folder=img_folder,
                                 logger=logger,
                                 use_gdal = config.use_gdal,
                                 use_PIL = config.use_pil,
                                 use_TIFF = config.use_tiff,
                                 use_RIO = config.use_rasterio)
    if config.use_concurrent_dataloader:
        dataloader = DataLoaderParallel(
            dataset=dataset_train,                              # standard parameters
            batch_size=config.batch_size,                         # ...
            num_workers=config.num_workers,
            shuffle=False,
            prefetch_factor=config.prefetch_factor,
            num_fetch_workers=config.num_workers, #args.num_fetch_workers,           # parallel threads used to load data
            fetch_impl='asyncio', #args.fetch_impl,                         # threads or asyncio
            batch_pool=None, #args.batch_pool,                         # only for threaded implementation (pool of pre-loaded batches)
            pin_memory=config.pin_memory #True if args.pin_memory == 1 else False, # if using fork, it must be 0
        )
    else:

        dataloader = DataLoader(
            dataset_train,
            shuffle=False,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            drop_last=True,
            #timeout=5,
            persistent_workers=config.persistent_workers,
            prefetch_factor=config.prefetch_factor,
            #worker_init_fn = worker_init_fn,
            pin_memory=config.pin_memory
        )
    print(f' Dataloader is {dataloader}')
        
        
    it2 = iter(dataloader)

    start_batching_torch = time.monotonic()

    logger.info(f' *********  XP WITH PYTORCH (DATALOADER) // DATA ACCESS ************************************** ')

    for batch in tqdm(range(num_batches)):
        #x,y = next(it2)
        x = next(it2)

    end_batching_torch = time.monotonic()

    #logger.info(f' *********************************************** ')
    #logger.info(f' *********************************************** ')
    logger.info(f' \t\t overall time spent on last {num_batches} batches is {end_batching_torch-start_batching_torch} -- ave { (end_batching_torch - start_batching_torch)/num_batches}')

    # clean
    #it2 = None
    #dataloader = None
    #dataset_train = None
    
    return num_workers, num_batches, end_batching_torch-start_batching_torch, (end_batching_torch-start_batching_torch)/num_batches

def run(config: Any):

    # SET NUMBER OF THREADS
    #torch.set_num_threads(config.num_workers)
    #cv2.setNumThreads(0)
    #cv2.ocl.setUseOpenCL(False)
    #torch.multiprocessing.set_start_method(config.spawn_method)
    
    if config.use_concurrent_dataloader:
        assert config.spaw_method == 'fork', f'Wrong spawn method for concurrent_dataloader'
        torch.multiprocessing.set_start_method('fork')
        torch.utils.data._utils.worker._worker_loop = _worker_loop_parallel
    # make a certain seed
    manual_seed(config.seed)

    # create output folder
    path = Path(config.output_dir)
    path.mkdir(parents=True, exist_ok=True)

    # read config file
    logger = setup_logging(config)

    logger.info("Configuration: \n%s", pformat(vars(config)))
    
    start = time.monotonic()
    train_csv = pd.read_csv(config.train_csv, sep=';', index_col=False)
    end = time.monotonic()
    logger.info(f' csv read time is {end - start}')

    start = time.monotonic()
    train_csv.rename(columns={train_csv.columns[0]: train_csv.columns[0].replace('#', '')}, inplace=True)
    end = time.monotonic()
    logger.info(f' csv rename time is {end - start}')

    img_folder = os.path.dirname(os.path.realpath(config.train_csv))

    stats = base_stats()
    
    for i in range(3):
        #bench_joblib(config, train_csv, img_folder, logger, use_gdal=False)
        stats.step(bench_dataloader(config, train_csv, img_folder, logger))
        #bench_joblib(config, train_csv, img_folder, logger, use_gdal=True)
        #bench_dataloader(config, train_csv, img_folder, logger, use_gdal=True)

    print(f'{stats.get()}')
        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config", help="Path to config file", type=str)
    parser.add_argument("-OF", "--output_dir", default="/tmp", help=" absolute path to the output folder", type=str, required=False)
    parser.add_argument("-BS", "--batch_size", default=16, help=" batch size ", type=int, required=False)
    parser.add_argument("-NW", "--num_workers", default=8, help=" num workers ", type=int, required=False)
    parser.add_argument("-NB", "--num_batches", default=10, help=" num batches to run xp for ", type=int, required=False)
    parser.add_argument("-PM", "--pin_memory", default=0, help=" use pin pemory 0 false anything else true ", type=int, required=False)

    parser.add_argument("-UPILL", "--use_pil", default=0, help=" use pil backend ? 0 false anything else true ", type=int, required=False)
    parser.add_argument("-UGDAL", "--use_gdal", default=0, help=" use gdal backend ? 0 false anything else true ", type=int, required=False)
    parser.add_argument("-URIO", "--use_rasterio", default=0, help=" use rasterio backend ? 0 false anything else true ", type=int, required=False)
    parser.add_argument("-UTIFF", "--use_tiff", default=0, help=" use TIFF backend ? 0 false anything else true ", type=int, required=False)
    parser.add_argument("-PF", "--prefetch_factor", default=2, help="  ", type=int, required=False)
    parser.add_argument("-UCD", "--use_concurrent_dataloader", default=0, help=" asyncio ? ", type=int, required=False)
    parser.add_argument("-PW", "--persistent_workers", default=0, help=" use persistent workers (pytorch dataloader) ", type=int, required=False)
    parser.add_argument("-SM", "--spawn_method", default='spawn', help=" how to start MT ? (spawn, fork, forkserver) ", type=str, required=False)  
    
    
    args = parser.parse_args()
    print(f' cli args are {args}')
    global config
    config = Config(args.config)
    config.output_dir = args.output_dir
    config.batch_size = args.batch_size
    config.num_workers = args.num_workers
    config.num_batches = args.num_batches
    config.pin_memory = not args.pin_memory == 0
    config.persistent_workers = not args.persistent_workers == 0
    config.spawn_method = args.spawn_method

    
    # only one at a time
    config.use_pil = False
    config.use_gdal = False
    config.use_rasterio = False
    config.use_tiff = False
    config.use_concurrent_dataloader = False

    # backend
    if not args.use_pil == 0:
        config.use_pil = not args.use_pil == 0
    elif not args.use_gdal == 0:
        config.use_gdal = not args.use_gdal == 0
    elif not args.use_rasterio == 0:
        config.use_rasterio = not args.use_rasterio == 0
    elif not args.use_tiff == 0:
        config.use_tiff = not args.use_tiff == 0
    else:
        config.use_tiff = True

    # use another dataloader
    if not args.use_concurrent_dataloader == 0:
        config.use_concurrent_dataloader  = not args.use_concurrent_dataloader == 0
        
    config.prefetch_factor = args.prefetch_factor

    print(f' Config is :\n{config}')

    torch.multiprocessing.set_start_method(config.spawn_method)
    
    run(config=config)



