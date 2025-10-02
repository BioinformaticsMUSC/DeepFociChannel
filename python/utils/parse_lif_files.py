""" Script to parse lif files """
from tqdm import tqdm
import readlif
from readlif.reader import LifFile
from tifffile import TiffWriter
import tifffile
import os
from bioio import BioImage
import bioio_lif
import numpy as np


def parse_lif_file(file_path:str, output_path:str):
    """ Function to parse a lif file and write out
        all of the given data to the outputpath.
    """

    if not os.path.exists("./tiffs"):
        os.mkdir("./tiffs")
    lif_file = BioImage(file_path,reader=bioio_lif.Reader)
    scenes = lif_file.scenes
    to_write = [x for x in scenes if "Processed" not in x]
    for scene in tqdm(to_write,
                      desc = "Writing tiff files..."):
        print(scene)
        lif_file.set_scene(scene)
        img = lif_file.data
        print(img.shape)
        img = np.transpose(img, (0, 2, 1, 3, 4))
        print(img.shape)
        with TiffWriter(os.path.join("./tiffs", scene + ".tiff"), bigtiff=True) as tif:
            tif.write(img,
                      metadata = {'axes':'TZCYX'})
    

parse_lif_file("npc_CPT_006.lif", "test.png")
