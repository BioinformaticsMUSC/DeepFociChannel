""" Script to parse lif files """
import bioformats



def parse_lif_file(file_path:str, output_path:str):
    """ Function to parse a life file and write out
        all of the given data to the outputpath.
    """
    image, scale = bioformats.load_image(file_path, rescale = False,
                                         wants_max_intensity = False)
    plt.imshow(image)
    plt.show(image)
    plt.savefig(output_path)



parse_lif_file("test.lif", "test.png")
