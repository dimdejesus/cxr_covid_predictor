import numpy as np

from skimage import io, filters, exposure
from skimage.transform import resize
from skimage import img_as_ubyte
from skimage.morphology import square, opening, closing, remove_small_objects, remove_small_holes, dilation, erosion

def auto_masking(img_file):
    length = 200
    
    try:
        img = img_as_ubyte(io.imread(img_file, as_gray=True))
        y_len, x_len = img.shape

        if y_len > x_len:    # height > width
            img = resize(img, [ round(length*y_len/x_len), length ])
            y_len = img.shape[0]
            crop_length = y_len - length
            img = img[ :y_len-crop_length , : ]    # crop from bottom

        elif y_len < x_len:    # height < width

            img = resize(img, [ length, round(length*x_len/y_len) ])
            x_len = img.shape[1]
            crop_length = x_len - length
            half_crop = crop_length//2
            img = img[ : , half_crop:x_len-half_crop ]    # crop on left and right

            if img.shape[0] != img.shape[1]:
                img = img[ : , :length ]    # make sure image is square

        else:    # height == width
            img = resize(img, [ length, length ])

        img = filters.gaussian(img)
    except FileNotFoundError:
        return "File Not Found"

    # CONTRAST STRETCHING + HISTOGRAM EQUALIZTIONA
    p2, p98 = np.percentile(img, (2, 98))
    img = exposure.rescale_intensity(img, in_range=(p2, p98))
    #img = exposure.equalize_hist(img)

    inverted = ~img_as_ubyte(img)
    thresh = filters.threshold_otsu(inverted)
    binary1 = inverted >= thresh

    #thresh = filters.threshold_multiotsu(img)
    #binary1 = ~(img >= thresh[0])

    kernel = square(7)
    op = opening(binary1, kernel)#select binary here
    op = dilation(remove_small_objects(erosion(op, kernel), 200), kernel)
    mask = closing(op, kernel)
    mask = remove_small_objects(mask, 1000)
    mask = remove_small_holes(mask, 1000)
    final1 = img * mask

    img_final = img_as_ubyte(final1)
    mask_image = img_as_ubyte(mask)

    return img_final, mask_image

