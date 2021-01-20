import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


'''
To Call this method:

Run ipython:
import sunpy.io.fits
from flicker import flicker
data, header = sunpy.io.fits.read('sunspot.fits')[1]
flicker(data[0], data[1])
'''


def flicker(image1, image2, rate=1, animation_path=None):

    image1 = image1.copy()

    image2 = image2.copy()

    image1 = image1 / np.nanmax(image1)

    image2 = image2 / np.nanmax(image2)

    final_image_1 = np.zeros(
        shape=(
            max(
                image1.shape[0],
                image2.shape[0]
            ),
            max(
                image1.shape[1],
                image2.shape[1]
            )
        )
    )

    final_image_2 = np.zeros(
        shape=(
            max(
                image1.shape[0],
                image2.shape[0]
            ),
            max(
                image1.shape[1],
                image2.shape[1]
            )
        )
    )

    final_image_1[0: image1.shape[0], 0: image1.shape[1]] = image1

    final_image_2[0: image2.shape[0], 0: image2.shape[1]] = image2

    imagelist = [final_image_1, final_image_2]

    rate = rate * 1000

    fig = plt.figure()  # make figure

    im = plt.imshow(
        imagelist[0],
        origin='lower',
        cmap='gray',
        interpolation='none'
    )

    # function to update figure
    def updatefig(j):
        # set the data in the axesimage object
        im.set_array(imagelist[j])
        # return the artists set
        return [im]
    # kick off the animation
    ani = animation.FuncAnimation(
        fig,
        updatefig,
        frames=range(2),
        interval=rate, blit=True
    )

    if animation_path:
        Writer = animation.writers['ffmpeg']

        writer = Writer(
            fps=1,
            metadata=dict(artist='Me'),
            bitrate=1800
        )

        ani.save(animation_path, writer=writer)

    else:
        plt.show()
