"""
The following code is purely to plot multicoloured lines of text for plotting purposes.
"""

# draw multicoloured text
import matplotlib.pyplot as plt


def rainbow_text_lines(xy, strings, colors, orientation='horizontal',
                       ax=None, **kwargs):
    """
    *** NOT MY FUNCTION. ADAPTED FROM: https://matplotlib.org/stable/gallery/text_labels_and_annotations/rainbow_text.html ***
    *** USES ANNOTATE INSTEAD OF TEXT AND DELTA_X/Y UPDATE FOR TEXT IS APPLIED DIFFERENTLY. ***

    Take a list of *strings* and *colors* and place them next to each
    other, with text strings[i] being shown in colors[i]. Each entry
    is its own line.

    Parameters
    ----------
    xy : (float, float)
        Text position in data coordinates.

    strings : list of str
        The strings to draw.

    colors : list of color
        The colors to use.

    orientation : {'horizontal', 'vertical'}

    ax : Axes, optional
        The Axes to draw into. If None, the current axes will be used.

    **kwargs
        All other keyword arguments are passed to plt.text(), so you can
        set the font size, family, etc.
    """
    if ax is None:
        ax = plt.gca()
    t = ax.transData
    canvas = ax.figure.canvas

    assert orientation in ['horizontal', 'vertical']
    if orientation == 'vertical':
        kwargs.update(rotation=90, verticalalignment='bottom')

    for s, c in zip(strings, colors):
        text = ax.annotate(s + " ", xy, color=c, transform=t, **kwargs)

        # Need to draw to update the text position.
        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()
        bbox_width, bbox_height = ax.get_window_extent().width, ax.get_window_extent().height # box size of plot
        if orientation == 'horizontal':
            xy = [xy[0], xy[1]-ex.height/bbox_height]
        else:
            xy = [xy[0]+ex.width/bbox_width, xy[1]]
