"""
Predefined colors for specific attribute values that can be used in fairness plots.

Currently, the following colormaps are defined:

===============================  =============  ========  ==============================================
Colormap                         Attribute      #Colors   Comment
===============================  =============  ========  ==============================================
``colormap_gender_telegraph``    Gender         3         See [1]_ for details. Extended with 3rd color
===============================  =============  ========  ==============================================

References
----------
.. [1] Lisa Charlotte Muth; An alternative to pink & blue: Colors for gender data;
   https://blog.datawrapper.de/gendercolor/

"""
from matplotlib.colors import ListedColormap

# Constants for specific values
GENDER_FEMALE = "female"
GENDER_MALE = "male"
GENDER_DIVERSE = "diverse"


class NamedColormap(ListedColormap):
    """
    Extension of the `ListedColormap` in which each color is identified by a name.

    This is especially useful, when certain colors are associated with a specific meaning.

    Parameters
    ----------
    colors : list, array
        Sequence of Matplotlib color specifications (color names or RGB(A)
        values).

    color_names: list
        The names of the colors. This allows to address colors by their name.

    name : str, optional
        String to identify the colormap.

    N : int, optional
        Number of entries in the map. The default is *None*, in which case
        there is one colormap entry for each element in the list of colors.
        If ::

            N < len(colors)

        the list will be truncated at *N*. If ::

            N > len(colors)

        the list will be extended by repetition.

    """
    def __init__(self, colors, color_names, name='named_from_list', N=None):
        self.color_names = color_names

        super().__init__(colors, name, N)

    def color_by_name(self, color_name):
        """
        Get the color by its name

        Parameters
        ----------
        color_name: str
            Name of the color. Must be one from the `color_names` list.

        Returns
        -------
        Tuple of RGBA values
        """
        try:
            idx = self.color_names.index(color_name)
        except ValueError:
            raise ValueError(f"Unknown color name '{color_name}'.")

        if idx >= self.N:
            raise ValueError(f"Color '{color_name}' out of bounds. This map only contains {self.N} colors.")

        return self(idx)

    def reorder(self, new_color_names, name=None, N=None):
        """
        Reorders the colormap.

        This is helpful, if groups are ordered differently from the predefined order.

        Parameters
        ----------
        new_color_names: list of str
            The new order of the colors, given by their names

        name: str, optional
            The Name of the reordered list. If None, "_reordered" will be appended to the current name.

        N: int, optional
            New number of colors. If not provided, the number of colors in `new_color_names` will be taken.
            This can differ from the length of the original colormap.

        Returns
        -------
        NamedColormap
            A reordered instance of the colormap.

        """
        if name is None:
            name = self.name + "_reordered"

        if N is None:
            N = len(new_color_names)

        colors_ro = [self.color_by_name(cn) for cn in new_color_names]
        new_cmap = NamedColormap(colors_ro, new_color_names, name=name, N=N)

        return new_cmap

    def reversed(self, name=None):
        """
        Return a reversed instance of the Colormap.

        Parameters
        ----------
        name : str, optional
            The name for the reversed colormap. If None, the
            name is set to ``self.name + "_r"``.

        Returns
        -------
        NamedColormap
            A reversed instance of the colormap.
        """
        if name is None:
            name = self.name + "_r"

        colors_r = list(reversed(self.colors))
        color_names_r = list(reversed(self.color_names))
        new_cmap = NamedColormap(colors_r, color_names_r, name=name, N=self.N)

        # Reverse the over/under values too
        new_cmap._rgba_over = self._rgba_under
        new_cmap._rgba_under = self._rgba_over
        new_cmap._rgba_bad = self._rgba_bad

        return new_cmap


colormap_gender_telegraph = NamedColormap(
    colors=['#933df5', '#1fc3aa', '#e69e22'],
    color_names=[GENDER_FEMALE, GENDER_MALE, GENDER_DIVERSE],
    name="gender_telegraph"
)