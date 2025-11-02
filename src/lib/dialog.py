"""This module contains utility functions for dialog window selection."""

from pathlib import Path
from tkinter import Tk, filedialog


def dialog_select_directory(prompt: None | str = None) -> Path:
    """
    Prompt the user to select a directory in a dialog box.

    Parameters
    ----------
    prompt : None | str, optional
        dialog text prompt, by default None

    Returns
    -------
    str
        absolute path to the selected directory
    """
    root: Tk = Tk()
    selected: str = filedialog.askdirectory(title=prompt)
    root.destroy()
    return Path(selected).resolve()


def dialog_select_file(prompt: None | str = None) -> Path:
    """
    Prompt the user to select a file in a dialog box.

    Parameters
    ----------
    prompt : None | str, optional
        dialog text prompt, by default None

    Returns
    -------
    str
        absolute path to the selected file
    """
    root: Tk = Tk()
    selected: str = filedialog.askopenfilename(title=prompt)
    root.destroy()
    return Path(selected).resolve()
