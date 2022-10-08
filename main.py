# ------------------------------------------------------#
# Import librairies
# ------------------------------------------------------#

import datetime
import urllib
import time
import cv2 as cv
import streamlit as st


from libraries.plugins import Motion_Detection
from libraries.utils import GUI, AppManager, DataManager

# ------------------------------------------------------#
# ------------------------------------------------------#


def imageWebApp(guiParam):
    """ """
    # Load the image according to the selected option
    conf = DataManager(guiParam)
    image = conf.load_image_or_video()

    # GUI
    switchProcessing = st.button("* Start Processing *")

    # Apply the selected plugin on the image
    bboxed_frame, output = AppManager(guiParam).process(image, True)

    # Display results
    st.image(bboxed_frame, channels="BGR", use_column_width=True)


def main():
    """ """
    # Get the parameter entered by the user from the GUI
    guiParam = GUI().getGuiParameters()

    # Check if the application if it is Empty
    if guiParam["appType"] == "Image Applications":
        if guiParam["selectedApp"] is not "Empty":
            imageWebApp(guiParam)

    else:
        raise st.ScriptRunner.StopException


# ------------------------------------------------------#
# ------------------------------------------------------#

if __name__ == "__main__":
    main()
