#------------------------------------------------------#
# Import librairies
#------------------------------------------------------#

import datetime
import urllib
import time
import cv2 as cv
import streamlit as st
from imutils.video import FPS

from libraries.plugins import Motion_Detection
from libraries.utils import GUI, AppManager, DataManager

#------------------------------------------------------#
#------------------------------------------------------#

def imageWebApp(guiParam):
    """
    """
    # Load the image according to the selected option
    conf = DataManager(guiParam)
    image = conf.load_image_or_video()
    

    # GUI
    switchProcessing = st.button('* Start Processing *')

    # Apply the selected plugin on the image
    bboxed_frame, output = AppManager(guiParam).process(image, True)

    # Display results
    st.image(bboxed_frame, channels="BGR",  use_column_width=True)


#------------------------------------------------------#
#------------------------------------------------------#

# def videoWebApp(guiParam):
#     """
#     """
#     # Load the video according to the selected option
#     video = DataManager(guiParam).load_image_or_video()

#     # Define streamlit GUI parameters
#     switchProcessing = st.button('* Start Processing *')
#     imageHolder = st.empty()
#     ph1 = st.empty()
#     ph2 = st.empty()
#     ph3 = st.empty()

#     # Instantiate objects
#     display = DisplayResult(guiParam)
#     motion_detection = Motion_Detection(guiParam)
#     application = AppManager(guiParam)

#     # Processing each frame
#    #######################################################
#     hasFrame = True
#     frameIdx = 0
#     fps = FPS().start()
#     frameMax = 100

#     while hasFrame:
#         ph1.markdown(
#             ":arrows_counterclockwise: [In progress] Processing the first "+ str(frameMax) + " frames of the video   ...")

#         hasFrame, frame = video.read()

#         if not hasFrame or frameIdx == frameMax:
#             ph2.warning("No more frames to process.")
#             break

#         frameIdx += 1
#         (H, W) = frame.shape[:2]

#         # convert each frame from BGR to RGB
#         # frame = cv.cvtColor(src=frame, code=cv.COLOR_BGR2RGB)

#         # Detect if there is motion in the frames
#         __, output_MD = motion_detection.run(frame)
#         motion_state = output_MD['motion_state']

#         # Run the plugin for the selected application
#         bboxed_frame, output = application.process(
#             frame, motion_state)

#         # update the results dictionnary
#         output.update(dict(displayFlag=guiParam["displayFlag"],
#                            height=H,
#                            width=W,
#                            frameIdx=frameIdx,
#                            motion_state=motion_state,
#                            timeStamp=datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
#                            pixel_count=output_MD['pixel_count'],
#                            bboxed_frame=bboxed_frame))

#         # Display results
#         display.real_time_results(output)

#         # Update the FPS computation
#         fps.update()

#     # stop the timer and display FPS
#     fps.stop()
#     ph3.markdown(':rocket: Estimated FPS : {0:.2f} '.format(fps.fps()))

#     # Display the final result and save them to csv file
#     try:
#         display.dataframe_to_csv()
#         st.success("Done ! :thumbsup:")

#     except:
#         st.warning("Releasing the cached data...")
#         st.caching.clear_cache()
#         video.release()

#     st.info(
#         ":bulb: **Information**   \n\
#         - Press on **'Start Processing'** button  \n\
#         - Change any parameter from the sidebar if you want to (Re)run this application  \n\
#         - Select another application from the sidebar.")

#------------------------------------------------------#
#------------------------------------------------------#

def main():
    """
    """
    # Get the parameter entered by the user from the GUI
    guiParam = GUI().getGuiParameters()

    # Check if the application if it is Empty
    if guiParam['appType'] == 'Image Applications':
        if guiParam["selectedApp"] is not 'Empty':
            imageWebApp(guiParam)

        # elif guiParam['appType'] == 'Video Applications':
        #     videoWebApp(guiParam)
    else:
        raise st.ScriptRunner.StopException

#------------------------------------------------------#
#------------------------------------------------------#

if __name__ == "__main__":
    main()
