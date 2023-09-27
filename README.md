# opencv_image_saver

### How to build
1. Copy this file to your catkin_ws dir
2. run `catkin_make`
3. source catkin_ws/devel/setup.bash
4. Run `roslaunch opencv_image_saver image_saver.launch`

### How to use

The node constructor ImageSaver has 3 parameters
- `robot_name`: The name of the robot. If this is wrong, no image will be subscribed. _By default duck7_
- `use_for_data-gathering`: Enables a different file naming scheme._By default True_. 
- `store_location`: The place where to store all the images. _By default this value is <~userhome>/duckietown\_dataset_

**In the python file, there is an option to use this for a dataset, by default this value is True** 
- In this mode, enter 3 numbers in this format `[NUM_DUCKS] [NUM_ROBOTS] [NUM_CONES]`. The spaces are _VERY IMPORTANT_
- To save an image, hit space and then enter

This will save the image in the specified folder With the SUffixes DK for number of Ducks, RB for number of robots and CN for number of cones