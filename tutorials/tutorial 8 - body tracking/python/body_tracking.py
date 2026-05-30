########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

import pyzed.sl as sl
import cv2
import numpy as np

def draw_skeleton(image, keypoints_2d, body_format=sl.BODY_FORMAT.BODY_18):
    """Draw skeleton lines connecting joints using ZED SDK skeleton format"""
    if keypoints_2d is None or len(keypoints_2d) == 0:
        return
    
    # Get the appropriate skeleton bones based on body format
    if body_format == sl.BODY_FORMAT.BODY_18:
        bones = sl.BODY_18_BONES
    elif body_format == sl.BODY_FORMAT.BODY_34:
        bones = sl.BODY_34_BONES
    elif body_format == sl.BODY_FORMAT.BODY_38:
        bones = sl.BODY_38_BONES
    else:
        bones = sl.BODY_18_BONES
    
    # Draw skeleton connections
    for bone in bones:
        # Bones are tuples of (start_enum, end_enum), convert to int values
        start_idx = bone[0].value
        end_idx = bone[1].value
        
        if start_idx < len(keypoints_2d) and end_idx < len(keypoints_2d):
            start = keypoints_2d[start_idx]
            end = keypoints_2d[end_idx]
            
            if not np.isnan(start[0]) and not np.isnan(end[0]):
                pt1 = (int(start[0]), int(start[1]))
                pt2 = (int(end[0]), int(end[1]))
                cv2.line(image, pt1, pt2, (0, 255, 0), 2)

def draw_keypoints(image, keypoints_2d):
    """Draw individual keypoints as circles"""
    if keypoints_2d is None or len(keypoints_2d) == 0:
        return
    
    for idx, kp in enumerate(keypoints_2d):
        if not np.isnan(kp[0]):
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            cv2.circle(image, (x, y), 5, (255, 0, 0), 2)

def draw_body_box_with_info(image, body, zed_camera):
    """Draw enhanced bounding box with 3D information"""
    top_left = body.bounding_box_2d[0]
    bottom_right = body.bounding_box_2d[2]
    
    # Color based on tracking state
    if hasattr(body, 'tracking_state'):
        if body.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
            color = (0, 255, 0)  # Green for tracked
        elif body.tracking_state == sl.OBJECT_TRACKING_STATE.SEARCHING:
            color = (0, 165, 255)  # Orange for searching
        else:
            color = (0, 0, 255)  # Red for off
    else:
        color = (0, 255, 0)
    
    # Draw bounding box
    cv2.rectangle(image, (int(top_left[0]), int(top_left[1])), 
                  (int(bottom_right[0]), int(bottom_right[1])), 
                  color, 2)
    
    # Prepare info text
    info_lines = []
    
    # Tracking ID
    if hasattr(body, 'id'):
        info_lines.append(f"ID: {int(body.id)}")
    
    # Confidence
    info_lines.append(f"Conf: {int(body.confidence)}%")
    
    # 3D Position
    pos = body.position
    info_lines.append(f"Pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]m")
    
    # Velocity
    vel = body.velocity
    speed = np.linalg.norm(vel)
    info_lines.append(f"Speed: {speed:.2f}m/s")
    
    # Draw info text with background
    y_offset = int(top_left[1]) - 10
    for i, line in enumerate(info_lines):
        y_pos = y_offset - (len(info_lines) - i) * 25
        
        # Text background
        text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(image, 
                     (int(top_left[0]), y_pos - text_size[1] - 5),
                     (int(top_left[0]) + text_size[0] + 10, y_pos + 5),
                     (0, 0, 0), -1)
        
        # Text
        cv2.putText(image, line, (int(top_left[0]) + 5, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def main():
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD720 video mode
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.METER
    init_params.sdk_verbose = 1

    # Open the camera
    err = zed.open(init_params)
    if err > sl.ERROR_CODE.SUCCESS:
        print("Camera Open : "+repr(err)+". Exit program.")
        exit()

    body_params = sl.BodyTrackingParameters()
    # Different model can be chosen, optimizing the runtime or the accuracy
    body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
    body_params.enable_tracking = True
    body_params.enable_segmentation = False
    # Optimize the person joints position, requires more computations
    body_params.enable_body_fitting = True
    body_params.body_format = sl.BODY_FORMAT.BODY_18

    if body_params.enable_tracking:
        positional_tracking_param = sl.PositionalTrackingParameters()
        # positional_tracking_param.set_as_static = True
        positional_tracking_param.set_floor_as_origin = True
        zed.enable_positional_tracking(positional_tracking_param)

    print("Body tracking: Loading Module...")

    err = zed.enable_body_tracking(body_params)
    if err > sl.ERROR_CODE.SUCCESS:
        print("Enable Body Tracking : "+repr(err)+". Exit program.")
        zed.close()
        exit()
    bodies = sl.Bodies()
    body_runtime_param = sl.BodyTrackingRuntimeParameters()
    # For outdoor scene or long range, the confidence should be lowered to avoid missing detections (~20-30)
    # For indoor scene or closer range, a higher confidence limits the risk of false positives and increase the precision (~50+)
    body_runtime_param.detection_confidence_threshold = 40
    i = 0 

    cv2.namedWindow("ZED", cv2.WINDOW_NORMAL)

    while True:
        if zed.grab() <= sl.ERROR_CODE.SUCCESS:
            err = zed.retrieve_bodies(bodies, body_runtime_param)

            img = sl.Mat()
            zed.retrieve_image(img, sl.VIEW.LEFT)
            img_cv = img.get_data()

            if bodies.is_new:
                body_array = bodies.body_list
                print(str(len(body_array)) + " Person(s) detected\n")
                if len(body_array) > 0:
                    first_body = body_array[0]
                    print("First Person attributes:")
                    print(" Confidence (" + str(int(first_body.confidence)) + "/100)")

                    if body_params.enable_tracking:
                        print(" Tracking ID: " + str(int(first_body.id)) + " tracking state: " + repr(
                            first_body.tracking_state) + " / " + repr(first_body.action_state))
                    position = first_body.position
                    velocity = first_body.velocity
                    dimensions = first_body.dimensions
                    print(" 3D position: [{0},{1},{2}]\n Velocity: [{3},{4},{5}]\n 3D dimentions: [{6},{7},{8}]".format(
                        position[0], position[1], position[2], velocity[0], velocity[1], velocity[2], dimensions[0],
                        dimensions[1], dimensions[2]))
                    if first_body.mask.is_init():
                        print(" 2D mask available")

                    print(" Keypoint 2D ")
                    keypoint_2d = first_body.keypoint_2d
                    for it in keypoint_2d:
                        print("    " + str(it))
                    print("\n Keypoint 3D ")
                    keypoint = first_body.keypoint
                    for it in keypoint:
                        print("    " + str(it))

                    # Enhanced visualization
                    draw_body_box_with_info(img_cv, first_body, zed)
                    draw_skeleton(img_cv, keypoint_2d, body_params.body_format)
                    draw_keypoints(img_cv, keypoint_2d)

            cv2.imshow("Body detection with ZED", img_cv)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        i+=1
    # Close the camera
    zed.disable_body_tracking()
    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
