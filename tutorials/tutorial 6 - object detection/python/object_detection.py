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

def draw_object_box_with_info(image, obj, zed_camera):
    """Draw enhanced bounding box with 3D information"""
    top_left = obj.bounding_box_2d[0]
    bottom_right = obj.bounding_box_2d[2]
    
    # Color based on tracking state
    if hasattr(obj, 'tracking_state'):
        if obj.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
            color = (0, 255, 0)  # Green for tracked
        elif obj.tracking_state == sl.OBJECT_TRACKING_STATE.SEARCHING:
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
    
    # Object label and confidence
    info_lines.append(f"Label: {obj.label}")
    info_lines.append(f"Conf: {int(obj.confidence)}%")
    
    # Tracking ID
    if hasattr(obj, 'id'):
        info_lines.append(f"ID: {int(obj.id)}")
    
    # 3D Position
    pos = obj.position
    info_lines.append(f"Pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]m")
    
    # Velocity
    vel = obj.velocity
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

def draw_object_segmentation(image, obj, alpha=0.4):
    """Display object segmentation mask as colored overlay within bounding box"""
    if not obj.mask.is_init():
        return image
    
    # Get mask data as numpy array
    mask_data = obj.mask.get_data()
    
    # Get bounding box coordinates
    top_left = obj.bounding_box_2d[0]
    bottom_right = obj.bounding_box_2d[2]
    x1, y1 = int(top_left[0]), int(top_left[1])
    x2, y2 = int(bottom_right[0]), int(bottom_right[1])
    
    # Ensure image is 3-channel BGR
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    
    # Resize mask to match bounding box dimensions
    bbox_height = y2 - y1
    bbox_width = x2 - x1
    mask_resized = cv2.resize(mask_data, (bbox_width, bbox_height), interpolation=cv2.INTER_NEAREST)
    
    # Create binary mask with adjustable threshold
    if mask_resized.max() <= 1.0:
        binary_mask = (mask_resized > 0.3).astype(np.uint8)
    else:
        binary_mask = (mask_resized.astype(np.uint8) > 100).astype(np.uint8)
    
    # Create colored overlay (blue for detected objects) - only in bounding box
    overlay = image.copy()
    overlay[y1:y2, x1:x2][binary_mask > 0] = [255, 0, 0]  # Blue: BGR format
    
    # Blend with original image
    output = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    return output

def main():
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.METER
    init_params.sdk_verbose = 1

    # Open the camera
    err = zed.open(init_params)
    if err > sl.ERROR_CODE.SUCCESS:
        print("Camera Open : "+repr(err)+". Exit program.")
        exit()

    obj_param = sl.ObjectDetectionParameters()
    obj_param.enable_tracking=True
    obj_param.enable_segmentation=True
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_MEDIUM

    if obj_param.enable_tracking :
        positional_tracking_param = sl.PositionalTrackingParameters()
        #positional_tracking_param.set_as_static = True
        zed.enable_positional_tracking(positional_tracking_param)

    print("Object Detection: Loading Module...")

    err = zed.enable_object_detection(obj_param)
    if err > sl.ERROR_CODE.SUCCESS :
        print("Enable object detection : "+repr(err)+". Exit program.")
        zed.close()
        exit()

    # Detection Output
    objects = sl.Objects()
    # Detection runtime parameters
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    obj_runtime_param.detection_confidence_threshold = 40
    zed.set_object_detection_runtime_parameters(obj_runtime_param) # can be set at any time

    cv2.namedWindow("ZED Object Detection", cv2.WINDOW_NORMAL)

    iter = 0
    while True:
        if zed.grab() <= sl.ERROR_CODE.SUCCESS:
            zed.retrieve_objects(objects)
            
            # Retrieve image for visualization
            img = sl.Mat()
            zed.retrieve_image(img, sl.VIEW.LEFT)
            img_cv = img.get_data()
            
            if objects.is_new :
                obj_array = objects.object_list
                print(str(len(obj_array))+" Object(s) detected ("+str(zed.get_current_fps())+" FPS)")
                if len(obj_array) > 0 :
                    first_object = obj_array[0]
                    print("First object attributes:")
                    print(" Label '"+repr(first_object.label)+"' (conf. "+str(int(first_object.confidence))+"/100)")
                    if obj_param.enable_tracking :
                        print(" Tracking ID: "+str(int(first_object.id))+" tracking state: "+repr(first_object.tracking_state)+" / "+repr(first_object.action_state))
                    position = first_object.position
                    velocity = first_object.velocity
                    dimensions = first_object.dimensions
                    print(" 3D position: [{0},{1},{2}]\n Velocity: [{3},{4},{5}]\n 3D dimentions: [{6},{7},{8}]".format(position[0],position[1],position[2],velocity[0],velocity[1],velocity[2],dimensions[0],dimensions[1],dimensions[2]))
                    if first_object.mask.is_init():
                        print(" 2D mask available")

                    print(" Bounding Box 2D ")
                    bounding_box_2d = first_object.bounding_box_2d
                    for it in bounding_box_2d :
                        print("    "+str(it),end='')
                    print("\n Bounding Box 3D ")
                    bounding_box = first_object.bounding_box
                    for it in bounding_box :
                        print("    "+str(it),end='')
                    print()
                    
                    # Draw all detected objects
                    for obj in obj_array:
                        draw_object_box_with_info(img_cv, obj, zed)
                        img_cv = draw_object_segmentation(img_cv, obj, alpha=0.3)
            
            # Display the image
            cv2.imshow("ZED Object Detection", img_cv)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        iter = iter + 1

    # Close the camera
    zed.disable_object_detection()
    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
