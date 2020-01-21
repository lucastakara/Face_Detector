# Basic imports
import argparse
import cv2

from inference import Network


CPU_EXTENSION = "/opt/intel/openvino_2019.3.376/inference_engine/lib/intel64/libcpu_extension.dylib"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input webcam")

    # -- Create the descriptions for the commands -- #
    m_desc = "The location of the model XML file"
    d_desc = "The device name, if not CPU"
    c_desc = "The color of the bounding boxes to draw. RED, GREEN or BLUE"
    ct_desc = "The confidence threshold to use with the bounding boxes"
    t_desc = "Text which you may want to write above the bounding boxes"
    # -- Add required and optional groups -- #
    parser._action_groups.pop()
    required = parser.add_argument_group("required arguments")
    optional = parser.add_argument_group("optional arguments")

    # Create arguments
    required.add_argument("-m", help = m_desc, required = True)
    optional.add_argument("-d", help = d_desc, default = "CPU")
    optional.add_argument("-c", help = c_desc, default = "GREEN")
    optional.add_argument("-ct", help = ct_desc, default = 0.5)
    optional.add_argument("-t", help = t_desc, default = "")
    args = parser.parse_args()

    return args

def convert_color(color_string):
    '''
    Get the BGR value of the desired bounding box color.
    Defaults to GREEN if an invalid color is given.
    '''
    colors = {"BLUE":(255,0,0),"GREEN":(0,255,0),"RED":(0,0,255)}
    out_color = colors.get(color_string)
    if out_color:
        return out_color
    else:
        return colors["GREEN"]

def draw_boxes(frame, result, args, width, height):
    '''
    Draw bounding onto the frame.
    '''
    for box in result[0][0]:
        conf = box[2]
        if conf >= args.ct:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax,ymax), args.c, 1)
            cv2.putText(frame, args.t, (xmin,ymin),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2)
    return frame

def infer_on_camera(args):
    # Convert the args for color and confidence
    args.c = convert_color(args.c)
    args.ct = float(args.ct)
    # Initialize the Inference Engine
    plugin = Network()
    # Load the network model into the IE
    plugin.load_model(args.m,args.d,CPU_EXTENSION)
    # Get input shape
    net_input_shape = plugin.get_input_shape()
    # Get and open video capture
    cap = cv2.VideoCapture(0)
    cap.open(0) # 0 for default camera
    # Grab the shape of the input
    width = int(cap.get(3))
    height = int(cap.get(4))
    # Process frames until video end or process is exited
    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        # Preprocess the frame
        p_frame = preprocessing(frame,384,672)
       # Perform inference on the frame
        plugin.async_inference(p_frame)
        # Get the output of the inference
        if plugin.wait() == 0:
            result = plugin.extract_output()
            # Update the frame to include detected bounding boxes
            frame = draw_boxes(frame,result,args,width,height)
        cv2.imshow("frame", frame)

        if key_pressed == 27:
            break



def preprocessing(frame,height,width):
    p_frame = cv2.resize(frame, (width, height))
    p_frame = p_frame.transpose((2,0,1))
    p_frame = p_frame.reshape(1, *p_frame.shape)

    return p_frame

def main():
    args = get_args()
    infer_on_camera(args)
    # Release image
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
