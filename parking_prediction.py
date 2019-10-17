import cv2
import json
import imutils
import requests
import numpy as np
from PIL import Image
from time import sleep
import opencv.colors as color
from keras.models import load_model
from opencv.base_camera import BaseCamera
from custom import LocalResponseNormalization

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

class Camera(BaseCamera):
    video_source = 0
    camara_address = 'rtsp://admin:123456@localhost/stream0'

    @staticmethod
    def set_video_source(source):
        Camera.video_source = 'rtsp://admin:123456@localhost/stream0'

    @staticmethod
    def frames():

        ######### Setting #########
        iterationCountInitializer = 52
        iterationCount = iterationCountInitializer
        saved_predictions = []

        # camera = cv2.VideoCapture('rtsp://192.168.0.23:554/12')
        camera = cv2.VideoCapture('rtsp://admin:123456@localhost/stream0')

        checkpoint_dir = 'trained_model/model_for_parking_cnn.h5'
        model = load_model(checkpoint_dir, custom_objects={'LocalResponseNormalization': LocalResponseNormalization})

        camera.set(cv2.CAP_PROP_POS_FRAMES, 0)

        coordinates = np.array([
            [[815, 72], [824, 138], [859, 130], [854, 67]],  # 52
            [[771, 72], [779, 140], [817, 135], [805, 73]],  # 53
            [[726, 74], [733, 145], [767, 140], [762, 75]],  # 54
            [[679, 75], [682, 148], [722, 140], [718, 78]],  # 55
            [[635, 83], [636, 143], [671, 142], [670, 79]],  # 56
            [[591, 79], [594, 149], [626, 146], [625, 80]],  # 57
            [[472, 96], [464, 159], [495, 157], [504, 92]],  # 58
            [[432, 97], [422, 162], [456, 159], [463, 94]],  # 59
            [[392, 101], [382, 167], [415, 165], [422, 101]],  # 60
            [[355, 109], [344, 172], [373, 170], [381, 105]],  # 61
            [[319, 114], [307, 177], [336, 174], [343, 111]],  # 62
            [[289, 119], [269, 182], [299, 179], [311, 116]],  # 63
            [[259, 121], [239, 186], [262, 182], [280, 120]],  # 64
            [[233, 128], [211, 190], [232, 188], [245, 129]],  # 65
            [[208, 137], [189, 193], [203, 191], [217, 137]],  # 66
            [[850, 282], [855, 389], [903, 389], [893, 275]],  # 67
            [[799, 281], [801, 393], [843, 391], [835, 283]],  # 68
            [[742, 288], [746, 397], [790, 393], [785, 283]],  # 69
            [[690, 285], [688, 399], [733, 394], [730, 286]],  # 70
            [[633, 290], [630, 398], [675, 397], [676, 288]],  # 71
            [[579, 290], [572, 399], [619, 397], [622, 290]],  # 72
            [[527, 291], [518, 394], [559, 395], [568, 293]],  # 73
            [[478, 295], [466, 394], [508, 394], [514, 292]],  # 74
            [[431, 293], [414, 395], [451, 395], [467, 295]],  # 75
            [[387, 299], [365, 393], [401, 396], [415, 298]],  # 76
            [[340, 298], [324, 391], [359, 393], [370, 301]],  # 77
            [[305, 299], [283, 390], [310, 388], [327, 299]],  # 78
            [[261, 302], [239, 389], [271, 389], [289, 305]],  # 79
            [[223, 302], [206, 385], [233, 385], [247, 302]],  # 80
            [[856, 407], [859, 523], [900, 523], [897, 407]],  # 81
            [[803, 412], [802, 528], [845, 526], [843, 410]],  # 82
            [[747, 414], [746, 528], [789, 529], [786, 412]],  # 83
            [[689, 413], [684, 533], [731, 531], [732, 412]],  # 84
            [[628, 412], [623, 533], [671, 532], [674, 410]],  # 85
            [[571, 414], [564, 532], [610, 533], [619, 413]],  # 86
            [[515, 410], [505, 529], [550, 530], [560, 412]],  # 87
            [[462, 412], [449, 525], [494, 527], [504, 410]],  # 88
            [[411, 410], [398, 523], [440, 526], [452, 410]],  # 89
            [[364, 406], [348, 518], [387, 520], [401, 410]],  # 90
            [[319, 406], [300, 513], [339, 514], [355, 406]],  # 91
            [[279, 403], [267, 503], [297, 506], [308, 406]],  # 92
            [[235, 401], [219, 496], [251, 503], [264, 402]],  # 93
            [[203, 400], [187, 493], [215, 493], [232, 399]]])  # 94

        coordinates_for_drawing = np.array([
            [[318, 352], [321, 367], [420, 338], [416, 330]],
            [[426, 321], [436, 335], [511, 303], [496, 291]],
            [[515, 284], [521, 295], [581, 268], [565, 257]],
            [[579, 253], [587, 261], [637, 240], [624, 229]],
            [[638, 225], [643, 232], [684, 215], [672, 204]],
            [[681, 201], [690, 212], [723, 197], [716, 186]],
            [[726, 182], [729, 193], [756, 182], [750, 170]],
            [[761, 170], [764, 178], [790, 173], [783, 162]],
            [[795, 161], [801, 171], [824, 168], [818, 157]],
            [[827, 158], [833, 167], [858, 165], [850, 155]],
            [[861, 158], [865, 164], [886, 165], [880, 154]]])

        coordinate_flag = False # Flag for using coordinates_for_drawing, True=Actiave, False=Deactive

        # print(coordinates.shape)

        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            _, img = camera.read()
            result, frame = camera.read()

            if result == False:
                print("Could not start camera")

                camera = cv2.VideoCapture('rtsp://admin:123456@localhost/stream0')
                camera.set(cv2.CAP_PROP_POS_FRAMES, 1)

                if not camera.isOpened():
                    print("Could not start camera")
                    continue
                    #raise RuntimeError('Could not start camera.')

                continue

            images = []
            positions = []

            im = Image.fromarray(frame)
            im_for_fullframe = Image.fromarray(frame)

            # width, height = im.size
            # print(str(width) + ", " + str(height))

            im_for_fullframe = im_for_fullframe.resize((1000, 625))
            im_for_fullframe = np.array(im_for_fullframe)

            im_origin = np.copy(im_for_fullframe)

            # Code for image rotation
            if(coordinate_flag == True):
                # 41 degree: Parking Site 6,
                im_for_fullframe = imutils.rotate_bound(im_for_fullframe, -71)
                im_for_fullframe = np.array(im_for_fullframe)
                cv2.imwrite("rotate_image_for_car.jpg", im_for_fullframe)

            ### For routine for image boundry selection

            x1 = 0
            x1_1 = 0
            y1 = 0
            y1_1 = 0
            i = 0

            while i < len(coordinates):

                ## x1
                if coordinates[i][0][0] - coordinates[i][1][0] > 0:
                    x1 = coordinates[i][1][0]
                elif coordinates[i][0][0] - coordinates[i][1][0] < 0:
                    x1 = coordinates[i][0][0]
                else:
                    x1 = coordinates[i][0][0]

                ## x1_1
                if coordinates[i][2][0] - coordinates[i][3][0] > 0:
                    x1_1 = coordinates[i][2][0]
                elif coordinates[i][2][0] - coordinates[i][3][0] < 0:
                    x1_1 = coordinates[i][3][0]
                else:
                    x1_1 = coordinates[i][2][0]

                ## y1
                if coordinates[i][0][1] - coordinates[i][3][1] > 0:
                    y1 = coordinates[i][3][1]
                elif coordinates[i][0][1] - coordinates[i][3][1] < 0:
                    y1 = coordinates[i][0][1]
                else:
                    y1 = coordinates[i][0][1]

                ## y1_1
                if coordinates[i][1][1] - coordinates[i][2][1] > 0:
                    y1_1 = coordinates[i][1][1]
                elif coordinates[i][1][1] - coordinates[i][2][1] < 0:
                    y1_1 = coordinates[i][2][1]
                else:
                    y1_1 = coordinates[i][1][1]

                im_ = Image.fromarray(im_for_fullframe[y1:y1_1, x1:x1_1])



                ## Code for checking selected images
                selected_image = np.array(im_)
                b_degree = np.ones(selected_image.shape, dtype="uint8") * 40
                selected_image = cv2.add(selected_image, b_degree)
                #cv2.imwrite("selected_image_" + str(i) + ".jpg", selected_image)

                im_ = Image.fromarray(selected_image)
                im_ = im_.resize((54, 32))
                im_ = np.array(im_)
                im_ = im_.transpose(1, 0, 2)

                images.append(im_)
                #cv2.imwrite("resized_selected_image_" + str(i) + ".jpg", im_)

                i = i + 1

            # Summarized the position for the prediction
            images = np.array(images)

            predictions = model.predict(images, verbose=1)

            predictions = np.hstack(predictions < 0.5).astype(int)

            # Procedures for uploading the prediction result to oneM2M Server (Mobius)

            # url_base = "http://localhost:7579/Mobius/iotParking/parkingSpot/"
            url_base = "http://localhost:7599/wdc_base/sync_parking_raw/parkingLot_KETI/"

            if len(saved_predictions) == 0: # Saving the initial parking lot status
                saved_predictions = predictions

                list_predictions = saved_predictions.tolist()

                for prediction_results_index in list_predictions:

                    if iterationCount < 10:
                        containerName = "parkingSpot_00" + str(iterationCount)
                    elif iterationCount < 100:
                        containerName = "parkingSpot_0" + str(iterationCount)
                    else:
                        containerName = "parkingSpot_" + str(iterationCount)

                    # Converting the results: 0 -> free, 1 -> occupied
                    result_converting = ""
                    if prediction_results_index == 0:
                        result_converting = "free"
                    else:
                        result_converting = "occupied"

                    url = url_base + containerName
                    payload = "{\r\n    \"m2m:cin\": {\r\n    \"con\":" + '"{}"'.format(result_converting) + "\r\n}\r\n}"
                    headers = {
                        'accept': "application/json",
                        'x-m2m-ri': "jaeyoung62590",
                        'x-m2m-origin': "SM",
                        'content-type': "application/json; ty=4"
                    }

                    print(url)
                    print(payload)
                    response = requests.request("POST", url, data=payload, headers=headers)
                    print(containerName + " has been saved: " + str(response))

                    iterationCount = iterationCount + 1

            elif np.array_equal(saved_predictions, predictions) == False: # Comparing the values between previous prediction results and current prediction results

                    result_comparison = (saved_predictions == predictions)
                    saved_predictions = predictions

                    list_predictions = predictions.tolist()

                    predctionValueCoutner = 0 # This variable is used to get the predicted value

                    for result_comparison_index in result_comparison:
                        if result_comparison_index == False:
                            if iterationCount < 10:
                                containerName = "parkingSpot_00" + str(iterationCount)
                            elif iterationCount < 100:
                                containerName = "parkingSpot_0" + str(iterationCount)
                            else:
                                containerName = "parkingSpot_" + str(iterationCount)

                            # Converting the results: 0 -> free, 1 -> occupied
                            result_converting = ""
                            if list_predictions[predctionValueCoutner] == 0:
                                result_converting = "free"
                            else:
                                result_converting = "occupied"

                            url = url_base + containerName
                            payload = "{\r\n    \"m2m:cin\": {\r\n    \"con\":" + '"{}"'.format(result_converting) + "\r\n}\r\n}"

                            headers = {
                                'accept': "application/json",
                                'x-m2m-ri': "jaeyoung62590",
                                'x-m2m-origin': "SM",
                                'content-type': "application/json; ty=4"
                            }

                            print(url)
                            print(payload)
                            response = requests.request("POST", url, data=payload, headers=headers)
                            print(containerName + " has been updated: " + str(response))

                        predctionValueCoutner = predctionValueCoutner + 1
                        iterationCount = iterationCount + 1

            iterationCount = iterationCountInitializer

            ### Drawing the predicted images ###
            position_counting = 0

            for x in predictions:
                partial_coordinate = []

                if(coordinate_flag == True):
                    partial_coordinate = coordinates_for_drawing[position_counting]
                else:
                    partial_coordinate = coordinates[position_counting]

                if (x == 1):
                    cv2.drawContours(im_origin, [partial_coordinate], 0, color=color.COLOR_RED, thickness=2)
                elif (x == 0):
                    cv2.drawContours(im_origin, [partial_coordinate], 0, color=color.COLOR_GREEN, thickness=2)

                position_counting += 1

            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', im_origin)[1].tobytes()
