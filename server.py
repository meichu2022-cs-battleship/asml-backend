from tkinter import E
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import io
import base64
import numpy as np
import threading, time
import image
import mail

app = Flask(__name__)
CORS(app)


def parse_image(img_src, filename):

    np_image = base64.b64decode(img_src)
    image = np.asarray(bytearray(np_image), dtype="uint8")

    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    cv2.imwrite(filename, image)
    return image


"""
Transform to Javascript format.
"""


def pack_image(img):
    img_encoded = cv2.imencode(".png", img)
    base64_str = img_encoded[1].tobytes()
    img_encoded = base64.b64encode(base64_str)
    return img_encoded.decode("ASCII")


@app.route("/uploadImage/", methods=["GET", "POST"])
def test():
    try:
        result = request.get_json()
        origin_image = result["origin_image"]
        golden_image = result["golden_image"]
        user_name = result["user_name"]
        user_email = result["user_email"]
        user_comment = result["user_comment"]

        origin_image = parse_image(origin_image, "origin.png")
        golden_image = parse_image(golden_image, "golden.png")

        # ---> type: dict[7] ndarray: uint8
        all_images = image.process_images(origin_image, golden_image)
        final = []
        result = {}
        for key in all_images.keys():
            cv2.imwrite(f"{key}.png", all_images[key])
            result[key] = pack_image(all_images[key])
            if key == "r_rect_count":
                kneck = all_images[key]
            elif key == "g_rect_count":
                bridge = all_images[key]

        result_json = jsonify(result)
        print("kneck", kneck)
        print("bridge", bridge)

        # need to do summary
        if user_name and user_email:
            thread = threading.Thread(
                target=mail.sending_email,
                args=(user_name, user_email, user_comment, kneck, bridge),
            )
            thread.start()
        print("return")
        return result_json

    except Exception as e:
        print(e)
        return "error"


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
