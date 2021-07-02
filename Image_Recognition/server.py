#!/usr/bin/env python3
"""
Very simple HTTP server in python for logging requests
Usage::
    ./server.py [<port>]
"""
from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import tensorflow as tf
import numpy as np
from util import labels, img_size, plot_spot, decode_base64
import sys
import base64
import cv2
import matplotlib.pyplot as plt
from urllib.parse import urlparse, parse_qs
import json


model = None;
class S(BaseHTTPRequestHandler):
    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        logging.info("GET request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers))
        self._set_response()
        self.wfile.write("GET request for {}".format(self.path).encode('utf-8'))

    def do_POST(self):
        x_test = []
       
        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        post_data = self.rfile.read(content_length) # <--- Gets the data itself

        images = decode_base64(json.loads(post_data.decode("utf-8"))["data"])
        
        prediction_matrices = model.predict(images)
        predictions = []
        for i, prediction in enumerate(prediction_matrices):

            predictions.append(str(np.argmax(prediction)))
            label = labels[np.argmax(prediction)]

        self._set_response()
        self.wfile.write(",".join(predictions).encode('utf-8'))
        

def run(server_class=HTTPServer, handler_class=S, port=8080):
    global model
    model = tf.keras.models.load_model("model.h5")
    logging.basicConfig(level=logging.INFO)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    logging.info('Starting httpd...\n')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logging.info('Stopping httpd...\n')

if __name__ == "__main__":
    
    run()