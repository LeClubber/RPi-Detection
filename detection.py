#!/usr/bin/env python3

import argparse
import sys
from functools import lru_cache

import cv2
import libcamera
import numpy as np
import time

from datetime import datetime
from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics

# Imports pour Discord
import discord
import asyncio
import os
from dotenv import load_dotenv

last_detections = []
data_folder = f"./images"

class Detection:
    def __init__(self, coords, category, conf, metadata):
        """Create a Detection object, recording the bounding box, category and confidence."""
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)

def parse_detections(metadata: dict):
    """Parse the output tensor into a number of detected objects, scaled to the ISP output."""
    global last_detections
    threshold = 0.55

    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = imx500.get_input_size()
    if np_outputs is None:
        return last_detections
    boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
    boxes = np.array_split(boxes, 4, axis=1)
    boxes = zip(*boxes)

    last_detections = [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes)
        if score > threshold
    ]
    return last_detections

def draw_detections(request, stream="main"):
    """Draw the detections for this request onto the ISP output."""
    detections = last_results
    if detections is None:
        return
    labels = intrinsics.labels
    with MappedArray(request, stream) as m:
        for detection in detections:
            x, y, w, h = detection.box
            label = f"{labels[int(detection.category)]} ({detection.conf:.2f})"

            # Calculate text size and position
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x = x + 5
            text_y = y + 15

            # Create a copy of the array to draw the background with opacity
            overlay = m.array.copy()

            # Draw the background rectangle on the overlay
            cv2.rectangle(overlay,
                          (text_x, text_y - text_height),
                          (text_x + text_width, text_y + baseline),
                          (255, 255, 255),  # Background color (white)
                          cv2.FILLED)

            alpha = 0.30
            cv2.addWeighted(overlay, alpha, m.array, 1 - alpha, 0, m.array)

            # Draw text on top of the background
            cv2.putText(m.array, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Draw detection box
            cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0, 0), thickness=2)

        if intrinsics.preserve_aspect_ratio:
            b_x, b_y, b_w, b_h = imx500.get_roi_scaled(request)
            color = (255, 0, 0)  # red
            cv2.putText(m.array, "ROI", (b_x + 5, b_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.rectangle(m.array, (b_x, b_y), (b_x + b_w, b_y + b_h), (255, 0, 0, 0))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path of the model",
                        default="/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk")
    parser.add_argument("--wait", type=int, default=5, help="Time between 2 records")
    parser.add_argument("--discord_channel_id", type=int, help="Discord channel ID")
    parser.add_argument("--discord_bot_token", type=str, help="Discord bot token")
    return parser.parse_args()

def getTime():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")

async def send_image_to_discord(image_path, channel_id, bot_token):

    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        channel = client.get_channel(channel_id)
        if channel is not None:
            await channel.send(file=discord.File(image_path))
        await client.close()

    await client.start(bot_token)

if __name__ == "__main__":
    args = get_args()

    # This must be called before instantiation of Picamera2
    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "object detection"
    elif intrinsics.task != "object detection":
        print("Network is not an object detection task", file=sys.stderr)
        exit()

    # Defaults
    with open("labels.txt", "r") as f:
        intrinsics.labels = f.read().splitlines()
    intrinsics.update_with_defaults()

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(controls={"FrameRate": intrinsics.inference_rate}, buffer_count=12)
    # Si l'image est à l'envers, décommentez la ligne suivante
    config["transform"] = libcamera.Transform(hflip=1, vflip=1)

    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=False)

    last_results = None
    # A commenter si vous ne voulez pas le cadre de détection
    picam2.pre_callback = draw_detections
    
    # Chargement du .env
    load_dotenv()
    discord_channel_id = int(os.getenv('DISCORD_CHANNEL_ID'))
    discord_bot_token = os.getenv('DISCORD_BOT_TOKEN')
    
    while True:
        last_results = parse_detections(picam2.capture_metadata())
        if (len(last_results) > 0):
            for result in last_results:
                if result.category == 15: 
                    print("Bird detected !")
                    img_path = f"{data_folder}/bird_{getTime()}.jpg"
                    # Record file to SD card
                    picam2.capture_file(img_path)
                    # Send file to Discord
                    asyncio.run(send_image_to_discord(img_path, discord_channel_id, discord_bot_token))
                    # Wait 
                    time.sleep(args.wait)
